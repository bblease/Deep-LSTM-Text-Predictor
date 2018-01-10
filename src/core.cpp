/*******************************************************************************
 * Name        : core.cpp
 * Author      : Ben Blease
 * Date        : 9/26/17
 * Description : LSTM Core structures and functions
 ******************************************************************************/

#include <iostream>
#include <fstream>
#include <math.h>
#include <unistd.h>
#include <string>
#include <random>
#include "core.h"

using namespace std;

/* Activation functions */
double sigmoid(const double& x){
  return 1/(1 + exp(-1 * x));
}

double deriv_sigmoid(const double& x){
  return (1 - sigmoid(x)) * sigmoid(x);
}

double deriv_tanh(const double& x){
  return 1 - pow(tanh(x), 2);
}

double tanh(const double& x){
  return sinh(x) / cosh(x);
}

double ReLu(const double& x){
  return max(x, 0.0);
}

//2D vectors passed to constructors are not matricies. 
//Instead, they provide z i f o information for indeces 0 - 3 
TimeStep::TimeStep(vector<vector<double> > g, 
                   vector<vector<double> > i, 
                   vector<double> o, 
                   vector<double> st, 
                   int n): 
                   gates(g), 
                   inputs(i), 
                   output(o), 
                   state(st) { 
  dels = vector<vector<double> >(5, vector<double>(n, 0.0));
}

void TimeStep::set_delts(vector<double> z, 
           vector<double> i, 
           vector<double> f, 
           vector<double> o, 
           vector<double> c){
  dels[Z] = z;
  dels[I] = i;
  dels[F] = f;
  dels[O] = o;
  dels[C] = c;
}

void TimeRange::push(int l, TimeStep t){
  if (q[l].size() == max_size){
    q[l].pop_front();
  }
  q[l].push_back(t);
}

/* Take the outer product of vectors */
Matrix<double> outer(const vector<double>& a, const vector<double>& b){
  vector<vector<double> > out;
  for (size_t i = 0; i < a.size(); i++)
    out.push_back(b * a[i]);
  return Matrix<double>(out);
}

/*
Calculate the delta resulting from the output (dE/dyt)
Returns an n x s vector
*/
Matrix<double> Output::calc_delta(vector<double> y){
  vector<double> out_err = o - y;
  return outer(x, out_err);
}

/* 
Produce output from the network
*/
void Output::forward(TimeRange* t_store, vector<double>* y){

  vector<double> out = vector<double>(inp_size, 0.0);
  Matrix<double> xt = w.h_prod(outer(x, vector<double>(inp_size, 1.0)));
  vector<double> weighted = xt.mat_sum_y();
  out = out + weighted + b;
  //use the softmax for output 
  //TODO potentially fix for clarity
  double weighted_sum;
  for (size_t i = 0; i < weighted.size(); i++)
    weighted_sum += exp(out[i]);
  for (size_t i = 0; i < weighted.size(); i++)
    out[i] = abs(exp(out[i]) / weighted_sum);

  o = out;
  //the delta of the final 
  if (y && t_store)
    t_store->back(t_store->q.size() - 1)->del_x = (calc_delta(*y).mat_sum_x()) / inp_size;

}

/* 
Generate the gradient dE/dyt 
*/
void Output::backprop(vector<double> y, double rate, double lambda){
  //calculate weight delta
  Matrix<double> delt = calc_delta(y);

  //update weights
  w = w - (delt - (w * lambda)) * rate;
  b = b - (o - y) * rate;
}

Output::Output(size_t s, size_t n): inp_size(s), block_num(n), w(Matrix<double>(s, n)), b(vector<double>(s, 1.0)) {
  w.randomize();
}

Output::~Output() { }


/*
Feed specific cell forward
x is input at time step t
return the output for the memory cell
*/
void Block::forward(TimeRange* t_store, vector<double>* y){
  vector<double> inp_z = (w[3] * x) + (u[3] * h) + b[3];
  vector<double> inp_i = (w[1] * x) + (u[1] * h) + b[1];
  vector<double> inp_f = (w[0] * x) + (u[0] * h) + b[0];
  vector<double> inp_o = (w[2] * x) + (u[2] * h) + b[2];

  vector<double> z_gate = activate(inp_z, &tanh);
  vector<double> i_gate = activate(inp_i, &sigmoid);
  vector<double> f_gate = activate(inp_f, &sigmoid);
  vector<double> o_gate = activate(inp_o, &sigmoid);

  state = h_prod(f_gate, state_prev) + h_prod(i_gate, z_gate);
  vector<double> out = h_prod(o_gate, activate(state, &sigmoid));

  //chain timesteps together
  h = out;
  state_prev = state;

  //create the timestep and include relevant data
  //if the pass is a training run
  if (t_store){

    vector<vector<double> > gates = {z_gate, i_gate, f_gate, o_gate};
    vector<vector<double> > inputs = {inp_z, inp_i, inp_f, inp_o};
    t_store->push(id, TimeStep(gates, inputs, out, state, block_num));

  }

  //pass to output node
  //potentially chained blocks
  if (next){
    next->x = out;
    next->forward(t_store, y);
  } 
  else if (out_node) {
    out_node->x = out;
    out_node->forward(t_store, y);
  } else {
    cerr << "Network formatted incorrectly" << endl;
  }
}

/* 
Perform BPTT 
Uses t as a record of previous values in the timeseries
h - the output of the block in the forward pass
err - the gradient calculated from the previous layer
*/
void Block::backprop(TimeRange* t_store, int block_size, double rate, double lambda){

  //calculate gate and output deltas
  for (int t = t_store->size(id) - 2; t >= 0; t--){
    TimeStep* curr = t_store->get(id, t);
    TimeStep* prev = t_store->get(id, t - 1);
    TimeStep* next = t_store->get(id, t + 1); 
    TimeStep* prev_layer = (id > 0) ? t_store->get(id - 1, t) : NULL;

    vector<double> del_y = (curr->del_x
                            + (u[0] * next->dels[F])
                            + (u[1] * next->dels[I])
                            + (u[2] * next->dels[O])
                            + (u[3] * next->dels[Z]))
                            / block_size;


    vector<double> del_o = h_prod(h_prod(del_y, activate(curr->state, &tanh)), activate(curr->inputs[O], &deriv_sigmoid)) / block_size;
    vector<double> del_c = (h_prod(del_y, h_prod(del_o, activate(curr->state, &deriv_tanh)))
                                 + h_prod(next->dels[C], next->gates[F])) / block_size;
    vector<double> del_f = h_prod(del_c, h_prod(prev->state, activate(curr->inputs[F], &deriv_sigmoid))) / block_size;
    vector<double> del_i = h_prod(del_c, h_prod(curr->gates[Z], activate(curr->inputs[I], &deriv_sigmoid))) / block_size;
    vector<double> del_z = h_prod(del_c, h_prod(curr->gates[I], activate(curr->inputs[Z], &deriv_tanh))) / block_size;

    if (prev_layer){
      vector<double> inp_del = (w[0].mult_x(del_f)
                             + w[1].mult_x(del_i)
                             + w[2].mult_x(del_o)
                             + w[3].mult_x(del_z))
                             / block_size;
      prev_layer->del_x = inp_del;
    }
    
                    
    //set the new values in the timestep
    curr->set_delts(del_z, del_i, del_f, del_o, del_c);
  }


  //calculate and apply deltas for final weights
  for (int i = 0; i < 4; i++){
    Matrix<double> del_w = Matrix<double>(inp_size, block_num, 0.0);
    Matrix<double> del_u = Matrix<double>(block_num, block_num, 0.0);
    vector<double> del_b = vector<double>(block_num, 0.0);

    //calculate
    for (int t = 0; t < t_store->size(id); t++){
      TimeStep* curr = t_store->get(id, t);

      del_w = del_w + outer(curr->dels[i], x);
      
      //if the current timestep is the last in the BPTT block, use the output delta
      if (t < t_store->size(id) - 1)
        del_u = del_u + outer(curr->dels[i], curr->output);
      del_b = del_b + curr->dels[i]; 
    }

    //apply
    w[i] = w[i] - (del_w * (rate)) - w[i] * (rate * lambda); 
    u[i] = u[i] - (del_u * (rate)) - u[i] * (rate * lambda); 
    b[i] = b[i] - del_b * (rate);
  }

  t_store->clear(id);
}

/*
n - size of the hidden layer of which this block is a current member
s - dimensionality of input vectors
*/
Block::Block(int i, 
      size_t s, 
      size_t n): 
      id(i),
      inp_size(s), 
      block_num(n),
      h(vector<double>(n, 0.0)),
      state(vector<double>(n, 0.0)),
      state_prev(vector<double>(n, 0.0)),
      out_node(NULL),
      next(NULL) {
  
  //initialize weights
  for(int i = 0; i < 4; i++){
    w[i] = Matrix<double>(s, n);
    u[i] = Matrix<double>(n, n);
    b[i] = vector<double>(n, 0.0);
  }

  for (int i = 0; i < 4; i++)
    w[i].randomize();

  //initialize u
  for (int i = 0; i < 4; i++)
    u[i].randomize();  }

Block::~Block() {
  delete next;
}



/*
 Pass the current time input to the hidden layer
*/
void Input::forward(TimeRange* t_store, vector<double> xt, vector<double>* y){
  next->x = xt;
  next->forward(t_store, y);
}

Input::Input(): next(NULL) { }

Input::~Input() {
  delete next;
}
/*
  Process:
    1. Feed information up to block_size
    2. When block_size is reached, backpropagate through the current TimeRange structure
*/
void Net::train(double rate, double lambda, int limit){
  cout << "Training . . . " << endl;
  for(size_t i = 0; (i < data->length() && i < limit); i++){
    //feed intial character vector into the network
    vector<double> curr = vectorize((*data)[i]);
    vector<double> curr_p1 = vectorize((*data)[i + 1]);
    input->forward(time_vals, curr, &curr_p1);

    //backpropagate throughout the deep layers
    if (i % (int) block_size == 0 && i != 0){
      output->backprop(curr, rate, lambda);
      for (int k = block.size() - 1; k >= 0; --k){
        block[k]->backprop(time_vals, block_size, rate, lambda);   
      }
    }

    //print only occasionally to avoid slowdowns
    if (i % 100 == 0){
     cout << "\r" << i;
     cout << " " << ((double) i / limit) * 100.0 << "%";
     fflush(stdout);
    }
  }

  trained = true;
}


string Net::run(size_t length, string s){
  if (!trained){
    cerr << "The network hasn't been trained yet." << endl;
    return "";
  }
  string out = s;
  vector<double> curr = vectorize(s[s.length() - 1]);
  //feed the starting string through the network, ignoring output
  input->forward(NULL, vectorize(s[0]), NULL);
  for(size_t i = 1; i < s.length() - 1; i++){
    input->forward(NULL, vectorize(s[i]), NULL);
    print_vector(output->o);
  }

  while(length-- > 0){
    input->forward(NULL, curr, NULL);
    
    char next = max_pick_char(output->o);
    curr = vectorize(next);
    out += next;
  }
  cout << out << endl;
  return out;
}

/*
Create the network of an arbitrary number of blocks
l - the number of deep layers
s - dimensionality of input
n - the number of hidden cells
b - the number of timesteps for BPTT
*/
Net::Net(string* i, 
         size_t l, 
         size_t s, 
         size_t n, 
         int b): 
         layer_num(l),
         inp_size(s),
         node_num(n),
         block_size(b){
  data = i;
  time_vals = new TimeRange(l, b);
  input = new Input();
  output = new Output(s, n);

  //set up chained block layers
  for (size_t k = 0; k < l; k++){
    //only the first block has the input size of the network input
    int block_inp_size = (k == 0) ? s : n;
    Block* curr_block = new Block(k, block_inp_size, n);
    block.push_back(curr_block); 
    if (k > 0)
      block[k - 1]->next = curr_block;
    if (k == l - 1)
      curr_block->out_node = output;
  }

  input->next = block[0];
}

Net::~Net() {
  delete time_vals;
  delete input;
  for (Block* b : block)
    delete b;
  delete output;
  delete data;
}
