/*******************************************************************************
 * Name        : core.h
 * Author      : Ben Blease
 * Date        : 6/12/16
 * Description : Core functions and structures
 ******************************************************************************/

#ifndef CORE_H_
#define CORE_H_

#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include "serialized.h"

//rnn.cpp
enum Gates{
	Z = 0, //block input
	I, //input
	F, //forget
	O, //output
	C //state
};

/* 
Store information of a specific time step
Stores:
  - Gate input
  - Gate output
  - Deltas
  - Delta from next layer
*/
struct TimeStep{
  //z i f o
  //n dimensional
  std::vector<std::vector<double> > gates;
  std::vector<std::vector<double> > inputs;
  std::vector<double> output; //output at the current time step
  std::vector<double> state;
  std::vector<std::vector<double> > dels; //z i f o c
  std::vector<double> del_x;
 

  void set_delts(std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>);

  TimeStep(std::vector<std::vector<double> >, std::vector<std::vector<double> >, std::vector<double>, std::vector<double>, int);

  ~TimeStep() { }

};

/*
Store all needed information about a time step and the current machine state for backpropagation through time
Stores deques for an arbitrary number of layers
*/
struct TimeRange{
  int max_size;
  std::vector<std::deque<TimeStep> > q;

  /* Push a timestep onto the deque */
  void push(int, TimeStep);

  /* Return the pointer of a timestep at index i */
  inline TimeStep* get (int l, int i){
    return &(q[l][i]);
  }

  /* Return the pointer of the last timestep */
  inline TimeStep* back (int l){
    return &(q[l].back());
  }

  inline void clear(int l){
    q[l].clear();
  }

  inline int size(int l){
  	return q[l].size();
  }

  TimeRange(int l_num, int s): max_size(s), q(std::vector<std::deque<TimeStep> >(l_num)) { }

  ~TimeRange() { 
    for(std::deque<TimeStep> d : q)
    	d.clear();
  }
};

/*
Technically d output nodes
Condensed for readability and memory
In the case of ASCII char, size of 128
*/
struct Output {
  size_t inp_size;
  size_t block_num;

  std::vector<double> x;
  Matrix<double> w;
  std::vector<double> b;
  std::vector<double> o;

  Matrix<double> calc_delta(std::vector<double>);

  void forward(TimeRange*, std::vector<double>*);

  void backprop(std::vector<double>, double, double);

  Output(size_t, size_t);

  ~Output();
};

/*
 LSTM memory block encapsulating and modifying a memory cell
 TODO allow for multiple chained memory blocks
 - Contain all block data as vectors; should only be minor changes
*/
struct Block {
  int id;
  Output* out_node;
  Block* next;

  size_t inp_size;
  size_t block_num;

  //sideways shift
  std::vector<double> x; //central input
  std::vector<double> h; //input from t - 1 block
  std::vector<double> state;
  std::vector<double> state_prev;
  
  //f, i, o, z
  Matrix<double> w[4]; //input weight (N x M)
  Matrix<double> u[4]; //chaining weight (N x N)
  std::vector<double> b[4]; //block biases

  void forward(TimeRange*, std::vector<double>*);

  void backprop(TimeRange*, int, double, double);

  Block(int, size_t, size_t);

  ~Block();
};

/*
 Encapsulates all 95 nodes for ASCII characters
*/
struct Input {
  Block* next;

  void forward(TimeRange*, std::vector<double>, std::vector<double>*);

  Input();

  ~Input();
};

/*
Recurrent Neural Network
*/
struct Net {
  TimeRange* time_vals; //network information for each time step
  Input* input; //input provides entrance into network
  std::vector<Block*> block; //blocks can be chained
  Output* output;
  std::string* data;

  //network information
  size_t layer_num;
  size_t inp_size;
  size_t node_num;

  int block_size;
  bool trained;

  /*
  Train the network on the input data
  */
  void train(double, double, int);

  /*
  Run the trained network
  */
  std::string run(size_t, std::string);

  Net(std::string*, size_t, size_t, size_t, int);

  ~Net();
};

//io.cpp
std::vector<double> vectorize(char);

char pick_char(const std::vector<double>&);

char max_pick_char(const std::vector<double>&);

void write_net(Net*, std::string);

Net* read_net(std::string);

#endif /* core.h */
