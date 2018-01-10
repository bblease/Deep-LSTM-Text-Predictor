/*******************************************************************************
 * Name        : main.cpp
 * Author      : Ben Blease
 * Date        : 9/26/17
 * Description : Run LSTM with user input
 ******************************************************************************/

#include <fstream>
#include <string.h>
#include "core.h"

using namespace std;

/*
default values, assuming text input
produces a 3 layer-deep network with 95 dimensional one-hot vectors
BPTT block is set to 100 timesteps
memory blocks consist of 64 cells (a relatively simple network) for quick training
64 cells allows for learning word lengths, spacing, and some output
*/
#define DEFAULT_HIDDEN_SIZE 64
#define DEFAULT_LAYER_SIZE 3
#define DEFAULT_INPUT_SIZE 95
#define DEFAULT_BLOCK_SIZE 100
#define DEFAULT_OUTPUT_SIZE 50
#define DEFAULT_READ_SIZE -1 //read the whole file
#define DEFAULT_LIMIT 100000
#define DEFAULT_LEARN 0.1
#define DEFAULT_LAMBDA 0.01

//paths and names
#define DEFAULT_Y_PATH "./output/"
#define DEFAULT_Y_NAME "y.txt"
#define DEFAULT_X_PATH "./input/"
#define DEFAULT_X_NAME "x.txt"
#define DEFAULT_SAVE_PATH "./saves/"
#define SAVE_NAME(l, s, b, n) "net"#l"_"#s"_"#b"_"#n".bin"

//behavior
#define SAVE_TRAINING true
#define RUN_TRAINED true
#define DEBUG false

/*
Read a large input file for use with the network
*/
string* read_input(string fname, string* in_pointer){
  ifstream infile;
  infile.open(fname);
  string ln;
  while(getline(infile, ln)){
    (*in_pointer) += "\n" + ln;
  }
  return in_pointer;
}

int main(int argc, char** argv){
  //use fallback values
  string in;
  if (argc == 1){
    read_input(DEFAULT_X_PATH DEFAULT_X_NAME, &in);
    Net* rnn = new Net(&in, DEFAULT_LAYER_SIZE, DEFAULT_INPUT_SIZE, DEFAULT_HIDDEN_SIZE, DEFAULT_BLOCK_SIZE);

    write_net(rnn, DEFAULT_SAVE_PATH SAVE_NAME(3, 95, 100, 64));
    try{
      rnn->train(DEFAULT_LEARN, DEFAULT_LAMBDA, DEFAULT_LIMIT);
    } catch(runtime_error& e){
      cerr << e.what() << endl;
      cerr << "This is usually caused by misrepresenting the dimensionality of your data" << endl;
    }
    cout << endl;
    rnn->run(50, "The");
  }    
}
