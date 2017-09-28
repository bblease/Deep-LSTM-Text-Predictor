/*******************************************************************************
 * Name        : main.cpp
 * Author      : Ben Blease
 * Date        : 9/26/17
 * Description : Run LSTM
 ******************************************************************************/

#include <fstream>
#include "core.h"

using namespace std;

int main(int argc, char** argv){
  ifstream infile;
  infile.open("./texts/y.txt");
  string in;  
  string ln;
  while(getline(infile, ln)){
    in += "\n" + ln;
  }

  Net* rnn = new Net(&in, 2, 95, 512, 100);

  try{
    rnn->train(0.1, 0.0, 10000000);
  } catch(invalid_argument& e){
    cerr << e.what() << endl;
  }
  
  cout << endl;
  rnn->run(50, "The").length();
}