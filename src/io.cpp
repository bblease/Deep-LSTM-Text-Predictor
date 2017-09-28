/*******************************************************************************
 * Name        : io.cpp
 * Author      : Ben Blease
 * Date        : 9/26/17
 * Description : Create input/ouput for machine
 ******************************************************************************/


#include "core.h"
#include <random>
#include <algorithm>

using namespace std;

/* Turn a relevant ASCII character into a one-hot vector */
vector<double> vectorize(char c){
  vector<double> out;
  for (int i = 32; i < 127; i++){
  	if (i == c)
  		out.push_back(1.0);
  	else
  		out.push_back(0.0);
  }
  return out;
}

/* Do a weighted random pick from probabilities */
char pick_char(const vector<double>& v){
  discrete_distribution<int> d = discrete_distribution<int>(begin(v), end(v));
  random_device r;
  mt19937 gen(r());
  int out = d(gen);
  return (char) (out + 32);
}

char max_pick_char(const vector<double>& v){
	int index = distance(v.begin(), max_element(v.begin(), v.end()));
	return (char) (index + 32);
}



