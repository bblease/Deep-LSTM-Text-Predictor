/*******************************************************************************
 * Name        : rw.cpp
 * Author      : Ben Blease
 * Date        : 10/11/17
 * Description : Read/Write save files (WIP)
 ******************************************************************************/

#include "core.h"
#include <fstream>

using namespace std;

/*
Write the network information to a binary file
*/
void write_net(Net* net, string fname){
	ofstream file;
	file.open(fname, ios::out | ios::binary);

	size_t l = net->layer_num;
	size_t s = net->inp_size;
	size_t n = net->node_num;
	int b = net->block_size;

	//write the basic information
	//first four positions are non-weight information
	file.seekp(0);
	file.write((char*) &l, sizeof (size_t));
	file.write((char*) &s, sizeof (size_t));
	file.write((char*) &n, sizeof (size_t));
	file.write((char*) &b, sizeof (int));
}


Net* read_file(string fname){
	ifstream file (fname, ios::in | ios::binary);
	return NULL;
}

