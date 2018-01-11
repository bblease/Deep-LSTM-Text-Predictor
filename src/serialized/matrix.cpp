/*******************************************************************************
 * Name        : matrix.cpp
 * Author      : Ben Blease
 * Date        : 9/26/17
 * Description : Implementations of Matrix methods
 ******************************************************************************/

#include <stdexcept>
#include "serialized.h"

template <class T>
std::vector<T> Matrix<T>::operator*(const std::vector<T>& b){
	std::vector<T> out;
	for (int i = 0; i < _y; i++){
		out.push_back(_v[i] * b);
	}
	return out;
}

template <class T>
Matrix<T> Matrix<T>::operator*(double b){
	Matrix<T> out = Matrix(_v);
	for (int i = 0; i < _y; i++){
		out._v[i] = out._v[i] * b;
	}
	return out;
}

template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& b){
	if (_x != b._x 
		|| _y != b._y){
		throw std::runtime_error("Addition matrices are not aligned");
		return *this;
	}

	Matrix<T> out = Matrix<T>(_v);
	for (int i = 0; i < _y; i++)
		out._v[i] = out._v[i] + b._v[i];
	return out;
}

template <class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& b){
	if (_x != b._x 
		|| _y != b._y){
		throw std::runtime_error("Subtraction matrices are not aligned");
		return *this;
	}

	Matrix<T> out = Matrix<T>(_v);
	for (int i = 0; i < _y; i++)
		out._v[i] = out._v[i] - b._v[i];
	return out;
}

template <class T>
Matrix<T> Matrix<T>::h_prod(const Matrix<T>& b){
	if (_x != b._x 
		|| _y != b._y){
		throw std::runtime_error("Hadamard matrices are not aligned");
		return *this;
	}
	
	Matrix<T> out = Matrix<T>(_x, _y);
	for (int i = 0; i < _y; i++)
    	for (int j = 0; j < _x; j++)
    		out._v[i][j] = _v[i][j] * b._v[i][j];
  	return out;
}

template <class T>
std::vector<T> Matrix<T>::mat_sum_x(){
	std::vector<T> out;
	for (int i = 0; i < _y; i++){
	    T sum;
	    for (int j = 0; j < _x; j++)
	      sum += _v[i][j];
	    out.push_back(sum);
  	}
  return out;
}

template <class T>
std::vector<T> Matrix<T>::mat_sum_y(){
	std::vector<T> out = std::vector<T>(_x, 0.0);
	for (int i = 0; i < _y; i++){
	    out = out + _v[i];
  	}
  return out;
}

template <class T>
std::vector<T> Matrix<T>::mult_x(const std::vector<T>& b){
	std::vector<T> out = this->mat_sum_x(); 
	for (int i = 0; i < out.size(); i++){
		out[i] *= b[i];
	}
	return out;
}

template <class T>
std::pair<int, int> Matrix<T>::size(){
	return std::pair<int, int>(_x, _y);
}

template <class T>
void Matrix<T>::randomize(){
	for (int i = 0; i < _y; i++)
      for (int j  = 0; j < _x; j++)
        _v[i][j] = ((double) rand() / RAND_MAX) * 2 - 1;
}

template <class T>
void Matrix<T>::print_matrix(){
	print_vector(_v[0]);
	std::cout << "..." << std::endl;
	print_vector(_v[_v.size() - 1]);
}

template <class T>
Matrix<T>::Matrix(const std::vector<std::vector<T> >& v): _v(v), _y(v.size()), _x(v[0].size()) { }

template <class T>
Matrix<T>::Matrix(int x, int y): _v(std::vector<std::vector<T> >(y, std::vector<T>(x))), _x(x), _y(y) { }

template <class T>
Matrix<T>::Matrix(int x, int y, T val): _v(std::vector<std::vector<T> >(y, std::vector<T>(x, val))), _x(x), _y(y) { }

template <class T>
Matrix<T>::Matrix() { }

//declare potential templated usage
template class Matrix<double>;