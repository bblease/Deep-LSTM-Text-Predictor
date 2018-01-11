/*******************************************************************************
 * Name        : core.h
 * Author      : Ben Blease
 * Date        : 9/27/16
 * Description : Vector and Matrix operations and structures
 * Note: not an exhausting implementation: header only includes functions needed for
   core functionality
 ******************************************************************************/

#ifndef SERIALIZED_H_
#define SERIALIZED_H_

#include <vector>
#include <iostream>
#include <utility>

//./serialized/vect.cpp
//./parallelized/vect.cu
void print_vector(const std::vector<double>&);

/* Dot product of two vectors of the same shape */
double operator*(const std::vector<double>&, const std::vector<double>&);

/* Sum of a vector */
double v_sum(const std::vector<double>&);

/* Multiply a vector by a scalar */
std::vector<double> operator*(const std::vector<double>&, double);

/* Divide a vector by a scalar */
std::vector<double> operator/(const std::vector<double>&, double);

/* Add two vectors of the same shape */
std::vector<double> operator+(const std::vector<double>&, const std::vector<double>&);

/* Subtract two vectors of the same shape */
std::vector<double> operator-(const std::vector<double>&, const std::vector<double>&);

/* Take the hadamard product of two vectors of the same shape */
std::vector<double> h_prod(const std::vector<double>&, const std::vector<double>&);

/* Activate a vector using the presented function */
std::vector<double> activate(const std::vector<double>&, double(*)(const double&));

//./serialized/matrix.cpp
//./parallelized/matrix.cu
/* A 2d Matrix */
template <class T>
class Matrix {
public:
	std::vector<T> operator*(const std::vector<T>&);

	Matrix<T> operator*(double);

	Matrix<T> operator/(const Matrix<T>&);

	Matrix<T> operator+(const Matrix<T>&);

	Matrix<T> operator-(const Matrix<T>&);

	/* Take the hadamard product of the current and another matrix */
	Matrix<T> h_prod(const Matrix<T>&);
	
	/* Sum all rows in the matrix */
	std::vector<T> mat_sum_x();

	/* Sum all columns in the matrix */
	std::vector<T> mat_sum_y();

	std::vector<T> mult_x(const std::vector<T>&);

	/* Return the x/y size of the matrix */
	std::pair<int, int> size();

	inline int total_size(){ return _x * _y; }

	inline std::vector<std::vector<T> > get_v() { return _v; }

	/* Randomize the values of the current matrix */
	void randomize();

	/* Generate a matrix from an already existing 2d vector */
	Matrix(const std::vector<std::vector<T> >&);

	/* Generate a matrix of 0s from given sizes */
	Matrix(int, int);

	Matrix(int, int, T);

	Matrix();

private:
	int _x; //columns
	int _y; //rows
	std::vector<std::vector<T> > _v;
};



#endif /* serialized.h */