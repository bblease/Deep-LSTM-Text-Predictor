/*******************************************************************************
 * Name        : serialized.h
 * Author      : Ben Blease
 * Date        : 9/26/17
 * Description : Serialized operations for vectors and matrices
 ******************************************************************************/

#ifndef SERIALIZED_H_
#define SERIALIZED_H_

#include <vector>
#include <iostream>
#include <utility>

//vect.cpp
void print_vector(const std::vector<double>&);

double operator*(const std::vector<double>&, const std::vector<double>&);

double v_sum(const std::vector<double>&);

std::vector<double> operator*(const std::vector<double>&, double);

std::vector<double> operator/(const std::vector<double>&, double);

//override the addition operator
std::vector<double> operator+(const std::vector<double>&, const std::vector<double>&);

std::vector<double> operator-(const std::vector<double>&, const std::vector<double>&);

std::vector<double> h_prod(const std::vector<double>&, const std::vector<double>&);

std::vector<double> activate(const std::vector<double>&, double(*)(const double&));

/* A 2d Matrix */
template <class T>
class Matrix {
public:
	std::vector<T> operator*(const std::vector<T>&);

	Matrix<T> operator*(double);

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

	/* Randomize the values of the current matrix */
	void randomize();

	/* Generate a matrix from an already existing 2d vector */
	Matrix(const std::vector<std::vector<T> >&);

	/* Generate a matrix of 0s from given sizes */
	Matrix(int, int);

	/* Generate a matrix of values of type T from given sizes */
	Matrix(int, int, T);

	Matrix();

private:
	int _x; //columns
	int _y; //rows
	std::vector<std::vector<T> > _v;
};



#endif /* serialized.h */