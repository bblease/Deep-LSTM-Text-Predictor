/*
 Vector function implementations
*/

#include "serialized.h"

/* Print a vector */
void print_vector(const std::vector<double>& v){
  std::cout << "[";
  for (int i = 0; i < v.size(); i++){
    std::cout << v[i];
    if (i != v.size() - 1)
      std::cout << " ";
  }
  std::cout << "]" << std::endl;
}

/* Vector product of a and b */
double operator*(const std::vector<double>& a, const std::vector<double>& b){
  double out = 0.0;
  if (a.size() != b.size()){
    throw std::runtime_error("Multiplication vectors not aligned ");
    return 0.0;
  }
  for (int i = 0; i < a.size(); i++)
    out += a[i] * b[i];
  return out;
}

/* Take the sum of a vector */
double v_sum(std::vector<double> v){
  double out;
  for (int i = 0; i < v.size(); i++)
    out += v[i];
  return out;
}

/* Multiply vector by double */
std::vector<double> operator*(const std::vector<double>& a, double b){
  std::vector<double> out;
  for (int i = 0; i < a.size(); i++)
    out.push_back(a[i] * b);
  return out;
}

std::vector<double> operator/(const std::vector<double>& a, double b){
  std::vector<double> out;
  for (int i = 0; i < a.size(); i++)
    out.push_back(a[i] / b);
  return out;
}

/* Add two vectors elementwise */
std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b){
  std::vector<double> out;
  if (a.size() != b.size()){
    throw std::runtime_error("Addition vectors not aligned ");
    return out;
  }
  for (int i = 0; i < a.size(); i++)
    out.push_back(a[i] + b[i]);
  return out;
}

std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b){
  std::vector<double> out;
  if (a.size() != b.size()){
    throw std::runtime_error("Subtraction vectors not aligned ");
    return out;
  }
  for (int i = 0; i < a.size(); i++)
    out.push_back(a[i] - b[i]);
  return out;
}

std::vector<double> h_prod(const std::vector<double>& a, const std::vector<double>& b){
  std::vector<double> out;
  if (a.size() != b.size()){
    throw std::runtime_error("Hadamard Product vectors not aligned");
    return out; //return an empty vector
  }
  for (int i = 0; i < a.size(); i++)
    out.push_back(a[i] * b[i]);
  return out;
}

/* Apply an activation function on a vector */
std::vector<double> activate(const std::vector<double>& x, double(*f)(const double&)){
  std::vector<double> out;
  for (int i = 0; i < x.size(); i++){
    out.push_back(f(x[i]));
  }
  return out;
}



// /* Multiply a 2d matrix and a vector */
// vector<double> operator*(const vector<vector<double> >& a, const vector<double>& b){
//   vector<double> out;
//   for (int i = 0; i < a.size(); i++){
//     out.push_back(a[i] * b);
//   }
//   return out;
// }

// /* Multiply a 2s matrix and a double */
// vector<vector<double> > operator*(const vector<vector<double> >& a, double b){
//   vector<vector<double> > out;
//   for (int i = 0; i < a.size(); i++){
//     out.push_back(a[i] * b);
//   }
//   return out;
// }

// /* Subtract two 2d matrices */
// vector<vector<double> > operator-(const vector<vector<double> >& a, const vector<vector<double> >& b){
//   vector<vector<double> > out;
//   if (a.size() != b.size()){
//     throw invalid_argument("Subtraction vectors not aligned ");
//     return out;
//   }
//   for (int i = 0; i < a.size(); i++)
//     out.push_back(a[i] - b[i]);
//   return out;
// }

// vector<vector<double> > operator+(const vector<vector<double> >& a, const vector<vector<double> >& b){
//   vector<vector<double> > out;
//   if (a.size() != b.size()){
//     throw invalid_argument("Addition vectors not aligned ");
//     return out;
//   }
//   for (int i = 0; i < a.size(); i++)
//     out.push_back(a[i] + b[i]);
//   return out;
// }

// vector<vector<double> > h_prod(const vector<vector<double> >& a, const vector<vector<double> >& b){
//   vector<vector<double> > out;
//   if (a.size() != b.size()){
//     throw invalid_argument("Hadamard product vectors not aligned");
//     return out; //return an empty vector
//   }
//   for (int i = 0; i < a.size(); i++)
//     out.push_back(h_prod(a[i], b[i]));
//   return out;
// }



// /* Take the sum of all rows of a matrix */
// vector<double> mat_sum(const vector<vector<double> >& a){
//   vector<double> out;
//   for (int i = 0; i < a.size(); i++){
//     double sum;
//     for (int j = 0; j < a[i].size(); j++)
//       sum += a[i][j];
//     out.push_back(sum);
//   }
//   return out;
// }

// /* Transpose a matrix */
// vector<vector<double> > transpose(const vector<vector<double> >& a){
//   vector<vector<double> > out;
//   for (int i = 0; i < a[0].size(); i++){
//     vector<double> curr;
//     for (int j = 0; j < a.size(); j++){
//       curr.push_back(a[j][i]);
//     }
//   out.push_back(curr);
//   }
//   return out;
// }

// vector<double> mult_x(const vector<vector<double> >& a, const vector<double>& b){
//   return h_prod(mat_sum(a), b);
// }
