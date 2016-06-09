#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "vli/integer.hpp"
#include "vli/polynomial.hpp"
#include "vli/vector.hpp"

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
using namespace std;

typedef vli::integer<128> integer_type_cpu_128; 
// Replace 14 with the maximum exponent one wants to use.
typedef vli::polynomial<integer_type_cpu_128, vli::max_order_each<14>, vli::var<'x'>, vli::var<'y'>, vli::var<'z'> >polynomial_type_each_x_128;
      
typedef vli::vector<polynomial_type_each_x_128> polynomial_operand;
typedef vli::polynomial_multiply_result_type<polynomial_type_each_x_128>::type polynomial_res;

void InitPolynomial(polynomial_operand& vp, int order, int dim, int who) {
  ofstream myfile;
  myfile.open ("pol" + who + order + dim);

  for (int a = 0; a < dim; ++a) {
    polynomial_type_each_x_128 p;
    int i = std::rand() % (order + 1);
    int j = std::rand() % (order + 1);
    int k = std::rand() % (order + 1);
    p(i, j, k) = std::rand();
    vp.push_back(p); 
    myfile << i << " " << j << " " << k << " " << p(i, j, k) << endl;
  }
  myfile << "pol" << endl;
  myfile << vp << endl;
  myfile.close();
}

void ReadPolynomial(polynomial_operand& vp, char* name, int dim) {
  ifstream myfile;
  myfile.open (name);
  int i, j, k, coeff;
  for (int a = 0; a < dim; ++a) {
    polynomial_type_each_x_128 p;
    myfile >> i >> j >> k >> coeff;  
    p(i, j, k) = coeff;
    vp.push_back(p); 
  }
  myfile.close();
}


int main(int argc, char * argv[]){
    polynomial_operand pA,pB;
    int order = atoi(argv[1]);          
    int dim = atoi(argv[2]);
    //InitPolynomial(pA, order, dim, 1);
    //InitPolynomial(pB, order, dim, 2);
    ReadPolynomial(pA, argv[3], dim);
    ReadPolynomial(pB, argv[4], dim);
 
    cudaError_t error;   
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    polynomial_res p_res;
                                      
    cudaEventRecord(start, NULL); 
    p_res = vli::inner_product(pA,pB);
    error = cudaEventRecord(stop, NULL);
    error = cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);
    std::cout << "Time= " <<  msecTotal << std::endl;
		
    //std::cout << p_res << std::endl;
}
