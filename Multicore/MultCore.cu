/**
 * Copyright 2013-2016 Diana-Andreea Popescu, University of Cambridge.  All rights reserved.
 *
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

#define MAX_EXP	100

inline
void checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
	exit(EXIT_FAILURE);
  }
}

struct is_order_less
{
	__host__ __device__
	bool operator() (const int x)
	{
		return (x < 0);
	}
};

void initPol(unsigned int *exps, double* coeffs, FILE* f, int dim) {
  unsigned int exp[3];
  double coeff;
  for (unsigned int a = 0; a < dim; ++a) {
    fscanf(f, "%u %u %u %lf", &exp[0], &exp[1], &exp[2], &coeff);
    for (unsigned int b = 0; b < 3; ++b) {
      exps[a + dim * b] = exp[b];
    } 
    coeffs[a] = coeff;   
  }
}

/**
 * Run a multivariate polynomial multiplication using CUDA
 */
int polynomMultiply(int argc, char **argv, unsigned int &dimA, unsigned int &dimB, 
		    unsigned int &order, unsigned int &nvars, FILE *fA, FILE* fB)
{
    // Allocate host memory for polynoms A and B
    unsigned int size_A = dimA * nvars;
    unsigned int mem_size_exp_A = sizeof(unsigned int) * size_A;
    unsigned int mem_size_coeff_A = sizeof(double) * dimA;
    unsigned int *exp_A = (unsigned int*) malloc(mem_size_exp_A);
   
    double *coeff_A = (double*) malloc(mem_size_coeff_A);
   
    unsigned int size_B = dimB * nvars;
    unsigned int mem_size_exp_B = sizeof(unsigned int) * size_B;
    unsigned int mem_size_coeff_B = sizeof(double) * dimB;
    unsigned int *exp_B = (unsigned int*) malloc(mem_size_exp_B);
    double *coeff_B = (double*)malloc(mem_size_coeff_B);

    // Initialize host memory
    initPol(exp_A, coeff_A, fA, dimA);
    initPol(exp_B, coeff_B, fB, dimB);

    // Allocate device memory
    double *final_coeff_C;
    unsigned long long *e_keys_C;
    unsigned long long *final_keys_C;

    // Allocate host polynomial C
    unsigned int dimC = dimA * dimB;
    unsigned int size_C = dimA * dimB * nvars;
    unsigned int mem_size_exp_C = size_C * sizeof(unsigned int);
    unsigned int *exp_C = (unsigned int*)malloc(mem_size_exp_C);

    unsigned int mem_size_keys_C = dimC * sizeof(unsigned long long); 
    unsigned int mem_size_coeff_C = sizeof(double) * dimC;
    double *coeff_C = (double*)malloc(mem_size_coeff_C);

    final_coeff_C = (double*)malloc(mem_size_coeff_C);

    e_keys_C = (unsigned long long*)malloc(mem_size_keys_C);

    final_keys_C = (unsigned long long*)malloc(mem_size_keys_C);

    //STENCIL FOR TRUNCATION - to be used in remove_if function if one wants to truncate the polynomial up to a certain order
    int *stencil = NULL;
    unsigned int mem_size_stencil = sizeof(int) * dimC;
    stencil = (int*)malloc(mem_size_stencil);
    
    printf("Computing result ...\n");

    int nIter = 1;
    unsigned long long ekey = 0, kd = 0;
    unsigned int cexp = 0;
    for (int it = 0; it < nIter; it++)
    {
      double start1 = omp_get_wtime();
#pragma omp parallel for shared(exp_A, exp_B, exp_C, coeff_A, coeff_B, coeff_C) firstprivate(ekey, cexp) schedule(static)
      for (int i = 0; i < dimB; i ++)
	for (int j = 0; j < dimA; j ++) {
	  coeff_C[i * dimA + j] = coeff_A[j] * coeff_B[i];
	  for (int k = 0; k < nvars; k ++){
	    cexp = exp_A[j + k * dimA] + exp_B[i + k * dimB];
	    exp_C[i * dimA + j + k * dimC] = cexp;
	    ekey = MAX_EXP * ekey + cexp;
	  }
	  e_keys_C[i * dimA + j] = ekey;
	
	}
      double end1 = omp_get_wtime();
      printf("Compute result terms=%lf\n", 1000 * (end1 - start1));
      thrust::device_vector<unsigned long long> keys_C_dev(e_keys_C, e_keys_C + dimC);
      thrust::device_vector<double> coeff_C_dev(coeff_C, coeff_C + dimC);
      thrust::device_vector<unsigned long long> final_keys_C_dev(final_keys_C, final_keys_C + dimC);
      thrust::device_vector<double> final_coeff_C_dev(final_coeff_C, final_coeff_C + dimC);
      double start2 = omp_get_wtime();
      thrust::sort_by_key(keys_C_dev.begin(), keys_C_dev.end(), coeff_C_dev.begin());
      double end2 = omp_get_wtime();
      printf("Sort time=%lf\n", 1000 * (end2 - start2));    
      double start3 = omp_get_wtime();
      thrust::pair<thrust::device_vector<unsigned long long>::iterator, thrust::device_vector<double>::iterator > end;
      end = thrust::reduce_by_key(keys_C_dev.begin(), keys_C_dev.end(), coeff_C_dev.begin(), final_keys_C_dev.begin(), final_coeff_C_dev.begin());
      int sizeC = end.first - final_keys_C_dev.begin();
      double end3 = omp_get_wtime();
      printf("Reduce time=%lf\n", 1000 * (end3 - start3)); 
      double start4 = omp_get_wtime();
#pragma omp parallel for private(kd, ekey) shared(exp_C, final_keys_C_dev, sizeC) schedule(static)
      for (int i = 0; i < sizeC; i ++){
		ekey = final_keys_C_dev[i];
		for (int k = nvars - 1; k >= 0; k--) {
		  	kd = ekey/MAX_EXP;
			exp_C[i + k * dimC] = ekey - kd * MAX_EXP; 
			ekey = kd;
		}
       }
       double end4 = omp_get_wtime();
       printf("Extract Keys time=%lf\n", 1000 * (end4 - start4));
       printf("TOTAL=%lf\n", 1000 * (end4 - start1)); 
 	
    }

    printf("Checking for errors: OK ");
    
    // Clean up memory
    free(exp_A);
    free(exp_B);
    free(exp_C);
    free(coeff_A);
    free(coeff_B);
    free(coeff_C);

    free(e_keys_C);
    free(final_keys_C);
    free(final_coeff_C);
    free(stencil);

    return EXIT_SUCCESS; 
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Multivariate Polynomial Multiplication Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -nA=NumberOfTermsA (Number of terms of polynomial A)\n");
        printf("      -nB=NumberOfTermsB (Number of terms of polynomial B)\n");
        printf("      -x=vars (Number of variables)\n");
	printf("      -o=order (Order of polynoms).\n");

        exit(EXIT_SUCCESS);
    }

    unsigned int dimA, dimB; 
    // number of terms of polynomial A
    if (checkCmdLineFlag(argc, (const char **)argv, "nA"))
    {
        dimA = getCmdLineArgumentInt(argc, (const char **)argv, "nA");
    }

    // number of terms of polynomial B
    if (checkCmdLineFlag(argc, (const char **)argv, "nB"))
    {
        dimB = getCmdLineArgumentInt(argc, (const char **)argv, "nB");
    }

    unsigned int order = 6;
    // Order of polynomials
    if (checkCmdLineFlag(argc, (const char **)argv, "o"))
    {
        order = getCmdLineArgumentInt(argc, (const char **)argv, "o");
    }

    unsigned int nvars = 6;
    // Number of variables
    if (checkCmdLineFlag(argc, (const char **)argv, "x"))
    {
		nvars = getCmdLineArgumentInt(argc, (const char **)argv, "x");
    }

    char *value;
    FILE* fA = NULL;
    // File pol A
    if (checkCmdLineFlag(argc, (const char **)argv, "fA"))
    {
       getCmdLineArgumentString(argc, (const char **)argv, "fA", &value);
       fA = fopen(value, "r");
    }
    FILE* fB = NULL;
    // File pol B
    if (checkCmdLineFlag(argc, (const char **)argv, "fB"))
    {
       getCmdLineArgumentString(argc, (const char **)argv, "fB", &value);
       fB = fopen(value, "r"); 
    }

    printf("PolynomialA(%d), PolynomialB(%d), Order = %d, Number of Variables = %d\n", dimA, dimB, order, nvars);
    int polynom_result = polynomMultiply(argc, argv, dimA, dimB, order, nvars, fA, fB);

    exit(polynom_result);
}
