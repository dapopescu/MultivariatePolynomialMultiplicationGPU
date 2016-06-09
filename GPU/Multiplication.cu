/**
 * Copyright 2013-2016 Diana-Andreea Popescu, University of Cambridge, UK.  All rights reserved.
 *
 */

/**
 * Multivariate Polynom multiplication: C = A * B.
 *
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <string>
// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
	#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

inline
void checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
	exit(EXIT_FAILURE);
  }
}

/**
 * Multivariate polynomial multiplication (CUDA Kernel) on the device: C = A * B
 * nA is A's number of terms and nB is B's number of terms
 */

struct is_order_less
{
	__host__ __device__
	bool operator() (const int x)
	{
		return (x < 0);
	}
};

template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int NVARS, int MAX_EXP> __global__ void
multivariateMulArbitrarySizedPolsCUDA(uint8_t *exp_C, unsigned long long *exp_keys, uint8_t *exp_A, uint8_t  *exp_B,
	double *coeff_C, double *coeff_A, double *coeff_B,
	unsigned int nC, unsigned int nA, unsigned int nB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tbx0 = tx + bx * BLOCK_SIZE_X;
    int tby0 = ty + by * BLOCK_SIZE_Y;

    int tbx = tbx0;
    int tby = tby0;

    int offsetx = BLOCK_SIZE_X * gridDim.x;
    int offsety = BLOCK_SIZE_Y * gridDim.y;
	
    // Declaration of the shared memory array As used to
    // store the sub-polynomial of A
    __shared__ uint8_t  Aes[BLOCK_SIZE_X * NVARS];
    __shared__ double Acs[BLOCK_SIZE_X];

    // Declaration of the shared memory array Bs used to
    // store the sub-polynomial of B
    __shared__ uint8_t  Bes[BLOCK_SIZE_Y * NVARS];
    __shared__ double Bcs[BLOCK_SIZE_Y];

    // Multiply the two polynomials together;
    // each thread computes one element
    // of the block sub-polynom
    uint8_t  Cexp[NVARS];
    unsigned long long ekey = 0;
    double Ccoeff = 0;
    int c = 0;
    // Load the polynoms from device memory
    // to shared memory; each thread loads
    // one element of each polynom

    while (tby < nB) {
	    while (tbx < nA) {
		    Acs[tx] = coeff_A[tbx];
		    Bcs[ty] = coeff_B[tby];
//#pragma unroll
		    for (int k = 0; k < NVARS; ++k) {
			    Aes[tx + k * BLOCK_SIZE_X] = exp_A[tbx + k * nA]; 
			    Bes[ty + k * BLOCK_SIZE_Y] = exp_B[tby + k * nB];
		    }
		    // Ccoeff is used to store the coefficient of the term product
		    // that is computed by the thread
		    Ccoeff = Acs[tx] * Bcs[ty];
		    ekey = 0;
		    // Write the block sub-polynomial to device memory;
		    // each thread writes one element
		    c = nA * tby + tbx;
		    coeff_C[c] = Ccoeff;
//#pragma unroll
		    for (int k = 0; k < NVARS; ++k) 
		    {
			Cexp[k] = Aes[tx + k * BLOCK_SIZE_X] + Bes[ty + k * BLOCK_SIZE_Y];
			exp_C[c + k * nC] = Cexp[k]; 
			ekey = MAX_EXP * ekey + Cexp[k];
		    }
		    exp_keys[c] = ekey;
			
		    //update index
		    tbx += offsetx; 
		}
		//update index
		tby += offsety;
		//reset 
		tbx = tbx0;
		
	}
}

template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int NVARS, int MAX_EXP> __global__ void
multivariatePolMulTruncateCUDA(unsigned int *exp_C, unsigned long long *exp_keys, unsigned int *exp_A, unsigned int *exp_B,
					double *coeff_C, double *coeff_A, double *coeff_B, 					
					unsigned int nC, unsigned int nA, unsigned int nB,
					int order, int* stencil)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	
    int tbx0 = tx + bx * BLOCK_SIZE_X;
    int tby0 = ty + by * BLOCK_SIZE_Y;
 
    int tbx = tbx0;
    int tby = tby0;

    int offsetx = BLOCK_SIZE_X * gridDim.x;
    int offsety = BLOCK_SIZE_Y * gridDim.y;
	
    // Declaration of the shared memory array As used to
    // store the sub-polynomial of A
    __shared__ unsigned int Aes[BLOCK_SIZE_X * NVARS];
    __shared__ double Acs[BLOCK_SIZE_X];

    // Declaration of the shared memory array Bs used to
    // store the sub-polynomial of B
    __shared__ unsigned int Bes[BLOCK_SIZE_Y * NVARS];
    __shared__ double Bcs[BLOCK_SIZE_Y];


    // Multiply the two polynomials together;
    // each thread computes one element
    // of the block sub-polynom
    unsigned int Cexp[NVARS];
    unsigned long long ekey = 0;
    double Ccoeff = 0;
    int c = 0;
    int sum = 0;
    // Load the polynoms from device memory
    // to shared memory; each thread loads
    // one element of each polynom

    while (tby < nB) {
	    while (tbx < nA) {
		    Acs[tx] = coeff_A[tbx];
		    Bcs[ty] = coeff_B[tby];
//#pragma unroll
		    for (int k = 0; k < NVARS; ++k) {
			    Aes[tx + k * BLOCK_SIZE_X] = exp_A[tbx + k * nA]; 
			    Bes[ty + k * BLOCK_SIZE_Y] = exp_B[tby + k * nB];
		    }
		    // Ccoeff is used to store the coefficient of the term product
		    // that is computed by the thread
		    Ccoeff = Acs[tx] * Bcs[ty];
		    ekey = 0;
		    sum = 0;
		    // Write the block sub-polynomial to device memory;
		    // each thread writes one element
		    c = nA * tby + tbx;
		    coeff_C[c] = Ccoeff;
//#pragma unroll
		    for (int k = 0; k < NVARS; ++k) 
		    {
			Cexp[k] = Aes[tx + k * BLOCK_SIZE_X] + Bes[ty + k * BLOCK_SIZE_Y];
			exp_C[c + k * nC] = Cexp[k]; 
			ekey = MAX_EXP * ekey + Cexp[k];
			sum += Cexp[k];
		    }
		    if (sum <= order)
		    	stencil[c] = 1;
		    else stencil[c] = 0;
		    exp_keys[c] = ekey;
			
		    //update index
		    tbx += offsetx; 
	    }
	    //update index
	    tby += offsety;
	    //reset 
	    tbx = tbx0;
    }  
}

template <int NVARS, int MAX_EXP> __global__ void
getExponentsFromKeysCUDA(uint8_t  *exp_C, unsigned long long *exp_keys, unsigned int nC)
{
	// Block index
	int bx = blockIdx.x;
	// Thread index
	int tx = threadIdx.x;
	int tbx = tx + bx * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	unsigned long long key = 0, kd = 0;
	while (tbx < nC) {
		key = exp_keys[tbx];
//#pragma unroll
		for (int k = NVARS - 1; k >= 0; k--) {
		  	kd = key/MAX_EXP;
			exp_C[tbx + k * nC] = key - kd * MAX_EXP; 
			key = kd;
		}
		tbx += offset;
	}
}

void initPol(uint8_t *exps, double* coeffs, FILE* f, int dim) {
  uint8_t exp[3];
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
int polynomMultiply(int argc, char **argv, int block_size, unsigned int &dimA, unsigned int &dimB, 
	unsigned int &order, unsigned int &nvars, FILE *fA, FILE *fB)
{
    cudaError_t error;
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    checkCuda(cudaEventCreate(&start));
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&stop));

    // Allocate host memory for polynomials A and B
    unsigned int size_A = dimA * nvars;
    unsigned int mem_size_exp_A = sizeof(uint8_t) * size_A;
    unsigned int mem_size_coeff_A = sizeof(double) * dimA;
    uint8_t  *exp_A = NULL;
    checkCuda(cudaMallocHost((void **)&exp_A, mem_size_exp_A));
    double *coeff_A = NULL;
    checkCuda(cudaMallocHost((void **)&coeff_A, mem_size_coeff_A));
    unsigned int size_B = dimB * nvars;
    unsigned int mem_size_exp_B = sizeof(uint8_t) * size_B;
    unsigned int mem_size_coeff_B = sizeof(double) * dimB;
    uint8_t  *exp_B = NULL;
    checkCuda(cudaMallocHost((void **)&exp_B, mem_size_exp_B));
    double *coeff_B = NULL;
    checkCuda(cudaMallocHost((void **)&coeff_B, mem_size_coeff_B));

    // Initialize host memory
    initPol(exp_A, coeff_A, fA, dimA);
    initPol(exp_B, coeff_B, fB, dimB);

    // Allocate device memory
    uint8_t  *e_A, *e_B, *e_C;
    double *c_A, *c_B, *c_C;
    double *final_coeff_C;
    unsigned long long *e_keys_C;
    unsigned long long *final_keys_C;

    // Allocate host polynomial C
    unsigned int dimC = dimA * dimB;
    unsigned int size_C = dimA * dimB * nvars;
    unsigned int mem_size_exp_C = size_C * sizeof(uint8_t);
    uint8_t  *exp_C = NULL;
    checkCuda(cudaMallocHost((void **)&exp_C, mem_size_exp_C));

    unsigned int mem_size_keys_C = dimC * sizeof(unsigned long long); 
    unsigned int mem_size_coeff_C = sizeof(double) * dimC;
    double *coeff_C = NULL;
    checkCuda(cudaMallocHost((void **)&coeff_C, mem_size_coeff_C));


    checkCuda(cudaMalloc((void **) &e_A, mem_size_exp_A));

    checkCuda(cudaMalloc((void **) &c_A, mem_size_coeff_A));

    checkCuda(cudaMalloc((void **) &e_B, mem_size_exp_B));

    checkCuda(cudaMalloc((void **) &c_B, mem_size_coeff_B));

    checkCuda(cudaMalloc((void **) &e_C, mem_size_exp_C));

    checkCuda(cudaMalloc((void **) &c_C, mem_size_coeff_C));

    checkCuda(cudaMalloc((void **) &final_coeff_C, mem_size_coeff_C));

    checkCuda(cudaMalloc((void **) &e_keys_C, mem_size_keys_C));

    checkCuda(cudaMalloc((void **) &final_keys_C, mem_size_keys_C));

    //STENCIL FOR TRUNCATION
    int *stencil = NULL;
    unsigned int mem_size_stencil = sizeof(int) * dimC;
    checkCuda(cudaMalloc((void **) &stencil, mem_size_stencil));

    // Record the start event
    checkCuda(cudaEventRecord(start, NULL));

    // copy host memory to device
    checkCuda(cudaMemcpy(e_A, exp_A, mem_size_exp_A, cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(c_A, coeff_A, mem_size_coeff_A, cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(e_B, exp_B, mem_size_exp_B, cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(c_B, coeff_B, mem_size_coeff_B, cudaMemcpyHostToDevice));

     // Record the stop event
    checkCuda(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCuda(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    float total = 0.0f;
    checkCuda(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("Transfer input =Time= %.3f\n", msecTotal);
    total += msecTotal;	
    // Setup execution parameters
    dim3 threads(block_size, block_size);
    int limitA = dimA % threads.x;
    int x1 = 1;
    if (limitA == 0)
	    x1 = 0;
    int limitB = dimB % threads.y;
    int y1 = 1;
    if (limitB == 0)
	    y1 = 0;
    dim3 grid(dimA / threads.x + x1, dimB / threads.y + y1); 

    // Performs warmup operation
        
    multivariateMulArbitrarySizedPolsCUDA<16, 16, 3, 100><<< grid, threads >>>(e_C, e_keys_C, 
    			e_A, e_B, c_C, c_A, c_B, dimC, dimA, dimB);
    thrust::device_ptr<unsigned long long> keys_C_dev(e_keys_C);
    thrust::device_ptr<double> coeffs_C_dev(c_C);
    thrust::device_ptr<unsigned long long> final_keys_C_dev(final_keys_C);
    thrust::device_ptr<double> final_coeff_C_dev(final_coeff_C);

    /*    thrust::device_ptr<int> stencil_dev(stencil);
    thrust::device_ptr<double> end_coeffs_dev = thrust::remove_if(coeffs_C_dev, coeffs_C_dev + dimC, stencil_dev, is_order_less());
    thrust::device_ptr<unsigned long long> end_keys_dev = thrust::remove_if(keys_C_dev, keys_C_dev + dimC, stencil_dev, is_order_less());
    thrust::sort_by_key(keys_C_dev, end_keys_dev, coeffs_C_dev);
    thrust::pair<thrust::device_ptr<unsigned long long>, thrust::device_ptr<double>> end;
    end = thrust::reduce_by_key(keys_C_dev, end_keys_dev, coeffs_C_dev, final_keys_C_dev, final_coeff_C_dev); */

    thrust::sort_by_key(keys_C_dev, keys_C_dev + dimC, coeffs_C_dev);
    thrust::pair<thrust::device_ptr<unsigned long long>, thrust::device_ptr<double> > end;
    end = thrust::reduce_by_key(keys_C_dev, keys_C_dev + dimC, coeffs_C_dev, final_keys_C_dev, final_coeff_C_dev); 

    unsigned int sizeC = end.first - final_keys_C_dev; 

    dim3 threads_exp(1024);
     
    if (sizeC % threads_exp.x == 0)
	x1 = 0;
    else x1 = 1;
    dim3 grid_exp(sizeC/threads_exp.x + x1);

    getExponentsFromKeysCUDA<3,100><<< grid_exp, threads_exp >>>(e_C, final_keys_C, sizeC);

    cudaDeviceSynchronize();

    // Record the start event
    checkCuda(cudaEventRecord(start, NULL));

    // Execute the kernel
    int nIter = 1;

    for (int j = 0; j < nIter; j++)
    {
	//If one wants to truncate the polynomials, they must use the multivariatePolMulTruncateCUDA kernel instead.
      	
	multivariateMulArbitrarySizedPolsCUDA<16, 16, 3, 100><<< grid, threads >>>(e_C, e_keys_C, 
    			e_A, e_B, c_C, c_A, c_B, dimC, dimA, dimB);	
		
	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	printf("Computation=  Time= %.3f\n", msecTotal);
        total += msecTotal;
 		
	
	thrust::device_ptr<unsigned long long> keys_C_dev(e_keys_C);
	thrust::device_ptr<double> values_C_dev(c_C);
	thrust::device_ptr<unsigned long long> final_keys_C_dev(final_keys_C);
	thrust::device_ptr<double> final_coeff_C_dev(final_coeff_C);
	
	/* This part is to be used if one wants to truncate the polynomials to a certain order.
	It uses the remove_if function from the NVIDIA Thrust library before sorting and reducing the result polynomial.

	thrust::device_ptr<int> stencil_dev(stencil);
	// Remove result terms whose order is larger than the maximum order
	thrust::device_ptr<double> end_coeffs_dev = thrust::remove_if(coeffs_C_dev, coeffs_C_dev + dimC, stencil_dev, is_order_less());
	thrust::device_ptr<unsigned long long> end_keys_dev = thrust::remove_if(keys_C_dev, keys_C_dev + dimC, stencil_dev, is_order_less());
	// Sort the result terms
	thrust::sort_by_key(keys_C_dev, end_keys_dev, coeffs_C_dev);
	thrust::pair<thrust::device_ptr<unsigned long long>, thrust::device_ptr<double>> end;
	// Reduce by key (add the terms)
	end = thrust::reduce_by_key(keys_C_dev, end_keys_dev, coeffs_C_dev, final_keys_C_dev, final_coeff_C_dev); 
	// Recover exponents
	getExponentsFromKeysCUDA<3,100><<< grid_exp, threads_exp >>>(e_C, final_keys_C, sizeC); */
		
	error = cudaEventRecord(start, NULL); 
	thrust::sort_by_key(keys_C_dev, keys_C_dev + dimC, coeffs_C_dev);
	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	printf("Sort=  Time= %.3f\n", msecTotal);
        total += msecTotal;
	error = cudaEventRecord(start, NULL); 
	thrust::pair<thrust::device_ptr<unsigned long long>, thrust::device_ptr<double> > end;
	end = thrust::reduce_by_key(keys_C_dev, keys_C_dev + dimC, values_C_dev, final_keys_C_dev, final_coeff_C_dev);
	
	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	printf("Reduce=  Time= %.3f\n", msecTotal);
	total += msecTotal;
	error = cudaEventRecord(start, NULL); 
	getExponentsFromKeysCUDA<3, 100><<< grid_exp, threads_exp >>>(e_C, final_keys_C, sizeC);
	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	printf("GetKeys=  Time= %.3f\n", msecTotal);
        total += msecTotal;
    }

    cudaDeviceSynchronize();
    checkCuda(cudaEventRecord(start, NULL));
    // Copy result from device to host
    checkCuda(cudaMemcpy(exp_C, e_C, sizeC * nvars * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    checkCuda(cudaMemcpy(coeff_C, final_coeff_C, sizeC * sizeof(double), cudaMemcpyDeviceToHost));

    checkCuda(cudaEventRecord(stop, NULL));
    checkCuda(cudaEventSynchronize(stop));
    msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Transfer output= Time= %.3f\n", msecTotal);
    total += msecTotal;
    printf("TOTAL=%.3f\n",total);
    printf("Checking for errors: ");
    printf("%s\n", error ? "FAIL" : "OK");

    // Clean up memory
    cudaFree(exp_A);
    cudaFree(exp_B);
    cudaFree(exp_C);
    cudaFree(e_A);
    cudaFree(e_B);
    cudaFree(e_C);
    cudaFree(coeff_A);
    cudaFree(coeff_B);
    cudaFree(coeff_C);
    cudaFree(c_A);
    cudaFree(c_B);
    cudaFree(c_C);

    cudaFree(e_keys_C);
    cudaFree(final_keys_C);
    cudaFree(final_coeff_C);
    cudaFree(stencil);
    cudaDeviceReset();

    if (error == cudaSuccess)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    } 
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
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -nA=NumberOfTermsA (Number of terms of polynomial A)\n");
        printf("      -nB=NumberOfTermsB (Number of terms of polynomial B)\n");
        printf("      -x=vars (Number of variables)\n");
	printf("      -o=order (Order of polynoms).\n");
	printf("      -b=block_size (Block size).\n");
        printf("      -fA=polA.\n");
        printf("      -fB=polB.\n"); 
        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

   
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    if (checkCmdLineFlag(argc, (const char **)argv, "b"))
    {
        block_size = getCmdLineArgumentInt(argc, (const char **)argv, "b");
    }
    unsigned int dimA = 16 * block_size;
    unsigned int dimB = 16 * block_size;

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
    // Order of polynoms
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
   
    int polynom_result = polynomMultiply(argc, argv, block_size, dimA, dimB, order, nvars, fA, fB);
   
    exit(polynom_result);
}
