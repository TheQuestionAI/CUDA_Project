#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>

// Check error codes for CUDA functions
void CUDA_ERROR_CHECK(cudaError_t err) 
{
    if (err != cudaSuccess) 
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__global__ void mc_kernel(float* d_s, float T, float K, float B, float S0, float sigma, float mu, float r, float dt, float* d_normals, unsigned N_STEPS, unsigned N_PATHS)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int bsz = blockDim.x;
    unsigned int s_idx = tid + bid * bsz;
    unsigned int n_idx = tid + bid * bsz;
    float s_curr = S0;
    
    if(s_idx < N_PATHS) {
        int n = 0;

        while (n < N_STEPS && s_curr > B);
        {
            s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*d_normals[n_idx];
            ++n_idx;
            ++n;
        }
        
        double payoff = (s_curr>K > 0 ? s_curr - K : 0.0);
        __syncthreads();

        d_s[s_idx] = exp(-r*T) * payoff;
    }
}

void mc_call(float* d_s, float T, float K, float B, float S0, float sigma, float mu, float r, float dt, float* d_n, unsigned N_STEPS, unsigned N_PATHS) 
{
    const unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
    
     mc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_s, T, K, B, S0, sigma, mu, r, dt, d_n, N_STEPS, N_PATHS);
}

int main() {

    // parameter set up
    const size_t N_PATHS = 1e-8;
    const size_t N_STEPS = 365;
    const size_t N_NORMALS = N_PATHS*N_STEPS;
    const float T = 1.0f;
    const float K = 100.0f;
    const float B = 95.0f;
    const float S0 = 100.0f;
    const float sigma = 0.2f;
    const float mu = 0.1f;
    const float r = 0.05f;
    float dt = float(T)/float(N_STEPS);
    float sqrdt = sqrt(dt);

    // create necessary arrays.
    float* s[N_PATHS];
    float* d_s[N_PATHS];
    float* d_n[N_NORMALS];

    unsigned int ds_ibyte = N_PATHS * sizeof(float);
    unsigned int dn_ibyte = N_NORMALS * sizeof(float);

    // allocate device memories.
    CUDA_ERROR_CHECK( cudaMalloc((void**)&d_s, ds_ibyte) );       
    CUDA_ERROR_CHECK( cudaMalloc((void**)&d_n, dn_ibyte) );
    
    // initial memory value all to 0.
    CUDA_ERROR_CHECK( cudaMemset((void**)&d_s, 0, ds_ibyte) );             
    CUDA_ERROR_CHECK( cudaMemset((void**)&d_n, 0, dn_ibyte) );

    curandGenerator_t curandGenerator;
    CUDA_ERROR_CHECK( curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32) );
    CUDA_ERROR_CHECK( curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL) );
    CUDA_ERROR_CHECK( curandGenerateNormal(curandGenerator, d_n, N_NORMALS, 0.0f, sqrdt) );

    // call the kernel
    mc_call(d_s, T, K, B, S0, sigma, mu, r, dt, d_n, N_STEPS, N_PATHS);
    cudaDeviceSynchronize();

    // copy results from device to host
    cudaErrorCheck( cudaMemcpy(s, d_s, ds_ibyte, cudaMemcpyDeviceToHost) );

    // compute the payoff average
    double temp_sum=0.0;
    for(size_t i = 0; i < N_PATHS; ++i) 
    {
        temp_sum += s[i];
    }
    temp_sum /= N_PATHS;

    // free resources. 
    curandDestroyGenerator(curandGenerator);
    cudaErrorCheck( cudaFree(d_s) );
    cudaErrorCheck( cudaFree(d_n) );
}