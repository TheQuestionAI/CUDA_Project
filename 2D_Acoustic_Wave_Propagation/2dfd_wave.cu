/*
Multiple GPUs-based CUDA implementation of 2D acoustic wave propagation using finite-difference scheme in time domain. 

The code is solving second order 2D wave equation:
	d^2u/dx^2 + d^u^2/du^2 = v^(-2) * d^2u/dt^2

	u = u(x,y; t) 		=> the wave field
	v = v(x,y) 			=> the constant wave velocity in medium.

Finite Difference:
	We use 17-point stencil template to approximate the partial derivative at a single wave field point.
						*
						*
						*
						*
					* * * * + * * * * 
						*
						*
						*
						*

Multiple GPUs implementation, each will be responsible for one sub-wave-field domain.

                  GPU 0                                          GPU 1 							... 				GPU N
      | | | | * * * * * * * * * * * - - - - | | | |     | | | | - - - - * * * * * * * * * * * - - - - | | | | 		...  	| | | | - - - - * * * * * * * * * * * | | | |
      | | | | * * * * * * * * * * * - - - - | | | |     | | | | - - - - * * * * * * * * * * * - - - - | | | |		...  	| | | | - - - - * * * * * * * * * * * | | | |
      | | | | * * * * * * * * * * * - - - - | | | |     | | | | - - - - * * * * * * * * * * * - - - - | | | |		...  	| | | | - - - - * * * * * * * * * * * | | | |
      | | | | * * * * * * * * * * * - - - - | | | |     | | | | - - - - * * * * * * * * * * * - - - - | | | |		...  	| | | | - - - - * * * * * * * * * * * | | | |
      | | | | * * * * * * * * * * * - - - - | | | |     | | | | - - - - * * * * * * * * * * * - - - - | | | |		...  	| | | | - - - - * * * * * * * * * * * | | | |
      | | | | * * * * * * * * * * * - - - - | | | |     | | | | - - - - * * * * * * * * * * * - - - - | | | |		...  	| | | | - - - - * * * * * * * * * * * | | | |
      | | | | * * * * * * * * * * * - - - - | | | |     | | | | - - - - * * * * * * * * * * * - - - - | | | |		...  	| | | | - - - - * * * * * * * * * * * | | | |
      | | | | * * * * * * * * * * * - - - - | | | |     | | | | - - - - * * * * * * * * * * * - - - - | | | |		...  	| | | | - - - - * * * * * * * * * * * | | | |
      | | | | * * * * * * * * * * * - - - - | | | |     | | | | - - - - * * * * * * * * * * * - - - - | | | |		...  	| | | | - - - - * * * * * * * * * * * | | | |

      padding		body     		halo    padding     padding   halo      	body 		   halo   padding 				padding   halo			body 		  padding
							
      x
      ^
      |
      |
      |
      |
      |
      |------------------------------------------------------------------------------------------------------------------------------------------> y
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// padding for fd scheme.
#define PAD 4
#define PAD2 8

// define wave equation and fd coefficients
#define a0  -2.8472222f
#define a1   1.6000000f
#define a2  -0.2000000f
#define a3   0.0253968f
#define a4  -0.0017857f

#define v 0.12f 		// wave velocity square.

// define thread block dimension, padding
#define BDIMX 256

// store the coefficient and wave evelocity to constant memory
__constant__ float dc_coeff[5];
__constant__ float dc_v;


// Check error codes for CUDA functions
void CUDA_ERROR_CHECK(cudaError_t err) 
{
    if (err != cudaSuccess) 
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}


// setup wave equation and finite difference coefficients.
void setup_coefficient()
{
	const float h_coef[5] = {a0, a1, a2, a3, a4};
	const float h_v = v;
	CUDA_ERROR_CHECK( cudaMemcpyToSymbol(dc_coef, h_coef, 5 * sizeof(float)) );
    CUDA_ERROR_HANDLE( cudaMemcpyToSymbol(&dc_v, &h_v, sizeof(float)) );
}

// calculate each intervals for the halo region and body region.
inline void calculate_halo_body_yregion(int* halo_start, int* halo_end, int* body_start, int* body_end, const int ngpus, const int iny) 
{
	if(ngpus == 0)		// one gpu special case
	{
        body_start[idx] = PAD;
        body_end[idx]   = iny - PAD2 - 1;

        halo_start[idx] = iny - PAD2;
        halo_end[idx]   = iny - PAD - 1;

        return;		
	}

    // halo regions
    for(int idx = 0; idx < 2*(ngpus-1); ++idx) 
    {
        if (idx == 0)				// GPU 0 -> only right hand side has halo region 
        {
            body_start[idx] = PAD;
            body_end[idx]   = iny - PAD2 - 1;

            halo_start[idx] = iny - PAD2;
            halo_end[idx]   = iny - PAD - 1;

        }
        else if(idx == ngpus - 1)		// GPU N -> only left hand side has halo region
        {
            halo_start[idx] = PAD;      
            halo_end[idx]   = PAD2 - 1;

            body_start[idx] = PAD2;
            body_end[idx]   = iny - PAD - 1;  
        }
        else  							// GPU 1 ... N-1 -> both left and right side have halo region
        {	// left halo
            halo_start[idx] = PAD;      
            halo_end[idx]   = PAD2 - 1;
            // body
            body_start[idx] = PAD2;
            body_end[idx]   = iny - PAD2 - 1;          	
            // right halo
            halo_start[++idx] = iny - PAD2;
            halo_end[++idx]   = iny - PAD - 1;

        }

}

// re-visited
inline void calcSkips(int* src_skip, int* dst_skip, const int ngpus, const int nx, const int iny) 
{
    src_skip[0] = nx * (iny - NPAD2);     // ?????????GPU????????????????????????. iny - NPAD2???????????????y??????????????????. ??????????????????GPU, ???????????????GPU????????????????????????halo??????. 
    dst_skip[0] = 0;                      // ??????GPU??????????????????.
    src_skip[1] = NPAD * nx;              // ??????padding/halo??????????????????
    dst_skip[1] = (iny - NPAD) * nx;      // ????????????GPU ???????????? + halo??????????????????. iny - NPAD???????????????+halo?????????y?????????.
}

// intial wavelet for the wave field at time = 0. Need to select the central place GPU to set.
__global__ void kernel_add_initial_wavelet(float* d_u, float init_wavelet, const int nx, const int iny, const int ngpus) 
{
    int src_ypos = (ngpus % 2 == 0 ? iny : iny / 2);			// ngpus even or odd, the central for y is differnt.
    int src_xpos = nx / 2;
    
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;	// Align with the thread block and thread grid dimension selected.
    unsigned int idx = src_ypos * nx + ix;		// 1D global index for wave field array
    if(ix == src_xpos) 							// put the initial wave at the center of wave field u(x,y; t)
		d_u[idx] += init_wavelet;
}

// The core finite difference kernel function. 
__global__ void kernel_2dfd(float *d_u1, float *d_u2, const int nx, const int iStart, const int iEnd) {

    unsigned int ix  = blockIdx.x * blockDim.x + threadIdx.x;           // ????????????????????????????????????1D???????????????x?????????. ???????????????????????????x-y??? ??? ?????????????????????x-y??? ????????????.

    __shared__ float line[BDIMX + PAD2];                               // ???????????????????????????????????????thread block???????????????. ??????????????????1D thread block, ????????????????????????????????????????????????1D???.
    // ?????????line?????????padding, ?????????????????????????????????case???. ????????????????????????????????????.

    // smem idx for current point
    unsigned int stx = threadIdx.x + PAD;                              // ?????????????????????????????????????????????. offset NPAD=4??????, ?????????NPAD?????????padding.
    unsigned int idx = ix + iStart * nx;                                // idx?????????????????????????????????GPU????????????????????????????????????1D???????????????. ????????????, wave field???????????????1Darray, ??????wave field???2D???.
    // iStart * nx ??????y??????index = iStart???????????????2D???????????????(??????GPU??????????????????). ix?????????????????????????????????????????????1D???????????????. 
    // ??????????????????idx, idx???????????????????????????????????????GPU??????????????????2D??????????????????1D???????????????.

    // register for y value. ???????????????????????????, ??????????????????????????????, ??????????????????????????????????????????????????????.
    float yval[9];      // y??????????????????????????????, ????????????????????????wave field???, ???y???????????????9???????????????.
    #pragma unroll
    for (unsigned int i = 0; i < 8; ++i) 
        yval[i] = d_u2[idx + (i - 4) * nx];   // ????????????????????????????????????????????????! ??????????????????y?????????8?????? => + + + + * + + + => ???9??????????????????for loop?????????.

    // skip for the bottom most y value
    int iskip = PAD * nx;  // skip?????????y???4???????????????2D??????, ??????????????????????????????????????????????????????.

    #pragma unroll 9
    for (unsigned int iy = iStart; iy < iEnd; ++iy) {  // ????????????????????????, ???????????????. ??????????????????????????????????????????2D????????????????????????y?????????????????????.
        // get yval[8] here
        yval[8] = d_u2[idx + iskip];

        // read halo part
        if(threadIdx.x < PAD) {      // ????????????NPAD??????halo??????. ???????????????????????????????????????.
            line[threadIdx.x]  = d_u2[idx - PAD];
            line[stx + BDIMX]  = d_u2[idx + BDIMX];   // halo????????????????????????????????????slot?????????????????????.
        }

        line[stx] = yval[4];
        __syncthreads();              // ????????????????????????????????????slot, ?????????????????????????????????????????????????????????????????????.

        // 8rd fd operator. ?????????????????????????????????????????????????????????????????????.
        if ( (ix >= PAD) && (ix < nx - PAD) ) {     // ????????????, ?????? ???????????? + halo?????????????????????.
            // center point
            float tmp = coef[0] * line[stx] * 2.0f;   // ??????????????????2???. ??????x??????, ??????y??????.
            #pragma unroll
            for(unsigned int d = 1; d <= 4; ++d) {
                tmp += coef[d] * (line[stx - d] + line[stx + d]);			// d^2u/dx^2
             	tmp += coef[d] * (yval[4 - d] + yval[4 + d]);				// d^u/dy^2
            }

            // time dimension
            d_u1[idx] = 2.0f * yval[4] - g_u1[idx] + dc_v * tmp;    // ????????????????????????.
        }

        #pragma unroll 8
        for (int i = 0; i < 8 ; i++) {
            yval[i] = yval[i + 1];      // ??????????????????????????????????????????! ???????????????!
        }

        // advancd on global idx
        idx  += nx;                     // ????????????idx??????????????????y?????????!!!!
        __syncthreads();                // ?????????????????????????????????!!!
    }
}


int main(int argc, char** argv) {

    int ngpus;                                              // make the choice simple, use all available CUDA-capable GPUs.
    CUDA_ERROR_CHECK( cudaGetDeviceCount(&ngpus) );
    printf("Single computing node CUDA-capable GPU count: %i\n", ngpus);		
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // size
    const int nsteps  = 3600;                                            // total iteration steps.
    const int nx      = 1024 * ngpus;                                    // wave field u(x,y;t) x (vertical) dimension.
    const int ny      = 1024 * ngpus;                                    // wave field u(x,y;t) y (horizontal) dimension.
    const int iny     = ny / ngpus + PAD * 2;                            // Split wave-field along the y axis. Each GPU a sub-wave-field (with padding to simplify the division).

    size_t isize = nx * iny;                                             // sub-wave-field state points.
    size_t ibyte = isize * sizeof(float);                                // sub-wave-field state points memory size.
    size_t iexchange = PAD * nx * sizeof(float);                         // the memory size needed to be exchanged between GPUs (i.e, halo region memory size).
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // double buffering in each GPU. one buffer d_u1 saves the wave field in previous state, another d_u2 saves the wave field in current state.
    float *d_u1[ngpus], *d_u2[ngpus];
    for(int idx = 0; idx < ngpus; ++idx) {
        // set device. Device must be set before any operation on the specified GPU.
        CUDA_ERROR_CHECK( cudaSetDevice(i) );                          

        // allocate device memories.
        CUDA_ERROR_CHECK( cudaMalloc((void**)&d_u1[i], ibyte) );       
        CUDA_ERROR_CHECK( cudaMalloc((void**)&d_u2[i], ibyte) );

        // initial memory value all to 0.
        CUDA_ERROR_CHECK( cudaMemset(d_u1[i], 0, ibyte) );             
        CUDA_ERROR_CHECK( cudaMemset(d_u2[i], 0, ibyte) );             

        setup_coefficient();   // Each device, set up its own coefficients. We must pass these coefficients onto each device's constant memory.     
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create the CUDA streams, overlap halo region data transfer with body state update computation.
    cudaStream_t halo_streams[ngpus], body_streams[ngpus]; 
    for (int idx = 0; idx < ngpus; ++idx) {
        CUDA_ERROR_CHECK( cudaSetDevice(idx) );      
        // create CUDA stream under associate CUDA device.                    	
        CUDA_ERROR_CHECK( cudaStreamCreate(&halo_streams[idx]) );		// Each CUDA stream and CUDA event can only associate to one signle CUDA device
        CUDA_ERROR_CHECK( cudaStreamCreate(&body_streams[idx]) );
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // calculate index for computation. ??????halo??????????????????????????????????????????!!!
    unsigned int size = (ngpus == 1 ? ngpus : 2*(ngpus - 1);
    int halo_starts[size], halo_ends[size]; 
    int body_starts[size], body_ends[size];
    calculate_halo_body_yregion(halo_starts, halo_ends, body_starts, body_ends, ngpus, iny);		// Remember iny is with padding left and right.

    int src_skip[ngpus], dst_skip[ngpus];

    if(ngpus > 1) 
        calcSkips(src_skip, dst_skip, nx, iny);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // kernel launch configuration
    dim3 block(BDIMX, 1, 1);                          	// 1d thread block + 1d thread grid. each thread block corresponds to 1 segment in x dimension.
    dim3 grid(nx/block.x, 1, 1);                    	// Split the single total line in x dimension to several BDIMX size segments.

    // set up event for timing
    CUDA_ERROR_CHECK( cudaSetDevice(0) );
    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK( cudaEventCreate(&start) );
    CUDA_ERROR_CHECK( cudaEventCreate(&stop ) );
    CUDA_ERROR_CHECK( cudaEventRecord(start, 0) );

    // main loop for wave propagation
    for(int istep = 0; istep < nsteps; ++istep) 
    {
        // add wavelet to the central of wave field at time = 0.
        if (istep == 0) {         // ???step = 0???, ??????initial value of wave.
            CUDA_ERROR_CHECK( cudaSetDevice(0) );
            kernel_add_initial_wavelet<<<grid, block>>>(d_u2, init_wavelet, const int nx, const int iny, const int ngpus) 

            kernel_add_wavelet<<<grid, block>>>(d_u2[0], 20.0, nx, iny, ngpus);
        }

        // halo part. ??????????????????.
        for (int i = 0; i < ngpus; i++) {
            CUDA_ERROR_CHECK( cudaSetDevice(i) );
            // compute halo
            kernel_2dfd<<<grid, block, 0, stream_halo[i]>>>(d_u1[i], d_u2[i], nx, haloStart[i], haloEnd[i]);
            // compute internal
            kernel_2dfd<<<grid, block, 0, stream_body[i]>>>(d_u1[i], d_u2[i], nx, bodyStart[i], bodyEnd[i]);
        }

        // exchange halo. halo??????????????????..
        if (ngpus > 1) {
            CUDA_ERROR_CHECK( cudaMemcpyAsync(d_u1[1] + dst_skip[0], d_u1[0] + src_skip[0], iexchange, cudaMemcpyDefault, stream_halo[0]) );
            CUDA_ERROR_CHECK( cudaMemcpyAsync(d_u1[0] + dst_skip[1], d_u1[1] + src_skip[1], iexchange, cudaMemcpyDefault, stream_halo[1]) );
        }

        for (int i = 0; i < ngpus; i++) {         // ???????????????????????????????????????????????????.
            CUDA_ERROR_CHECK( cudaSetDevice(i) );
            CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

            float *tmpu0 = d_u1[i];               // ???????????????, ????????????????????????????????????wave field?????????, ?????????????????????????????????????????????wave field?????????.
            d_u1[i] = d_u2[i];
            d_u2[i] = tmpu0;
        }
    }

    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );
    CUDA_ERROR_CHECK( cudaGetLastError() );

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // clear
    for (int i = 0; i < ngpus; i++) {
        CUDA_ERROR_CHECK( cudaSetDevice(i) );

        CUDA_ERROR_CHECK( cudaStreamDestroy(stream_halo[i]) );
        CUDA_ERROR_CHECK( cudaStreamDestroy(stream_body[i]) );

        CUDA_ERROR_CHECK( cudaFree(d_u1[i]) );
        CUDA_ERROR_CHECK( cudaFree(d_u2[i]) );

        CUDA_ERROR_CHECK( cudaDeviceReset() );
    }

    return 0;
}
