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
    src_skip[0] = nx * (iny - NPAD2);     // 计算源GPU内点区域所有点数. iny - NPAD2即内点区域y轴区间的长度. 记住只有俩个GPU, 也即任意一GPU的计算只有一边有halo区域. 
    dst_skip[0] = 0;                      // 目的GPU什么都不跳过.
    src_skip[1] = NPAD * nx;              // 计算padding/halo区域所有点数
    dst_skip[1] = (iny - NPAD) * nx;      // 计算目的GPU 内点区域 + halo区域所有点数. iny - NPAD即内点区域+halo区域的y轴长度.
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

    unsigned int ix  = blockIdx.x * blockDim.x + threadIdx.x;           // 计算当前线程在线程网格的1D线性化全局x轴索引. 注意线程网格使用的x-y轴 和 波场网格使用的x-y轴 刚好相反.

    __shared__ float line[BDIMX + PAD2];                               // 共享内存的大小基本上是跟着thread block的大小走的. 这里采用的是1D thread block, 因此共享内存简单起见也会被定义成1D的.
    // 注意在line两边加padding, 这样的话就不用考虑边界case了. 所有点都用同样的代码模式.

    // smem idx for current point
    unsigned int stx = threadIdx.x + PAD;                              // 计算当前线程对应的共享内存索引. offset NPAD=4必加, 最开头NPAD个点是padding.
    unsigned int idx = ix + iStart * nx;                                // idx计算的是当前线程对应的GPU细分区域下的全局波场数组1D线性化索引. 特别注意, wave field数组是一个1Darray, 即使wave field是2D的.
    // iStart * nx 即是y轴下index = iStart之前的波场2D区域的点数(当前GPU的细分区域下). ix表示的是当前线程网格下的线程的1D线性化索引. 
    // 两者相加得到idx, idx即是当前线程要计算的对应的GPU细分区域下的2D全局波场点的1D线性化索引.

    // register for y value. 这里直接使用寄存器, 本质是与共享内存一样, 把全局内存的访问拉近到对寄存器的访问.
    float yval[9];      // y轴方向使用寄存器存储, 每个线程计算一个wave field点, 而y轴方向需要9点求偏导数.
    #pragma unroll
    for (unsigned int i = 0; i < 8; ++i) 
        yval[i] = d_u2[idx + (i - 4) * nx];   // 这里并没有对全局内存的连续访问啊! 这里只写入了y轴上的8个点 => + + + + * + + + => 第9个点放置在了for loop里写入.

    // skip for the bottom most y value
    int iskip = PAD * nx;  // skip掉当前y轴4个点形成的2D区域, 这样我们可以到第九个点进行写入寄存器.

    #pragma unroll 9
    for (unsigned int iy = iStart; iy < iEnd; ++iy) {  // 循环使用共享内存, 以节约空间. 这里一个线程一次性将计算给定2D波长区域的一整条y轴区间长度的点.
        // get yval[8] here
        yval[8] = d_u2[idx + iskip];

        // read halo part
        if(threadIdx.x < PAD) {      // 索引小于NPAD就是halo区域. 这里是对共享内存区域的写入.
            line[threadIdx.x]  = d_u2[idx - PAD];
            line[stx + BDIMX]  = d_u2[idx + BDIMX];   // halo区域的的点将会多负责两个slot的共享内存写入.
        }

        line[stx] = yval[4];
        __syncthreads();              // 每个线程写入共享内存一个slot, 最后需要同步以使得线程对共享内存的写入全部完成.

        // 8rd fd operator. 这里开始真正的有限差分计算求当前时刻波场状态了.
        if ( (ix >= PAD) && (ix < nx - PAD) ) {     // 特别注意, 只有 内点区域 + halo区域要计算波场.
            // center point
            float tmp = coef[0] * line[stx] * 2.0f;   // 中心点会被用2次. 一次x偏导, 一次y偏导.
            #pragma unroll
            for(unsigned int d = 1; d <= 4; ++d) {
                tmp += coef[d] * (line[stx - d] + line[stx + d]);			// d^2u/dx^2
             	tmp += coef[d] * (yval[4 - d] + yval[4 + d]);				// d^u/dy^2
            }

            // time dimension
            d_u1[idx] = 2.0f * yval[4] - g_u1[idx] + dc_v * tmp;    // 有限差分计算相加.
        }

        #pragma unroll 8
        for (int i = 0; i < 8 ; i++) {
            yval[i] = yval[i + 1];      // 寄存器中的值是可以循环利用的! 向左移一位!
        }

        // advancd on global idx
        idx  += nx;                     // 全局索引idx提步到下一个y轴位置!!!!
        __syncthreads();                // 同步线程块中的所有线程!!!
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
    // calculate index for computation. 计算halo区域和网格内点区域的区间索引!!!
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
        if (istep == 0) {         // 在step = 0时, 引入initial value of wave.
            CUDA_ERROR_CHECK( cudaSetDevice(0) );
            kernel_add_initial_wavelet<<<grid, block>>>(d_u2, init_wavelet, const int nx, const int iny, const int ngpus) 

            kernel_add_wavelet<<<grid, block>>>(d_u2[0], 20.0, nx, iny, ngpus);
        }

        // halo part. 波动仿真计算.
        for (int i = 0; i < ngpus; i++) {
            CUDA_ERROR_CHECK( cudaSetDevice(i) );
            // compute halo
            kernel_2dfd<<<grid, block, 0, stream_halo[i]>>>(d_u1[i], d_u2[i], nx, haloStart[i], haloEnd[i]);
            // compute internal
            kernel_2dfd<<<grid, block, 0, stream_body[i]>>>(d_u1[i], d_u2[i], nx, bodyStart[i], bodyEnd[i]);
        }

        // exchange halo. halo区域数据交换..
        if (ngpus > 1) {
            CUDA_ERROR_CHECK( cudaMemcpyAsync(d_u1[1] + dst_skip[0], d_u1[0] + src_skip[0], iexchange, cudaMemcpyDefault, stream_halo[0]) );
            CUDA_ERROR_CHECK( cudaMemcpyAsync(d_u1[0] + dst_skip[1], d_u1[1] + src_skip[1], iexchange, cudaMemcpyDefault, stream_halo[1]) );
        }

        for (int i = 0; i < ngpus; i++) {         // 每一次迭代最后都要记得同步所有设备.
            CUDA_ERROR_CHECK( cudaSetDevice(i) );
            CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

            float *tmpu0 = d_u1[i];               // 双缓冲策略, 一个数组用于保存当前波场wave field的状态, 另一个数组用于保存更新后的波场wave field的状态.
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
