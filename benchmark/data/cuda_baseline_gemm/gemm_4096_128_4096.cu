#include <cuda_runtime.h>

namespace {
constexpr int TILE = 16;
}

__global__ void gemm(const float *A, const float *B, float *C, int m, int k, int n) {
  __shared__ float tile_a[TILE][TILE];
  __shared__ float tile_b[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  for (int tile_k = 0; tile_k < k; tile_k += TILE) {
    int a_col = tile_k + threadIdx.x;
    int b_row = tile_k + threadIdx.y;

    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? A[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? B[b_row * n + col] : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      sum += tile_a[threadIdx.y][kk] * tile_b[kk][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    C[row * n + col] = sum;
  }
}

extern "C" void gemm_kernel(float *A, float *B, float *C, int m, int k, int n) {
  float *d_A;
  float *d_B;
  float *d_C;

  cudaMalloc(&d_A, static_cast<size_t>(m) * k * sizeof(float));
  cudaMalloc(&d_B, static_cast<size_t>(k) * n * sizeof(float));
  cudaMalloc(&d_C, static_cast<size_t>(m) * n * sizeof(float));

  cudaMemcpy(d_A, A, static_cast<size_t>(m) * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, static_cast<size_t>(k) * n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block_size(TILE, TILE);
  dim3 grid_size((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  gemm<<<grid_size, block_size>>>(d_A, d_B, d_C, m, k, n);

  cudaMemcpy(C, d_C, static_cast<size_t>(m) * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
