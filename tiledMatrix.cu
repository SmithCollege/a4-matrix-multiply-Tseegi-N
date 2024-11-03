#include <iostream>
#include <cuda_runtime.h>

// define constants
#define SIZE 10000
#define TILE_SIZE 10

__global__ void MatrixMul(float* M, float* N, float* P, int n) {
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  // Loop over tiles
  for (int k = 0; k < n / TILE_SIZE; ++k) {
    // Load tiles into shared memory
    tile_A[threadIdx.y][threadIdx.x] = M[row * n + (k * TILE_SIZE + threadIdx.x)];
    tile_B[threadIdx.y][threadIdx.x] = N[(k * TILE_SIZE + threadIdx.y) * n + col];
    __syncthreads();

    // Multiply elements of the tile and accumulate results
    for (int j = 0; j < TILE_SIZE; ++j) {
      sum += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
    }
    __syncthreads();
  }

  // Store the result
  if (row < n && col < n) {
    P[row * n + col] = sum;
  }
}

int main() {
  int grid = SIZE * SIZE * sizeof(float);
  float* x = (float*)malloc(grid);
  float* y = (float*)malloc(grid);
  float* z = (float*)malloc(grid);

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      x[i * SIZE + j] = 1; // x[i][j]
      y[i * SIZE + j] = 1;
    }
  }

  // Allocate GPU matrices
  float *d_x, *d_y, *d_z;
  cudaMalloc(&d_x, grid);
  cudaMalloc(&d_y, grid);
  cudaMalloc(&d_z, grid);

  // Copy matrices to GPU
  cudaMemcpy(d_x, x, grid, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, grid, cudaMemcpyHostToDevice);

  // Kernel launch parameters
  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize((SIZE + TILE_SIZE - 1) / TILE_SIZE, (SIZE + TILE_SIZE - 1) / TILE_SIZE);

  // Launch the matrix multiplication kernel
  MatrixMul<<<gridSize, blockSize>>>(d_x, d_y, d_z, SIZE);

  // Copy result back to host
  cudaMemcpy(z, d_z, grid, cudaMemcpyDeviceToHost);

  std::cout << "z[0]: " << z[0] << " z[N-1]: " << z[SIZE-1] << std::endl;

  // Free memory
  free(x);
  free(y);
  free(z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  return 0;
}
