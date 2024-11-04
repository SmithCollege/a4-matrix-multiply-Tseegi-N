#include <iostream>
#include <cuda_runtime.h>

// define constants
#define SIZE 10000

__global__ void MatrixMul(float* M, float* N, float* P, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  if (row < n && col < n) {
    for (int k = 0; k < n; k++) {
      sum += M[row * n + k] * N[k * n + col];
    }
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
  dim3 blockSize(16, 16);
  dim3 gridSize((SIZE + blockSize.x - 1) / blockSize.x, (SIZE + blockSize.y - 1) / blockSize.y);

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
