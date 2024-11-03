#include <iostream>
#include "cublas_v2.h"
#include <cuda_runtime.h>

// define constants
#define M 900  // Rows of x and z
#define N 900  // Columns of y and z
#define K 900  // Columns of x and rows of y

void matrixMul(const float* x, const float* y, float* z, int m, int n, int k) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  // Perform the matrix multiplication: z = alpha * x * y + beta * z
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
    x, m,
    y, k,
    &beta,
    z, m);

  cublasDestroy(handle);
}

int main() {
  // Allocate CPU matrices
  float x[M * K], y[K * N], z[M * N] = {0};
  for (int i = 0; i < M * K; i++) x[i] = 1.0f;
  for (int i = 0; i < K * N; i++) y[i] = 1.0f;

  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  std::cout << "Before allocation: Free memory: " << freeMem << " bytes, Total memory: " << totalMem << " bytes\n";

  // Allocate GPU matrices
  float *d_x, *d_y, *d_z;
  cudaMalloc((void**)&d_x, M * K * sizeof(float));
  cudaMalloc((void**)&d_y, K * N * sizeof(float));
  cudaMalloc((void**)&d_z, M * N * sizeof(float));

  cudaMemGetInfo(&freeMem, &totalMem);
  std::cout << "After allocation: Free memory: " << freeMem << " bytes, Total memory: " << totalMem << " bytes\n";

  // Copy matrices to GPU
  cudaMemcpy(d_x, x, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, K * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_z, 0, M * N * sizeof(float));
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
  matrixMul(d_x, d_y, d_z, M, N, K);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  // Copy result back to host
  cudaMemcpy(z, d_z, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "z[0]: " << z[0] << " z[N-1]: " << z[N -1] << std::endl;

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  return 0;
}
