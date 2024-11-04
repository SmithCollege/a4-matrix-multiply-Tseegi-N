#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#define SIZE 1000000

// clock func
double  get_clock() {
  struct timeval tv;
  int ok;
  ok = gettimeofday(&tv, NULL);
  if (ok<0) {
    //printf('gettimeofday error\n');
  }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

void MatrixMulOnHost(float* M, float* N, float* P, int width) {
  for (int i = 0; i < width; ++i){
    for (int j = 0; j < width; ++j) {
      float sum = 0;
      for (int k = 0; k < width; ++k) {
        float a = M[i * width + k];
        float b = N[k * width + j];
        sum += a * b;
      }
      P[i * width + j] = sum;
    }
  }
}

int main() {
  float* x = malloc(sizeof(float) * SIZE * SIZE);
  float* y = malloc(sizeof(float) * SIZE * SIZE);
  float* z = malloc(sizeof(float) * SIZE * SIZE);

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      x[i * SIZE + j] = 1; // x[i][j]
      y[i * SIZE + j] = 1;
    }
  }

  // Get time
  double time0, time1;
  time0 = get_clock();

  MatrixMulOnHost(x, y, z, SIZE);

  // Final time
  time1 = get_clock();
  printf("time: %f seconds\n", (time1-time0));

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      if (z[i * SIZE + j] != SIZE) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * SIZE + j]);
      }
    }
  }
  printf("z[0]: %f \n", z[0]);
  printf(" z[N-1]: %f \n", z[SIZE-1]);


  return 0;
}
