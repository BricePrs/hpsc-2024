#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <iostream>
#include <chrono>




__global__ void bucket_sort(int* d_bucket, int* key, int range, int n) {

  int keyIdx = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ int shared_bucket[];
  shared_bucket[threadIdx.x] = 0;
  if (blockIdx.x == 0) {
    d_bucket[threadIdx.x] = 0;
  }
  __syncthreads();


  if (keyIdx < n) {
    atomicAdd(shared_bucket + key[keyIdx], 1);
  }
  __syncthreads();

  atomicAdd(d_bucket + threadIdx.x, shared_bucket[threadIdx.x]);
  __syncthreads();

  if (blockIdx.x != 0) { return; }

  int offset = 0;
  for (int j = 0; j < threadIdx.x; j++) {
    offset += d_bucket[j];
  }

  int c = d_bucket[threadIdx.x];
  for (; c>0; c--) {
    key[offset++] = threadIdx.x;
  }
}



int main() {
  int n = 1<<28;
  int range = 1024;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    //printf("%d ",key[i]);
  }
  printf("\n");

  int group_size = range;
  int group_count = (n + group_size - 1) / group_size;
  std::vector<int> bucket(range);


  int *d_bucket, *d_key;
  cudaMalloc(&d_key, key.size()*sizeof(int));
  cudaMalloc(&d_bucket, range*sizeof(int)*group_count);
  cudaMemcpy(d_key, key.data(), key.size() * sizeof(int), cudaMemcpyHostToDevice);
  auto start = std::chrono::high_resolution_clock::now(); // Start measuring time
  bucket_sort<<<group_count, group_size, range*sizeof(int)>>>(d_bucket, d_key, range, n);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time
  cudaMemcpy(key.data(), d_key, key.size() * sizeof(int), cudaMemcpyDeviceToHost);


  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Time taken (without GPU memory back and forth transfer): " << elapsed_seconds.count() << " seconds" << std::endl;


  for (int i=0; i<n; i++) {
    //printf("%d ",key[i]);
  }
  printf("\n");
}
