#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <iostream>
#include <chrono>




__global__ void bucket_sort(int* d_bucket, int* key, int range, int n) {
  extern __shared__ int shared_bucket[];
  int keyIdx = blockIdx.x * blockDim.x + threadIdx.x;
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
}

__global__ void fill_key(int* d_bucket, int* d_bucket_acc, int* key, int range, int n) {
  int c = d_bucket[threadIdx.x];
  int offset = d_bucket_acc[threadIdx.x];
  for (; c>0; c--) {
    key[offset++] = threadIdx.x;
  }
}




int main() {
  int n = 1<<29;
  int range = 128;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    //printf("%d ",key[i]);
  }
  printf("\n");

  int group_size = range;
  int group_count = (n + group_size - 1) / group_size;
  std::vector<int> bucket(range);


  int *d_bucket, *d_key, *d_bucket_acc;
  cudaMalloc(&d_key, key.size()*sizeof(int));
  cudaMalloc(&d_bucket, range*sizeof(int)*group_count);
  cudaMalloc(&d_bucket_acc, range*sizeof(int)*group_count);
  cudaMemcpy(d_key, key.data(), key.size() * sizeof(int), cudaMemcpyHostToDevice);
  auto start = std::chrono::high_resolution_clock::now(); // Start measuring time
  bucket_sort<<<group_count, group_size, range*sizeof(int)>>>(d_bucket, d_key, range, n);
  cudaDeviceSynchronize();
  cudaMemcpy(bucket.data(), d_bucket, bucket.size() * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 1; i < range; ++i) {
    bucket[i] += bucket[i-1];
  }
  cudaMemcpy(d_bucket_acc, bucket.data(), bucket.size() * sizeof(int), cudaMemcpyHostToDevice);
  fill_key<<<1, range>>>(d_bucket, d_bucket_acc, d_key, range, n);
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
