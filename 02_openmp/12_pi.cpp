#include <cstdio>
#include <omp.h>
#include <chrono>
#include <iostream>

#define NUM_THREADS 32

int main() {
  auto start = std::chrono::high_resolution_clock::now();
  long int n = 10000000000;
  double dx = 1. / n;
  double pi = 0;
  double pi_thread[NUM_THREADS];
  omp_set_num_threads(NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; i++) {
    pi_thread[i] = 0;
  }
  #pragma omp parallel for
  for (long int i=0; i<n; i++) {
    //int id = omp_get_thread_num();
    //printf("%d\n",id);
    double x = (i + 0.5) * dx;
    pi_thread[omp_get_thread_num()]+= 4.0 / (1.0 + x * x) * dx;
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    pi += pi_thread[i];
  }

  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<double> duration = end - start;
  
  // Print the elapsed time
  std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
    

  printf("%17.15f\n",pi);
}
