#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cassert>

void bucket_sort(std::vector<int> &vector, int range) {
 
   std::vector<int> bucket(range,0); 
  for (auto &elt :vector)
      bucket[elt]++;
  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      vector[j++] = i;
    }
  }

}

void par_bucket_sort(std::vector<int> &vector, int range) {
 
  std::vector<int> bucket(range,0); 
  for (auto &elt :vector)
      bucket[elt]++;
  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];
  omp_set_num_threads(range);
#pragma parallel for firstprivate(bucket) lastprivate(vector)
  for (int i=0; i<range; i++) {
    //printf("Using thread %i", omp_get_num_threads());
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      vector[j++] = i;
    }
  }

}

std::vector <int> create_random_vec(int n, int range) {
  
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    // printf("%d ",key[i]);
  }
  return key;
}

int main() {
	int n = 100000000;
	int range = 100;

	{
		auto key = create_random_vec(n, range);
		auto start = std::chrono::high_resolution_clock::now(); // Start measuring time
		bucket_sort(key, range);
		auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;
	}
	{
		auto key = create_random_vec(n, range);
		auto start = std::chrono::high_resolution_clock::now(); // Start measuring time
		par_bucket_sort(key, range);
		auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;
		int previous = key[0];
		for (int i=1; i<n; i++) {
			assert(key[i] >= previous);
			previous = key[i];
		}
	}

}
