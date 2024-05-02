#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <chrono>

int main() {
  int n = 1<<29;
  int range = 256;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    //printf("%d ",key[i]);
  }
  printf("\n");
  auto start = std::chrono::high_resolution_clock::now(); // Start measuring time

  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }



  auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;

  for (int i=0; i<n; i++) {
    //printf("%d ",key[i]);
  }
  printf("\n");
}
