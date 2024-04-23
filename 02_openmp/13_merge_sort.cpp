#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cassert>

void merge(int* vec, int begin, int mid, int end) {
    std::vector<int> tmp(end-begin+1);
    int left = begin;
    int right = mid+1;
    for (int i=0; i<tmp.size(); i++) {
        if (left > mid)
            tmp[i] = vec[right++];
        else if (right > end)
            tmp[i] = vec[left++];
        else if (vec[left] <= vec[right])
            tmp[i] = vec[left++];
        else
            tmp[i] = vec[right++];
    }
    auto size = tmp.size();
    for (int i=0; i<size; i++)
        vec[begin++] = tmp[i];
}

void par_merge(int* vec, int begin, int mid, int end) {
    std::vector<int> tmp(end-begin+1);
    int left = begin;
    int right = mid+1;
    for (int i=0; i<tmp.size(); i++) {
        if (left > mid)
            tmp[i] = vec[right++];
        else if (right > end)
            tmp[i] = vec[left++];
        else if (vec[left] <= vec[right])
            tmp[i] = vec[left++];
        else
            tmp[i] = vec[right++];
    }
    auto size = tmp.size();

#pragma omp parallel firstprivate(tmp, size, begin)
    for (int i=0; i<size; i++)
        vec[begin++] = tmp[i];
}

void merge_sort(int* vec, int begin, int end) {
    if(begin < end) {
        int mid = (begin + end) / 2;
        merge_sort(vec, begin, mid);
        merge_sort(vec, mid+1, end);
        merge(vec, begin, mid, end);
    }
}

void par_merge_sort(int* vec, int begin, int end) {
    int n = end + 1 - begin;
    if(begin < end) {
        int mid = (begin + end) / 2;
        if (n > 128) {

            #pragma omp task
            par_merge_sort(vec, begin, mid);
            #pragma omp task
            par_merge_sort(vec, mid+1, end);
            #pragma omp taskwait

        } else {
            merge_sort(vec, begin, end);
            merge_sort(vec, begin, end);
        }
        merge(vec, begin, mid, end);

    }
}

#define PRINT

int main() {
  //int n = 20;
  //std::vector<int> vec(n);
  //for (int i=0; i<n; i++) {
  //  vec[i] = rand() % (10 * n);
  //  printf("%d ",vec[i]);
  //}
  //printf("\n");
  //merge_sort(vec, 0, n-1);
  //for (int i=0; i<n; i++) {
  //  printf("%d ",vec[i]);
  //}
  //printf("\n");
    int n = 10000000;
    omp_set_num_threads(512);
    {

        std::vector<int> vec(n);
        for (int i = 0; i < n; i++) {
            vec[i] = rand() % (10 * n);
        }
        printf("\n");
        auto start = std::chrono::high_resolution_clock::now(); // Start measuring time
        merge_sort(vec.data(), 0, n - 1);
        auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time

        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;

    }

    {
        std::vector<int> vec(n);
        for (int i = 0; i < n; i++) {
            vec[i] = rand() % (10 * n);
#ifdef PRINT
            //printf("%d ",vec[i]);
#endif
        }
        printf("\n");
        auto start = std::chrono::high_resolution_clock::now(); // Start measuring time
#pragma omp parallel
        {
#pragma omp single
            par_merge_sort(vec.data(), 0, n - 1);
        }

        auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;

        int previous = vec[0];
        for (int i=1; i<n; i++) {
            assert(vec[i] >= previous);
            previous = vec[i];
        }

    }


    printf("\n");

}
