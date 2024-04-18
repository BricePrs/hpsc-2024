#include <cstdio>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <tuple>
#include <cmath>

#define NUM_THREADS 4


double DefaultSolution(long int n) {
	auto start = std::chrono::high_resolution_clock::now();
	double dx = 1. / n;
	double pi = 0;
	for (int i=0; i<n; i++) {
		double x = (i + 0.5) * dx;
		pi += 4.0 / (1.0 + x * x) * dx;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	return duration.count();
}

std::tuple<double, double> MesureTimeFor(long int n, int NumThread) {
	
	auto start = std::chrono::high_resolution_clock::now();
	
	double dx = 1. / n;
	double pi = 0;
	omp_set_num_threads(NumThread);
	double pi_thread[NumThread];
	for (int i = 0; i < NumThread; i++) {
		pi_thread[i] = 0;
	}
	#pragma omp parallel for firstprivate(dx, n)
	for (long int i=0; i<n; i++) {
		double x = (i + 0.5) * dx;
		pi_thread[omp_get_thread_num()]+= 4.0 / (1.0 + x * x) * dx;
	}
	auto end_parallel = std::chrono::high_resolution_clock::now();


	for (int i = 0; i < NumThread; i++) {
		pi += pi_thread[i];
	}

	auto end = std::chrono::high_resolution_clock::now();
	//printf("%17.15f\n",pi);

	// Calculate the duration
	std::chrono::duration<double> duration = end - start;
	std::chrono::duration<double> duration_parallel = end_parallel - start;
	return std::tuple(duration.count(), duration_parallel.count());
}

int main() {
	

	// for (int i = 0; i < 9; i++) {
	// 	long int n = powf(10, i);
	// 	auto duration = MesureTimeFor(n, NUM_THREADS);
	// 	printf("Elapsed time for n=10^%i and t=%i: %fs\n", i, NUM_THREADS, duration.count());
	// }

	int exp = 9;
	long int n = powf(10, exp);
	auto defaultTimeSolution = DefaultSolution(n);

	for (int i = 5; i < 12; i++) {
		int t = 1 << i;
		auto duration = MesureTimeFor(n, t);
		printf("%.2f Elapsed time for n=10^%i and t=%i: %fs parallel_portion:%f\n ", std::get<0>(duration)/defaultTimeSolution, exp, t, std::get<0>(duration), std::get<1>(duration)/std::get<0>(duration));
	}


}
