#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>
#include <chrono>

void SimdSolution(float *x, float *y, float *fx, float *fy, float *m, int N) {

	for(int i=0; i<N; i++) {
		__m512 xi_vec = _mm512_set1_ps(x[i]);
		__m512 yi_vec = _mm512_set1_ps(y[i]);
		for(int j=0; j<N; j+=16) {
			__m512 m_vec = _mm512_load_ps(&m[j]);
			__m512 x_vec = _mm512_load_ps(&x[j]);
			__m512 y_vec = _mm512_load_ps(&y[j]);

			__m512 rx_vec = _mm512_sub_ps(xi_vec, x_vec);
			__m512 ry_vec = _mm512_sub_ps(yi_vec, y_vec);

			__m512 r_vec = _mm512_add_ps(
					_mm512_mul_ps(rx_vec, rx_vec),
					_mm512_mul_ps(ry_vec, ry_vec)
			);

			__m512 rn_vec = _mm512_rsqrt14_ps(r_vec);
			__m512 rn3_vec = _mm512_mul_ps(_mm512_mul_ps(rn_vec, rn_vec), rn_vec);

			unsigned short mask = -1;
			mask ^= (1 << i) >> j;
      //printf("mask = %i", mask);

			__m512 temp = _mm512_mul_ps(m_vec, rn3_vec);
			//temp = _mm512_mask_blend_ps(mask, _mm512_set1_ps(0), temp);

      float red_temp_x = _mm512_reduce_add_ps(_mm512_mul_ps(temp, rx_vec));
      float red_temp_y = _mm512_reduce_add_ps(_mm512_mul_ps(temp, ry_vec));

			fx[i] -= red_temp_x;
			fy[i] -= red_temp_y;
      printf("%d %g %g\n",i,fx[i],fy[i]);
    }

	}
}

void DefaultSolution(float *x, float *y, float *fx, float *fy, float *m, int N) {
	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			float rx = x[i] - x[j];
			float ry = y[i] - y[j];
			float r = std::sqrt(rx * rx + ry * ry);
			fx[i] -= rx * m[j] / (r * r * r);
			fy[i] -= ry * m[j] / (r * r * r);
      if (j % 16 == 15) {
        printf("%d %g %g\n",i,fx[i],fy[i]);
      }

		}
	}
}

int main() {
	const int N = 128;
	float x[N], y[N], m[N], fx[N], fy[N];

	srand48(0);
	for(int i=0; i<N; i++) {
		x[i] = drand48();
		y[i] = drand48();
		m[i] = drand48();
		fx[i] = fy[i] = 0;
    printf("x = %lf, y = %lf, fx = %lf, fy = %lf, m = %lf\n", x[i], y[i], fx[i], fy[i], m[i]);
	}


	auto start = std::chrono::high_resolution_clock::now();
	DefaultSolution(x, y, fx, fy, m, N);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end-start;

  srand48(0);
	for(int i=0; i<N; i++) {
		x[i] = drand48();
		y[i] = drand48();
		m[i] = drand48();
		fx[i] = fy[i] = 0;
    printf("x = %lf, y = %lf, fx = %lf, fy = %lf, m = %lf\n", x[i], y[i], fx[i], fy[i], m[i]);
	}
	auto par_start = std::chrono::high_resolution_clock::now();
	SimdSolution(x, y, fx, fy, m, N);
	auto par_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> par_elapsed = par_end-par_start;

	printf("Default: %lf s, SIMD: %lf s", elapsed.count(), par_elapsed.count());

}
