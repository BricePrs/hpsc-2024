#include <iostream>
#include <vector>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const float dx = 2.0f / (nx - 1);
const float dy = 2.0f / (ny - 1);
const float dt = 0.01f;
const float rho = 1.0f;
const float nu = 0.02f;

__global__ void build_up_b(float* b, float* u, float* v, int nx, int ny, float rho, float dt, float dx, float dy) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > 0 && j < ny - 1 && i > 0 && i < nx - 1) {
    b[j * nx + i] = rho * (1.0f / dt * ((u[j * nx + (i + 1)] - u[j * nx + (i - 1)]) / (2.0f * dx) +
                                        (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0f * dy)) -
                           ((u[j * nx + (i + 1)] - u[j * nx + (i - 1)]) / (2.0f * dx)) *
                           ((u[j * nx + (i + 1)] - u[j * nx + (i - 1)]) / (2.0f * dx)) -
                           2.0f * ((u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2.0f * dy) *
                                   (v[j * nx + (i + 1)] - v[j * nx + (i - 1)]) / (2.0f * dx)) -
                           ((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0f * dy)) *
                           ((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0f * dy)));
  }
}

__global__ void pressure_poisson(float* p, float* pn, float* b, int nx, int ny, float dx, float dy) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > 0 && j < ny - 1 && i > 0 && i < nx - 1) {
    p[j * nx + i] = ((pn[j * nx + (i + 1)] + pn[j * nx + (i - 1)]) * dy * dy +
                     (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]) * dx * dx -
                     b[j * nx + i] * dx * dx * dy * dy) /
                    (2.0f * (dx * dx + dy * dy));
  }
}

__global__ void velocity_update(float* u, float* v, float* un, float* vn, float* p, int nx, int ny, float dx, float dy, float dt, float rho, float nu) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > 0 && j < ny - 1 && i > 0 && i < nx - 1) {
    u[j * nx + i] = un[j * nx + i] - un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + (i - 1)]) -
                    un[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) -
                    dt / (2.0f * rho * dx) * (p[j * nx + (i + 1)] - p[j * nx + (i - 1)]) +
                    nu * dt / (dx * dx) * (un[j * nx + (i + 1)] - 2.0f * un[j * nx + i] + un[j * nx + (i - 1)]) +
                    nu * dt / (dy * dy) * (un[(j + 1) * nx + i] - 2.0f * un[j * nx + i] + un[(j - 1) * nx + i]);

    v[j * nx + i] = vn[j * nx + i] - vn[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + (i - 1)]) -
                    vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]) -
                    dt / (2.0f * rho * dx) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) +
                    nu * dt / (dx * dx) * (vn[j * nx + (i + 1)] - 2.0f * vn[j * nx + i] + vn[j * nx + (i - 1)]) +
                    nu * dt / (dy * dy) * (vn[(j + 1) * nx + i] - 2.0f * vn[j * nx + i] + vn[(j - 1) * nx + i]);
  }
}

void boundary_conditions(float* u, float* v, float* p, int nx, int ny) {
  for (int j = 0; j < ny; j++) {
    u[j * nx] = 0.0f;
    u[j * nx + (nx - 1)] = 0.0f;
    v[j * nx] = 0.0f;
    v[j * nx + (nx - 1)] = 0.0f;
  }
  for (int i = 0; i < nx; i++) {
    u[i] = 0.0f;
    u[(ny - 1) * nx + i] = 1.0f;
    v[i] = 0.0f;
    v[(ny - 1) * nx + i] = 0.0f;
  }
}

int main() {
  thrust::host_vector<float> h_u(ny * nx, 0.0f);
  thrust::host_vector<float> h_v(ny * nx, 0.0f);
  thrust::host_vector<float> h_p(ny * nx, 0.0f);
  thrust::host_vector<float> h_b(ny * nx, 0.0f);

  thrust::device_vector<float> d_u = h_u;
  thrust::device_vector<float> d_v = h_v;
  thrust::device_vector<float> d_p = h_p;
  thrust::device_vector<float> d_b = h_b;
  thrust::device_vector<float> d_un(ny * nx);
  thrust::device_vector<float> d_vn(ny * nx);
  thrust::device_vector<float> d_pn(ny * nx);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

  for (int n = 0; n < nt; n++) {
    build_up_b<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_u.data()), thrust::raw_pointer_cast(d_v.data()), nx, ny, rho, dt, dx, dy);
    cudaDeviceSynchronize();

    for (int it = 0; it < nit; it++) {
      thrust::copy(d_p.begin(), d_p.end(), d_pn.begin());
      pressure_poisson<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_p.data()), thrust::raw_pointer_cast(d_pn.data()), thrust::raw_pointer_cast(d_b.data()), nx, ny, dx, dy);
      cudaDeviceSynchronize();
    }

    thrust::copy(d_u.begin(), d_u.end(), d_un.begin());
    thrust::copy(d_v.begin(), d_v.end(), d_vn.begin());

    velocity_update<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_u.data()), thrust::raw_pointer_cast(d_v.data()), thrust::raw_pointer_cast(d_un.data()), thrust::raw_pointer_cast(d_vn.data()), thrust::raw_pointer_cast(d_p.data()), nx, ny, dx, dy, dt, rho, nu);
    cudaDeviceSynchronize();

    boundary_conditions(thrust::raw_pointer_cast(d_u.data()), thrust::raw_pointer_cast(d_v.data()), thrust::raw_pointer_cast(d_p.data()), nx, ny);
  }

  h_u = d_u;
  h_v = d_v;
  h_p = d_p;

  // Optionally, print or visualize the results here

  return 0;
}
