#include <iostream>
#include <vector>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <fstream>
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
                    vn[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) -
                    dt / (2.0f * rho * dx) * (p[j * nx + (i + 1)] - p[j * nx + (i - 1)]) +
                    nu * dt / (dx * dx) * (un[j * nx + (i + 1)] - 2.0f * un[j * nx + i] + un[j * nx + (i - 1)]) +
                    nu * dt / (dy * dy) * (un[(j + 1) * nx + i] - 2.0f * un[j * nx + i] + un[(j - 1) * nx + i]);

    v[j * nx + i] = vn[j * nx + i] - un[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + (i - 1)]) -
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

__global__ void set_boundary_conditions(float* u, float* v, float* p, int nx, int ny) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < ny) {
    // u[:, 0]  = 0
    u[idx * nx] = 0.0f;
    // u[:, -1] = 0
    u[idx * nx + (nx - 1)] = 0.0f;
    // v[:, 0]  = 0
    v[idx * nx] = 0.0f;
    // v[:, -1] = 0
    v[idx * nx + (nx - 1)] = 0.0f;
  }

  if (idx < nx) {
    // u[0, :]  = 0
    u[idx] = 0.0f;
    // u[-1, :] = 1
    u[(ny - 1) * nx + idx] = 1.0f;
    // v[0, :]  = 0
    v[idx] = 0.0f;
    // v[-1, :] = 0
    v[(ny - 1) * nx + idx] = 0.0f;
  }
}

__global__ void apply_boundary_conditions_p(float* p, int nx, int ny) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nx) {
    // Top boundary (p[0, :] = p[1, :])
    p[idx] = p[idx + nx];
    // Bottom boundary (p[-1, :] = 0)
    p[(ny - 1) * nx + idx] = 0.0f;
  }
  if (idx < ny) {
    // Left boundary (p[:, 0] = p[:, 1])
    p[idx*nx] = p[nx*idx+1];
    // Right boundary (p[:, -1] = p[:, -2])
    p[idx*nx + nx - 1] = p[idx * nx + nx - 2];
  }
}

void save_to_file(std::ofstream& file, const std::vector<float>& data, int nx, int ny) {
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      file << data[j * nx + i] << " ";
    }
  }
  file << "\n";
}

int main() {

  int N = ny * nx;
  std::vector<float> h_u(N, 0.0f);
  std::vector<float> h_v(N, 0.0f);
  std::vector<float> h_p(N, 0.0f);
  std::vector<float> h_b(N, 0.0f);
  float* d_u;  cudaMalloc(&d_u , N*sizeof(float)); cudaMemcpy(d_u, h_u.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  float* d_v;  cudaMalloc(&d_v , N*sizeof(float)); cudaMemcpy(d_u, h_v.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  float* d_p;  cudaMalloc(&d_p , N*sizeof(float)); cudaMemcpy(d_u, h_p.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  float* d_b;  cudaMalloc(&d_b , N*sizeof(float)); cudaMemcpy(d_u, h_b.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  float* d_un; cudaMalloc(&d_un, N*sizeof(float));
  float* d_vn; cudaMalloc(&d_vn, N*sizeof(float));
  float* d_pn; cudaMalloc(&d_pn, N*sizeof(float));

  std::ofstream u_file("u.dat", std::ios::out | std::ios::trunc);
  std::ofstream v_file("v.dat", std::ios::out | std::ios::trunc);
  std::ofstream p_file("p.dat", std::ios::out | std::ios::trunc);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

  for (int n = 0; n < nt; n++) {

    build_up_b<<<numBlocks, threadsPerBlock>>>(d_b, d_u, d_v, nx, ny, rho, dt, dx, dy);
    cudaDeviceSynchronize();
    for (int it = 0; it < nit; it++) {
      cudaMemcpy(d_pn, d_p, N*sizeof(float), cudaMemcpyDeviceToDevice);
      pressure_poisson<<<numBlocks, threadsPerBlock>>>(d_p, d_pn, d_b, nx, ny, dx, dy);
      cudaDeviceSynchronize();
      apply_boundary_conditions_p<<<(max(nx, ny) + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x>>>(d_p, nx, ny);
      cudaDeviceSynchronize();
    }

    cudaMemcpy(d_un, d_u, N*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vn, d_v, N*sizeof(float), cudaMemcpyDeviceToDevice);

    velocity_update<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_un, d_vn, d_p, nx, ny, dx, dy, dt, rho, nu);
    cudaDeviceSynchronize();

    //boundary_conditions(d_u, d_v, d_p, nx, ny);
    int numBoundaryThreads = max(nx, ny);
    set_boundary_conditions<<<(numBoundaryThreads + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x>>>(d_u, d_v, d_p, nx, ny);
    cudaDeviceSynchronize();

    cudaMemcpy(h_u.data(), d_u, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v.data(), d_v, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p.data(), d_p, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    save_to_file(u_file, h_u, nx, ny);
    save_to_file(v_file, h_v, nx, ny);
    save_to_file(p_file, h_p, nx, ny);

  }

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_p);
  cudaFree(d_b);
  cudaFree(d_un);
  cudaFree(d_vn);
  cudaFree(d_pn);

	u_file.close();
	v_file.close();
	p_file.close();

  return 0;
}
