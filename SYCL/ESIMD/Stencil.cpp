//==---------------- Stencil.cpp  - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

#define WIDTH 16
#define HEIGHT 16

using namespace cl::sycl;

void InitializeSquareMatrix(float *matrix, size_t const Dim,
                            bool const bSkipDataGeneration) {
  memset(matrix, 0, Dim * Dim * sizeof(float));
  if (!bSkipDataGeneration) {
    for (unsigned int iRow = 0; iRow < Dim; ++iRow) {
      for (unsigned int iCol = 0; iCol < Dim; ++iCol) {
        matrix[iRow * Dim + iCol] = static_cast<float>(iRow + iCol);
      }
    }
  }
}

bool CheckResults(float *out, float *in, unsigned n) {
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if ((5 <= i) && (i < n - 5) && (5 <= j) && (j < n - 5)) {
        float res = +in[(i - 5) * n + (j + 0)] * -0.02f +
                    in[(i - 4) * n + (j + 0)] * -0.025f +
                    in[(i - 3) * n + (j + 0)] * -0.0333333333333f +
                    in[(i - 2) * n + (j + 0)] * -0.05f +
                    in[(i - 1) * n + (j + 0)] * -0.1f +
                    in[(i + 0) * n + (j - 5)] * -0.02f +
                    in[(i + 0) * n + (j - 4)] * -0.025f +
                    in[(i + 0) * n + (j - 3)] * -0.0333333333333f +
                    in[(i + 0) * n + (j - 2)] * -0.05f +
                    in[(i + 0) * n + (j - 1)] * -0.1f +
                    in[(i + 0) * n + (j + 1)] * 0.1f +
                    in[(i + 0) * n + (j + 2)] * 0.05f +
                    in[(i + 0) * n + (j + 3)] * 0.0333333333333f +
                    in[(i + 0) * n + (j + 4)] * 0.025f +
                    in[(i + 0) * n + (j + 5)] * 0.02f +
                    in[(i + 1) * n + (j + 0)] * 0.1f +
                    in[(i + 2) * n + (j + 0)] * 0.05f +
                    in[(i + 3) * n + (j + 0)] * 0.0333333333333f +
                    in[(i + 4) * n + (j + 0)] * 0.025f +
                    in[(i + 5) * n + (j + 0)] * 0.02f;

        // check result
        if (abs(res - out[i * n + j]) >= 0.0015f) {
          std::cout << "out[" << i << "][" << j << "] = " << out[i * n + j]
                    << " expect result " << res << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  if (argc > 2) {
    std::cerr << "Usage: stencil.exe [dim_size]" << std::endl;
    exit(1);
  }
  // Default dimension size is 1024
  const unsigned DIM_SIZE = (argc == 2) ? atoi(argv[1]) : 1 << 10;
  const unsigned SQUARE_SZ = DIM_SIZE * DIM_SIZE + 16;
  uint range_width =
      (DIM_SIZE - 10) / WIDTH + (((DIM_SIZE - 10) % WIDTH == 0) ? 0 : 1);
  uint range_height =
      (DIM_SIZE - 10) / HEIGHT + (((DIM_SIZE - 10) % HEIGHT == 0) ? 0 : 1);
  cl::sycl::range<2> GlobalRange{range_width, range_height};

  std::cout << "width = " << range_width << " height = " << range_height
            << std::endl;
  cl::sycl::range<2> LocalRange{1, 1};

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(),
          property::queue::enable_profiling{});

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();

  // create and init matrices
  float *inputMatrix =
      static_cast<float *>(malloc_shared(SQUARE_SZ * sizeof(float), dev, ctxt));
  float *outputMatrix =
      static_cast<float *>(malloc_shared(SQUARE_SZ * sizeof(float), dev, ctxt));
  InitializeSquareMatrix(inputMatrix, DIM_SIZE, false);
  InitializeSquareMatrix(outputMatrix, DIM_SIZE, true);

  // Start Timer
  esimd_test::Timer timer;
  double start;

  double kernel_times = 0;
  unsigned num_iters = 10;

  try {
    for (int iter = 0; iter <= num_iters; ++iter) {
      auto e = q.submit([&](handler &cgh) {
        cgh.parallel_for<class Stencil_kernel>(
            GlobalRange * LocalRange, [=](item<2> it) SYCL_ESIMD_KERNEL {
              using namespace sycl::ext::intel::esimd;
              uint h_pos = it.get_id(0);
              uint v_pos = it.get_id(1);

              simd<float, (HEIGHT + 10) * 32> vin;
              // matrix HEIGHT+10 x 32
              auto in = vin.bit_cast_view<float, HEIGHT + 10, 32>();

              //
              // rather than loading all data in
              // the code will interleave data loading and compute
              // first, we load enough data for the first 16 pixels
              //
              unsigned off = (v_pos * HEIGHT) * DIM_SIZE + h_pos * WIDTH;
#pragma unroll
              for (unsigned i = 0; i < 10; i++) {
                simd<float, 32> data;
                data.copy_from(inputMatrix + off);
                in.row(i) = data;
                off += DIM_SIZE;
              }

              unsigned out_off =
                  (((v_pos * HEIGHT + 5) * DIM_SIZE + (h_pos * WIDTH) + 5)) *
                  sizeof(float);
              simd<unsigned, WIDTH> elm16(0, 1);

#pragma unroll
              for (unsigned i = 0; i < HEIGHT; i++) {
                simd<float, 32> data;
                data.copy_from(inputMatrix + off);
                in.row(10 + i) = data;
                off += DIM_SIZE;

                simd<float, WIDTH> sum =
                    in.row(i + 0).select<WIDTH, 1>(5) * -0.02f +
                    in.row(i + 1).select<WIDTH, 1>(5) * -0.025f +
                    in.row(i + 2).select<WIDTH, 1>(5) * -0.0333333333333f +
                    in.row(i + 3).select<WIDTH, 1>(5) * -0.05f +
                    in.row(i + 4).select<WIDTH, 1>(5) * -0.1f +
                    in.row(i + 5).select<WIDTH, 1>(0) * -0.02f +
                    in.row(i + 5).select<WIDTH, 1>(1) * -0.025f +
                    in.row(i + 5).select<WIDTH, 1>(2) * -0.0333333333333f +
                    in.row(i + 5).select<WIDTH, 1>(3) * -0.05f +
                    in.row(i + 5).select<WIDTH, 1>(4) * -0.1f +
                    in.row(i + 5).select<WIDTH, 1>(6) * 0.1f +
                    in.row(i + 5).select<WIDTH, 1>(7) * 0.05f +
                    in.row(i + 5).select<WIDTH, 1>(8) * 0.0333333333333f +
                    in.row(i + 5).select<WIDTH, 1>(9) * 0.025f +
                    in.row(i + 5).select<WIDTH, 1>(10) * 0.02f +
                    in.row(i + 6).select<WIDTH, 1>(5) * 0.1f +
                    in.row(i + 7).select<WIDTH, 1>(5) * 0.05f +
                    in.row(i + 8).select<WIDTH, 1>(5) * 0.0333333333333f +
                    in.row(i + 9).select<WIDTH, 1>(5) * 0.025f +
                    in.row(i + 10).select<WIDTH, 1>(5) * 0.02f;

                // predciate output
                simd_mask<WIDTH> p = (elm16 + h_pos * WIDTH) < (DIM_SIZE - 10);

                simd<unsigned, WIDTH> elm16_off =
                    elm16 * sizeof(float) + out_off;
                scatter<float, WIDTH>(outputMatrix, sum, elm16_off, p);
                out_off += DIM_SIZE * sizeof(float);

                if (v_pos * HEIGHT + 10 + i >= DIM_SIZE - 1)
                  break;
              }
            });
      });
      e.wait();
      double etime = esimd_test::report_time("kernel time", e, e);
      if (iter > 0)
        kernel_times += etime;
      else
        start = timer.Elapsed();
    }
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(inputMatrix, ctxt);
    free(outputMatrix, ctxt);
    return 1;
  }

  // End timer.
  double end = timer.Elapsed();

  esimd_test::display_timing_stats(kernel_times, num_iters,
                                   (end - start) * 1000);

  // check result
  bool passed = CheckResults(outputMatrix, inputMatrix, DIM_SIZE);
  if (passed) {
    std::cout << "PASSED" << std::endl;
  } else {
    std::cout << "FAILED" << std::endl;
  }
  free(inputMatrix, ctxt);
  free(outputMatrix, ctxt);
  return 0;
}
