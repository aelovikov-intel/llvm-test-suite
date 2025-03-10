//==------------ lsc_block_prefetch_u8.cpp - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "Inputs/lsc_block_load.hpp"

constexpr uint32_t seed = 322;
using T = uint8_t;

constexpr cache_hint L1H = cache_hint::cached;
constexpr cache_hint L3H = cache_hint::uncached;

int main(void) {
  srand(seed);
  bool passed = true;

  // These parameters require unpadding. It is not implemented yet
  // passed &= test<0, T, 2, 2, 2, 2>(16, 4, 16, 1, 1);

  // non transposed, non transformed
  passed &= test<1, T, 1, 1, 16, 32, 2, false, false, L1H, L3H, true>(
      40, 64, 64, 4, 21);
  passed &=
      test<2, T, 2, 2, 8, 8, 2, false, false, L1H, L3H, true>(16, 16, 64, 8, 5);
  passed &= test<3, T, 1, 1, 8, 32, 2, false, false, L1H, L3H, true>(16, 80, 64,
                                                                     4, 1);

  // transformed
  passed &= test<4, T, 1, 1, 16, 4, 4, false, true, L1H, L3H, true>(100, 10,
                                                                    128, 16, 5);
  passed &= test<5, T, 1, 1, 12, 20, 1, false, true, L1H, L3H, true>(16, 40, 64,
                                                                     0, 0);
  passed &=
      test<6, T, 1, 1, 16, 4, 2, false, true, L1H, L3H, true>(32, 4, 64, 4, 1);
  passed &=
      test<7, T, 2, 2, 4, 16, 2, false, true, L1H, L3H, true>(4, 20, 64, 0, 3);
  passed &= test<8, T, 1, 1, 16, 32, 1, false, true, L1H, L3H, true>(24, 80, 64,
                                                                     4, 14);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
