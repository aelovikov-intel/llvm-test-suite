// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PROGRAM_COMPILE_OPTIONS="-cl-opt-disable" %GPU_RUN_PLACEHOLDER %t.out

#include "draft.cpp"
