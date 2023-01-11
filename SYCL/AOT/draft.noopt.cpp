// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out /Od
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "draft.cpp"
