// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>
using namespace sycl;

template <int Dims> auto get_global_range(range<Dims> Range) { return Range; }
template <int Dims> auto get_global_range(nd_range<Dims> NDRange) {
  return NDRange.get_global_range();
}

template <int Dims, bool WithOffset>
auto get_global_id(item<Dims, WithOffset> Item) {
  return Item.get_id();
}
template <int Dims> auto get_global_id(nd_item<Dims> NDItem) {
  return NDItem.get_global_id();
}

template <int Dims> auto get_global_id(id<Dims> Id) { return Id; }

template <bool UseUSM, bool InitToIdentity,
          detail::reduction::strategy Strategy, typename RangeTy>
static void test(RangeTy Range) {
  queue q;

  // We can select strategy explicitly so no need to test all combinations of
  // types/operations.
  using T = int;
  using BinOpTy = std::plus<T>;

  T Init{19};

  auto Red = [&]() {
    if constexpr (UseUSM)
      return malloc_device<T>(1, q);
    else
      return buffer<T, 1>{1};
  }();
  auto GetRedAcc = [&](handler &cgh) {
    if constexpr (UseUSM)
      return Red;
    else
      return accessor{Red, cgh};
  };

  q.submit([&](handler &cgh) {
     auto RedAcc = GetRedAcc(cgh);
     cgh.single_task([=]() { RedAcc[0] = Init; });
   }).wait();

  q.submit([&](handler &cgh) {
     auto RedSycl = [&]() {
       if constexpr (UseUSM)
         if constexpr (InitToIdentity)
           return reduction(Red, BinOpTy{},
                            property::reduction::initialize_to_identity{});
         else
           return reduction(Red, BinOpTy{});
       else if constexpr (InitToIdentity)
         return reduction(Red, cgh, BinOpTy{},
                          property::reduction::initialize_to_identity{});
       else
         return reduction(Red, cgh, BinOpTy{});
     }();
     detail::reduction_parallel_for<detail::auto_name, Strategy>(
         cgh, Range, ext::oneapi::experimental::detail::empty_properties_t{},
         RedSycl, [=](auto Item, auto &Red) { Red.combine(T{1}); });
   }).wait();

  auto *Result = malloc_shared<T>(1, q);
  q.submit([&](handler &cgh) {
     auto RedAcc = GetRedAcc(cgh);
     cgh.single_task([=]() { *Result = RedAcc[0]; });
   }).wait();

  auto N = get_global_range(Range).size();
  int Expected = InitToIdentity ? N : Init + N;
#ifdef __PRETTY_FUNCTION__
  std::cout << __PRETTY_FUNCTION__ << ": " << *Result << ", expected "
            << Expected << std::endl;
#endif
  assert(*Result == Expected);

  if constexpr (UseUSM)
    free(Red, q);
  free(Result, q);
}

template <int... Inds, class F>
void loop_impl(std::integer_sequence<int, Inds...>, F &&f) {
  (f(std::integral_constant<int, Inds>{}), ...);
}

template <int count, class F> void loop(F &&f) {
  loop_impl(std::make_integer_sequence<int, count>{}, std::forward<F>(f));
}

template <bool UseUSM, bool InitToIdentity, typename RangeTy>
void testAllStrategies(RangeTy Range) {
  loop<(int)detail::reduction::strategy::multi>([&](auto Id) {
    constexpr auto Strategy =
        // Skip auto_select == 0.
        detail::reduction::strategy{decltype(Id)::value + 1};
    test<UseUSM, InitToIdentity, Strategy>(Range);
  });
}

int main() {
  auto TestRange = [](auto Range) {
    testAllStrategies<true, true>(Range);
    testAllStrategies<true, false>(Range);
    testAllStrategies<false, true>(Range);
    testAllStrategies<false, false>(Range);
  };

  TestRange(range<1>{42});
  TestRange(range<2>{8, 8});
  TestRange(range<3>{7, 7, 5});
  TestRange(nd_range<1>{range<1>{7}, range<1>{7}});
  TestRange(nd_range<1>{range<1>{3 * 3}, range<1>{3}});

  // TODO: Strategies historically adopted from sycl::range implementation only
  // support 1-Dim case.
  //
  // TestRange(nd_range<2>{range<2>{7, 3}, range<2> {7, 3}});
  // TestRange(nd_range<2>{range<2>{14, 9}, range<2> {7, 3}});
  return 0;
}
