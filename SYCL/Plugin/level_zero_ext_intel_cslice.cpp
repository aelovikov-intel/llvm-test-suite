// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env ZEX_NUMBER_OF_CCS=0:4 env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-PVC

// Requires: level_zero

#include <sycl/sycl.hpp>

using namespace sycl;

void test_pvc(device &d) {
  std::cout << "Test PVC Begin" << std::endl;
  // CHECK-PVC: Test PVC Begin
  bool IsPVC = [&]() {
    if (!d.has(aspect::ext_intel_device_id))
      return false;
    return (d.get_info<ext::intel::info::device::device_id>() & 0xff0) == 0xbd0;
  }();
  std::cout << "IsPVC: " << std::boolalpha << IsPVC << std::endl;
  if (IsPVC) {
    auto Contains = [](auto Range, auto Elem) {
      return std::find(Range.begin(), Range.end(), Elem) != Range.end();
    };
    auto PartitionableBy = [&](device &d, info::partition_property Prop) {
      return Contains(d.get_info<info::device::partition_properties>(), Prop);
    };
    auto PartitionableByCSlice = [&](device &d) {
      return PartitionableBy(
          d, info::partition_property::ext_intel_partition_by_cslice);
    };
    auto PartitionableByAffinityDomain = [&](device &d) {
      return PartitionableBy(
          d, info::partition_property::partition_by_affinity_domain);
    };

    assert(PartitionableByAffinityDomain(d));
    assert(!PartitionableByCSlice(d));
    {
      try {
        std::ignore = d.create_sub_devices<
            info::partition_property::ext_intel_partition_by_cslice>();
        assert(false && "Expected an exception to be thrown earlier!");
      } catch (sycl::exception &e) {
        assert(e.code() == errc::feature_not_supported);
      }
    }

    auto sub_devices = d.create_sub_devices<
        info::partition_property::partition_by_affinity_domain>(
        info::partition_affinity_domain::next_partitionable);
    device &sub_device = sub_devices[1];
    assert(!PartitionableByAffinityDomain(sub_device));
    assert(PartitionableByCSlice(sub_device));
    assert(sub_device.get_info<info::device::partition_type_property>() ==
           info::partition_property::partition_by_affinity_domain);

    {
      try {
        std::ignore = sub_device.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::next_partitionable);
        assert(false && "Expected an exception to be thrown earlier!");
      } catch (sycl::exception &e) {
        assert(e.code() == errc::feature_not_supported);
      }
    }

    auto sub_sub_devices = sub_device.create_sub_devices<
        info::partition_property::ext_intel_partition_by_cslice>();
    auto &sub_sub_device = sub_sub_devices[0];
    assert(!PartitionableByAffinityDomain(sub_sub_device));
    assert(!PartitionableByCSlice(sub_sub_device));
    assert(sub_sub_device.get_info<info::device::partition_type_property>() ==
           info::partition_property::ext_intel_partition_by_cslice);
  } else {
    // Make FileCheck pass.
    std::cout << "Fake ZE_DEBUG output for FileCheck:" << std::endl;
    // clang-format off
    // clang-format on
  }
  std::cout << "Test PVC End" << std::endl;
  // CHECK-PVC: Test PVC End
}

int main() {
  device d;

  test_pvc(d);

  return 0;
}
