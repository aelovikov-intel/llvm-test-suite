// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -O0
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;
struct my_sub_group_mask {
  static constexpr size_t max_bits = 32;

  // enable reference to individual bit
  struct reference {
    reference &operator=(bool x) {
      if (x) {
        Ref |= RefBit;
      } else {
        Ref &= ~RefBit;
      }
      return *this;
    }
    reference &operator=(const reference &x) {
      operator=((bool)x);
      return *this;
    }
    bool operator~() const { return !(Ref & RefBit); }
    operator bool() const { return Ref & RefBit; }
    reference &flip() {
      operator=(!(bool)*this);
      return *this;
    }

    reference(my_sub_group_mask &gmask, size_t pos) : Ref(gmask.Bits) {
      RefBit = (pos < gmask.bits_num) ? (1UL << pos) : 0;
    }

    // Reference to the word containing the bit
    uint32_t &Ref;
    // Bit mask where only referenced bit is set
    uint32_t RefBit;
  };

  bool operator[](id<1> id) const {
    return (Bits & ((id.get(0) < bits_num) ? (1UL << id.get(0)) : 0));
  }

  reference operator[](id<1> id) { return {*this, id.get(0)}; }
  bool test(id<1> id) const { return operator[](id); }
  bool all() const { return count() == bits_num; }
  bool any() const { return count() != 0; }
  bool none() const { return count() == 0; }
  uint32_t count() const {
    unsigned int count = 0;
    auto word = (Bits & valuable_bits(bits_num));
    while (word) {
      word &= (word - 1);
      count++;
    }
    return count;
  }
  uint32_t size() const { return bits_num; }
  id<1> find_low() const {
    size_t i = 0;
    while (i < size() && !operator[](i))
      i++;
    return {i};
  }
  id<1> find_high() const {
    size_t i = size() - 1;
    while (i > 0 && !operator[](i))
      i--;
    return {operator[](i) ? i : size()};
  }

  template <typename Type,
            typename = sycl::detail::enable_if_t<std::is_integral<Type>::value>>
  void insert_bits(Type bits, id<1> pos = 0) {
    size_t insert_size = sizeof(Type) * CHAR_BIT;
    uint32_t insert_data = (uint32_t)bits;
    insert_data <<= pos.get(0);
    uint32_t mask = 0;
    if (pos.get(0) + insert_size < size())
      mask |= (valuable_bits(bits_num) << (pos.get(0) + insert_size));
    if (pos.get(0) < size() && pos.get(0))
      mask |= (valuable_bits(max_bits) >> (max_bits - pos.get(0)));
    Bits &= mask;
    Bits += insert_data;
  }

  /* The bits are stored in the memory in the following way:
  marray id |     0     |     1     |     2     |     3     |...
  bit id    |7   ..    0|15   ..   8|23   ..  16|31  ..   24|...
  */
  template <typename Type, size_t Size,
            typename = sycl::detail::enable_if_t<std::is_integral<Type>::value>>
  void insert_bits(const marray<Type, Size> &bits, id<1> pos = 0) {
    size_t cur_pos = pos.get(0);
    for (auto elem : bits) {
      if (cur_pos < size()) {
        this->insert_bits(elem, cur_pos);
        cur_pos += sizeof(Type) * CHAR_BIT;
      }
    }
  }

  template <typename Type,
            typename = sycl::detail::enable_if_t<std::is_integral<Type>::value>>
  void extract_bits(Type &bits, id<1> pos = 0) const {
    auto Res = Bits;
    Res &= valuable_bits(bits_num);
    if (pos.get(0) < size()) {
      if (pos.get(0) > 0) {
        Res >>= pos.get(0);
      }

      if (sizeof(Type) * CHAR_BIT < max_bits) {
        Res &= valuable_bits(sizeof(Type) * CHAR_BIT);
      }
      bits = (Type)Res;
    } else {
      bits = 0;
    }
  }

  template <typename Type, size_t Size,
            typename = sycl::detail::enable_if_t<std::is_integral<Type>::value>>
  void extract_bits(marray<Type, Size> &bits, id<1> pos = 0) const {
    size_t cur_pos = pos.get(0);
    for (auto &elem : bits) {
      if (cur_pos < size()) {
        this->extract_bits(elem, cur_pos);
        cur_pos += sizeof(Type) * CHAR_BIT;
      } else {
        elem = 0;
      }
    }
  }

  template <typename Type, size_t Size,
            typename = sycl::detail::enable_if_t<std::is_integral<Type>::value>>
  void extract_bits_no_range_for(marray<Type, Size> &bits,
                                 id<1> pos = 0) const {
    size_t cur_pos = pos.get(0);
    for (int j = 0; j < 6; ++j) {
      Type &elem = bits[j];
      if (cur_pos < size()) {
        this->extract_bits(elem, cur_pos);
        cur_pos += sizeof(Type) * CHAR_BIT;
      } else {
        elem = 0;
      }
    }
  }

  template <typename Type, size_t Size,
            typename = sycl::detail::enable_if_t<std::is_integral<Type>::value>>
  void extract_bits_no_range_for_no_ref(marray<Type, Size> &bits,
                                        id<1> pos = 0) const {
    size_t cur_pos = pos.get(0);
    for (int j = 0; j < 6; ++j) {
      Type elem;
      if (cur_pos < size()) {
        this->extract_bits(elem, cur_pos);
        cur_pos += sizeof(Type) * CHAR_BIT;
      } else {
        elem = 0;
      }
      bits[j] = elem;
    }
  }

  void set() { Bits = valuable_bits(bits_num); }
  void set(id<1> id, bool value = true) { operator[](id) = value; }
  void reset() { Bits = uint32_t{0}; }
  void reset(id<1> id) { operator[](id) = 0; }
  void reset_low() { reset(find_low()); }
  void reset_high() { reset(find_high()); }
  void flip() { Bits = (~Bits & valuable_bits(bits_num)); }
  void flip(id<1> id) { operator[](id).flip(); }

  bool operator==(const my_sub_group_mask &rhs) const {
    return Bits == rhs.Bits;
  }
  bool operator!=(const my_sub_group_mask &rhs) const {
    return !(*this == rhs);
  }

  my_sub_group_mask &operator&=(const my_sub_group_mask &rhs) {
    Bits &= rhs.Bits;
    return *this;
  }
  my_sub_group_mask &operator|=(const my_sub_group_mask &rhs) {
    Bits |= rhs.Bits;
    return *this;
  }

  my_sub_group_mask &operator^=(const my_sub_group_mask &rhs) {
    Bits ^= rhs.Bits;
    Bits &= valuable_bits(bits_num);
    return *this;
  }

  my_sub_group_mask &operator<<=(size_t pos) {
    Bits <<= pos;
    Bits &= valuable_bits(bits_num);
    return *this;
  }

  my_sub_group_mask &operator>>=(size_t pos) {
    Bits >>= pos;
    return *this;
  }

  my_sub_group_mask operator~() const {
    auto Tmp = *this;
    Tmp.flip();
    return Tmp;
  }
  my_sub_group_mask operator<<(size_t pos) const {
    auto Tmp = *this;
    Tmp <<= pos;
    return Tmp;
  }
  my_sub_group_mask operator>>(size_t pos) const {
    auto Tmp = *this;
    Tmp >>= pos;
    return Tmp;
  }

  my_sub_group_mask(const my_sub_group_mask &rhs)
      : Bits(rhs.Bits), bits_num(rhs.bits_num) {}

  template <typename Group>
  friend detail::enable_if_t<
      std::is_same<std::decay_t<Group>, sub_group>::value, my_sub_group_mask>
  group_ballot(Group g, bool predicate);

  friend my_sub_group_mask operator&(const my_sub_group_mask &lhs,
                                     const my_sub_group_mask &rhs) {
    auto Res = lhs;
    Res &= rhs;
    return Res;
  }

  friend my_sub_group_mask operator|(const my_sub_group_mask &lhs,
                                     const my_sub_group_mask &rhs) {
    auto Res = lhs;
    Res |= rhs;
    return Res;
  }

  friend my_sub_group_mask operator^(const my_sub_group_mask &lhs,
                                     const my_sub_group_mask &rhs) {
    auto Res = lhs;
    Res ^= rhs;
    return Res;
  }

  my_sub_group_mask(uint32_t rhs, size_t bn) : Bits(rhs), bits_num(bn) {
    assert(bits_num <= max_bits);
  }
  inline uint32_t valuable_bits(size_t bn) const {
    assert(bn <= max_bits);
    uint32_t one = 1;
    if (bn == max_bits)
      return -one;
    return (one << bn) - one;
  }
  uint32_t Bits;
  // Number of valuable bits
  size_t bits_num;
};

constexpr int global_size = 128;
constexpr int local_size = 32;
int main() {
  queue q;
  buffer<uint32_t> MyBuf(1024);
  {
    q.submit([&](handler &cgh) {
       accessor my_acc{MyBuf, cgh};
       cgh.single_task([=]() {
         int my_idx = 0;
         uint32_t Magic = 0xb6db55b6;
         my_sub_group_mask my_mask(Magic, 32);
         {
           marray<unsigned char, 6> my_mr{1};
           my_mask.extract_bits(my_mr);
           for (int j = 0; j < 6; ++j)
             my_acc[my_idx++] = my_mr[j];
           my_acc[my_idx++] = 0x42;
         }
         {
           size_t cur_pos = 0;
           marray<unsigned char, 6> my_mr{1};
           for (auto &elem : my_mr) {
             if (cur_pos < my_mask.size()) {
               my_mask.extract_bits(elem, cur_pos);
               cur_pos += CHAR_BIT;
             } else {
               elem = 0;
             }
             my_acc[my_idx++] = elem;
           }
           my_acc[my_idx++] = 0x42;
           for (int j = 0; j < 6; ++j)
             my_acc[my_idx++] = my_mr[j];
           my_acc[my_idx++] = 0x42;
         }
         my_acc[my_idx++] = 0x42;
         my_acc[my_idx++] = 0x142;
       });
     }).wait();
  }
  host_accessor host_acc{MyBuf};
  for (int elem : host_acc) {
    std::cout << " " << std::hex << elem << " (" << std::bitset<8>(elem) << ")";
    if (elem == 0x20)
      std::cout << "\n       ";
    if (elem == 0x42)
      std::cout << std::endl;
    if (elem == 0x142)
      break;
  }
  std::cout << std::endl;
  return 1;
}
