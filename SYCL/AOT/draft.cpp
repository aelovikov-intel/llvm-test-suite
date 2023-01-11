// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Od
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;
template <typename Type, std::size_t NumElements> struct my_marray {
  using DataT = Type;

  using value_type = Type;
  using reference = Type &;
  using const_reference = const Type &;
  using iterator = Type *;
  using const_iterator = const Type *;

  value_type MData[NumElements];

  template <class...> struct conjunction : std::true_type {};
  template <class B1, class... tail>
  struct conjunction<B1, tail...>
      : std::conditional<bool(B1::value), conjunction<tail...>, B1>::type {};

  // TypeChecker is needed for (const ArgTN &... Args) ctor to validate Args.
  template <typename T, typename DataT_>
  struct TypeChecker : std::is_convertible<T, DataT_> {};

  // Shortcuts for Args validation in (const ArgTN &... Args) ctor.
  template <typename... ArgTN>
  using EnableIfSuitableTypes = typename std::enable_if<
      conjunction<TypeChecker<ArgTN, DataT>...>::value>::type;

  constexpr void initialize_data(const Type &Arg) {
    for (size_t i = 0; i < NumElements; ++i) {
      MData[i] = Arg;
    }
  }

  constexpr my_marray() : MData{} {}

  explicit constexpr my_marray(const Type &Arg) : MData{Arg} {
    initialize_data(Arg);
  }

  template <
      typename... ArgTN, typename = EnableIfSuitableTypes<ArgTN...>,
      typename = typename std::enable_if<sizeof...(ArgTN) == NumElements>::type>
  constexpr my_marray(const ArgTN &...Args)
      : MData{static_cast<Type>(Args)...} {}

  constexpr my_marray(const my_marray<Type, NumElements> &Rhs) = default;

  constexpr my_marray(my_marray<Type, NumElements> &&Rhs) = default;

  // Available only when: NumElements == 1
  template <std::size_t Size = NumElements,
            typename = typename std::enable_if<Size == 1>>
  operator Type() const {
    return MData[0];
  }

  static constexpr std::size_t size() noexcept { return NumElements; }

  // subscript operator
  reference operator[](std::size_t index) { return MData[index]; }

  const_reference operator[](std::size_t index) const { return MData[index]; }

  my_marray &operator=(const my_marray<Type, NumElements> &Rhs) = default;

  // broadcasting operator
  my_marray &operator=(const Type &Rhs) {
    for (std::size_t I = 0; I < NumElements; ++I) {
      MData[I] = Rhs;
    }
    return *this;
  }

  // iterator functions
  iterator begin() { return MData; }

  const_iterator begin() const { return MData; }

  iterator end() { return MData + NumElements; }

  const_iterator end() const { return MData + NumElements; }

#ifdef __SYCL_BINOP
#error "Undefine __SYCL_BINOP macro"
#endif

#ifdef __SYCL_BINOP_INTEGRAL
#error "Undefine __SYCL_BINOP_INTEGRAL macro"
#endif

#define __SYCL_BINOP(BINOP, OPASSIGN)                                          \
  friend my_marray operator BINOP(const my_marray &Lhs,                        \
                                  const my_marray &Rhs) {                      \
    my_marray Ret;                                                             \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] BINOP Rhs[I];                                            \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  friend typename std::enable_if<                                              \
      std::is_convertible<DataT, T>::value &&                                  \
          (std::is_fundamental<T>::value ||                                    \
           std::is_same<typename std::remove_const<T>::type, half>::value),    \
      my_marray>::type                                                         \
  operator BINOP(const my_marray &Lhs, const T &Rhs) {                         \
    return Lhs BINOP my_marray(static_cast<DataT>(Rhs));                       \
  }                                                                            \
  friend my_marray &operator OPASSIGN(my_marray &Lhs, const my_marray &Rhs) {  \
    Lhs = Lhs BINOP Rhs;                                                       \
    return Lhs;                                                                \
  }                                                                            \
  template <std::size_t Num = NumElements>                                     \
  friend typename std::enable_if<Num != 1, my_marray &>::type                  \
  operator OPASSIGN(my_marray &Lhs, const DataT &Rhs) {                        \
    Lhs = Lhs BINOP my_marray(Rhs);                                            \
    return Lhs;                                                                \
  }

#define __SYCL_BINOP_INTEGRAL(BINOP, OPASSIGN)                                 \
  template <typename T = DataT,                                                \
            typename = std::enable_if<std::is_integral<T>::value, my_marray>>  \
  friend my_marray operator BINOP(const my_marray &Lhs,                        \
                                  const my_marray &Rhs) {                      \
    my_marray Ret;                                                             \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] BINOP Rhs[I];                                            \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T, typename BaseT = DataT>                                \
  friend typename std::enable_if<std::is_convertible<T, DataT>::value &&       \
                                     std::is_integral<T>::value &&             \
                                     std::is_integral<BaseT>::value,           \
                                 my_marray>::type                              \
  operator BINOP(const my_marray &Lhs, const T &Rhs) {                         \
    return Lhs BINOP my_marray(static_cast<DataT>(Rhs));                       \
  }                                                                            \
  template <typename T = DataT,                                                \
            typename = std::enable_if<std::is_integral<T>::value, my_marray>>  \
  friend my_marray &operator OPASSIGN(my_marray &Lhs, const my_marray &Rhs) {  \
    Lhs = Lhs BINOP Rhs;                                                       \
    return Lhs;                                                                \
  }                                                                            \
  template <std::size_t Num = NumElements, typename T = DataT>                 \
  friend typename std::enable_if<Num != 1 && std::is_integral<T>::value,       \
                                 my_marray &>::type                            \
  operator OPASSIGN(my_marray &Lhs, const DataT &Rhs) {                        \
    Lhs = Lhs BINOP my_marray(Rhs);                                            \
    return Lhs;                                                                \
  }

  __SYCL_BINOP(+, +=)
  __SYCL_BINOP(-, -=)
  __SYCL_BINOP(*, *=)
  __SYCL_BINOP(/, /=)

  __SYCL_BINOP_INTEGRAL(%, %=)
  __SYCL_BINOP_INTEGRAL(|, |=)
  __SYCL_BINOP_INTEGRAL(&, &=)
  __SYCL_BINOP_INTEGRAL(^, ^=)
  __SYCL_BINOP_INTEGRAL(>>, >>=)
  __SYCL_BINOP_INTEGRAL(<<, <<=)
#undef __SYCL_BINOP
#undef __SYCL_BINOP_INTEGRAL

#ifdef __SYCL_RELLOGOP
#error "Undefine __SYCL_RELLOGOP macro"
#endif

#ifdef __SYCL_RELLOGOP_INTEGRAL
#error "Undefine __SYCL_RELLOGOP_INTEGRAL macro"
#endif

#define __SYCL_RELLOGOP(RELLOGOP)                                              \
  friend my_marray<bool, NumElements> operator RELLOGOP(                       \
      const my_marray &Lhs, const my_marray &Rhs) {                            \
    my_marray<bool, NumElements> Ret;                                          \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] RELLOGOP Rhs[I];                                         \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  friend typename std::enable_if<std::is_convertible<T, DataT>::value &&       \
                                     (std::is_fundamental<T>::value ||         \
                                      std::is_same<T, half>::value),           \
                                 my_marray<bool, NumElements>>::type           \
  operator RELLOGOP(const my_marray &Lhs, const T &Rhs) {                      \
    return Lhs RELLOGOP my_marray(static_cast<const DataT &>(Rhs));            \
  }

#define __SYCL_RELLOGOP_INTEGRAL(RELLOGOP)                                     \
  template <typename T = DataT>                                                \
  friend typename std::enable_if<std::is_integral<T>::value,                   \
                                 my_marray<bool, NumElements>>::type           \
  operator RELLOGOP(const my_marray &Lhs, const my_marray &Rhs) {              \
    my_marray<bool, NumElements> Ret;                                          \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] RELLOGOP Rhs[I];                                         \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T, typename BaseT = DataT>                                \
  friend typename std::enable_if<std::is_convertible<T, DataT>::value &&       \
                                     std::is_integral<T>::value &&             \
                                     std::is_integral<BaseT>::value,           \
                                 my_marray<bool, NumElements>>::type           \
  operator RELLOGOP(const my_marray &Lhs, const T &Rhs) {                      \
    return Lhs RELLOGOP my_marray(static_cast<const DataT &>(Rhs));            \
  }

  __SYCL_RELLOGOP(==)
  __SYCL_RELLOGOP(!=)
  __SYCL_RELLOGOP(>)
  __SYCL_RELLOGOP(<)
  __SYCL_RELLOGOP(>=)
  __SYCL_RELLOGOP(<=)

  __SYCL_RELLOGOP_INTEGRAL(&&)
  __SYCL_RELLOGOP_INTEGRAL(||)

#undef __SYCL_RELLOGOP
#undef __SYCL_RELLOGOP_INTEGRAL

#ifdef __SYCL_UOP
#error "Undefine __SYCL_UOP macro"
#endif

#define __SYCL_UOP(UOP, OPASSIGN)                                              \
  friend my_marray &operator UOP(my_marray &Lhs) {                             \
    Lhs OPASSIGN 1;                                                            \
    return Lhs;                                                                \
  }                                                                            \
  friend my_marray operator UOP(my_marray &Lhs, int) {                         \
    my_marray Ret(Lhs);                                                        \
    Lhs OPASSIGN 1;                                                            \
    return Ret;                                                                \
  }

  __SYCL_UOP(++, +=)
  __SYCL_UOP(--, -=)
#undef __SYCL_UOP

  // Available only when: dataT != cl_float && dataT != cl_double
  // && dataT != cl_half
  template <typename T = DataT>
  friend typename std::enable_if<std::is_integral<T>::value, my_marray>::type
  operator~(const my_marray &Lhs) {
    my_marray Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = ~Lhs[I];
    }
    return Ret;
  }

  friend my_marray<bool, NumElements> operator!(const my_marray &Lhs) {
    my_marray<bool, NumElements> Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = !Lhs[I];
    }
    return Ret;
  }

  friend my_marray operator+(const my_marray &Lhs) {
    my_marray Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = +Lhs[I];
    }
    return Ret;
  }

  friend my_marray operator-(const my_marray &Lhs) {
    my_marray Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = -Lhs[I];
    }
    return Ret;
  }
};

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
  my_marray id |     0     |     1     |     2     |     3     |...
  bit id    |7   ..    0|15   ..   8|23   ..  16|31  ..   24|...
  */
  template <typename Type, size_t Size,
            typename = sycl::detail::enable_if_t<std::is_integral<Type>::value>>
  void insert_bits(const my_marray<Type, Size> &bits, id<1> pos = 0) {
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
  void extract_bits(my_marray<Type, Size> &bits, id<1> pos = 0) const {
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
  void extract_bits_no_range_for(my_marray<Type, Size> &bits,
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
  void extract_bits_no_range_for_no_ref(my_marray<Type, Size> &bits,
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
  buffer<uint64_t> MyBuf(1024);
  {
    q.submit([&](handler &cgh) {
       accessor my_acc{MyBuf, cgh};
       cgh.single_task([=]() {
         int my_idx = 0;
         uint32_t Magic = 0xb6db55b6;
         my_sub_group_mask my_mask(Magic, 32);
         {
           my_marray<unsigned char, 6> my_mr{1};
           my_mask.extract_bits(my_mr);
           for (int j = 0; j < 6; ++j)
             my_acc[my_idx++] = my_mr[j];
           my_acc[my_idx++] = 0x42;
         }
         {
           size_t cur_pos = 0;
           my_marray<unsigned char, 6> my_mr{1};
           my_acc[my_idx++] = reinterpret_cast<uintptr_t>(&my_mr.MData[0]);
           my_acc[my_idx++] = 0x42;
           for (auto &elem : my_mr) {
             if (cur_pos < my_mask.size()) {
               my_mask.extract_bits(elem, cur_pos);
               cur_pos += CHAR_BIT;
             } else {
               elem = 0;
             }
             my_acc[my_idx++] = reinterpret_cast<uintptr_t>(&elem);
             my_acc[my_idx++] = elem;
           }
           my_acc[my_idx++] = 0x42;
           for (int j = 0; j < 6; ++j) {
             my_acc[my_idx++] = reinterpret_cast<uintptr_t>(&my_mr[j]);
             my_acc[my_idx++] = my_mr[j];
           }
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

#if 0
Bad
 b6 (10110110) 55 (01010101) db (11011011) b6 (10110110) 0 (00000000) 0 (00000000) 42 (01000010)
 b6 (10110110) 55 (01010101) db (11011011) b6 (10110110) 0 (00000000) 0 (00000000) 42 (01000010)
 b6 (10110110) 1 (00000001) 1 (00000001) 1 (00000001) 1 (00000001) 1 (00000001) 42 (01000010)
 42 (01000010)
 142 (01000010)


Good
 b6 (10110110) 55 (01010101) db (11011011) b6 (10110110) 0 (00000000) 0 (00000000) 42 (01000010)
 b6 (10110110) 55 (01010101) db (11011011) b6 (10110110) 0 (00000000) 0 (00000000) 42 (01000010)
 b6 (10110110) 55 (01010101) db (11011011) b6 (10110110) 0 (00000000) 0 (00000000) 42 (01000010)
 42 (01000010)
 142 (01000010)
#endif
