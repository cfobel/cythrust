#ifndef ___CYTHRUST_REDUCE__HPP___
#define ___CYTHRUST_REDUCE__HPP___

#include <thrust/reduce.h>

namespace cythrust {

template<typename InputIterator, typename T, typename BinaryFunction>
inline T reduce_n(InputIterator first, size_t n, T init_value,
           BinaryFunction binary_op) {
  return thrust::reduce(first, first + n, init_value, binary_op);
}

}

#endif  // #ifndef ___CYTHRUST_REDUCE__HPP___
