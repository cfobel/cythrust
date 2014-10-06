#ifndef ___REPEATED_RANGE__H___
#define ___REPEATED_RANGE__H___

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

// This example illustrates how to make repeated access to a range of values
// examples:
//
//   RepeatedRange([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   RepeatedRange([0, 1, 2, 3], 2) -> [0, 0, 1, 1, 2, 2, 3, 3]
//   RepeatedRange([0, 1, 2, 3], 3) -> [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
//   ...

namespace thrust {

  template <typename Iterator, typename T>
  class repeated_range {
      public:

      struct repeat_functor : public thrust::unary_function<T,T> {
          T repeats;

          __host__ __device__
          repeat_functor(T repeats) : repeats(repeats) {}

          __host__ __device__
          T operator()(const T& i) const {
              return i / repeats;
          }
      };

      typedef typename thrust::counting_iterator<T> CountingIterator;
      typedef typename thrust::transform_iterator<repeat_functor,
                                                  CountingIterator>
          TransformIterator;
      typedef typename thrust::permutation_iterator<Iterator, TransformIterator>
          PermutationIterator;

      // Type of the `RepeatedRange` iterator
      typedef PermutationIterator iterator;

      // Construct `RepeatedRange` for the range `[first,last)`
      __host__ __device__
      repeated_range(Iterator first, T repeats)
          : first(first), repeats(repeats) {}

      __host__ __device__
      iterator begin(void) const {
          return PermutationIterator(first,
                                     TransformIterator(CountingIterator(0),
                                                       repeat_functor(repeats)));
      }

      protected:

      Iterator first;
      Iterator last;
      T repeats;
  };

}  // namespace thrust

#endif  // #ifndef ___REPEATED_RANGE__H___
