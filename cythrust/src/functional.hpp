#include <thrust/tuple.h>


namespace cythrust {

  template <typename T>
  struct plus5 {
    typedef T result_type;

    T operator() (T a, T b, T c, T d, T e) {
      return a + b + c + d + e;
    }
  };


  template <typename T>
  struct plus_tuple2 {
    typedef T result_type;

    template <typename Tuple>
    T operator() (Tuple const &t) {
      return (thrust::get<0>(t) + thrust::get<1>(t));
    }
  };


  template <typename T>
  struct plus_tuple5 {
    typedef T result_type;

    template <typename Tuple>
    T operator() (Tuple const &t) {
      return (thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<2>(t) +
              thrust::get<3>(t) + thrust::get<4>(t));
    }
  };
}
