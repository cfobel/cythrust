#include <thrust/tuple.h>


namespace cythrust {

  template <typename T>
  struct square {
    typedef T result_type;

    T operator() (T a) { return a * a; }
  };


  template <typename T>
  struct reduce_plus4 {
    typedef thrust::tuple<T, T, T, T> result_type;

    template <typename Tuple1, typename Tuple2>
    result_type operator() (Tuple1 const &a, Tuple2 const &b) {
      return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                thrust::get<1>(a) + thrust::get<1>(b),
                                thrust::get<2>(a) + thrust::get<2>(b),
                                thrust::get<3>(a) + thrust::get<3>(b));
    }
  };


  template <typename T>
  struct plus4 {
    typedef T result_type;

    T operator() (T a, T b, T c, T d) { return a + b + c + d; }
  };


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
