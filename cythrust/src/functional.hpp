#ifndef ___CYTHRUST__FUNCTIONAL__HPP___
#define ___CYTHRUST__FUNCTIONAL__HPP___

#include <thrust/tuple.h>


namespace cythrust {

  template <typename T>
  struct power {
    typedef T result_type;

    template <typename T1, typename T2>
    __host__ __device__
    result_type operator() (T1 a, T2 b) {
      return pow(a, b);
    }
  };


  template <typename T>
  struct reverse_divides {
    typedef T result_type;

    template <typename T1, typename T2>
    __host__ __device__
    result_type operator() (T1 a, T2 b) {
      return b / a;
    }
  };


  template <typename T>
  struct duplicate {
    typedef thrust::tuple<T, T> result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::make_tuple(a, a);
    }
  };


  template <typename T>
  struct minmax {
    typedef thrust::tuple<T, T> result_type;

    template <typename T1, typename T2>
    __host__ __device__
    result_type operator() (T1 a, T2 b) {
      return thrust::make_tuple((a < b) ? a : b, (a < b) ? b : a);
    }
  };


  template <typename T>
  struct minmax_tuple {
    typedef thrust::tuple<T, T> result_type;

    template <typename T1, typename T2>
    __host__ __device__
    result_type operator() (T1 const &a, T2 const &b) {
      return thrust::make_tuple(
          (thrust::get<0>(a) < thrust::get<0>(b)) ? thrust::get<0>(a)
                                                  : thrust::get<0>(b),
          (thrust::get<0>(a) < thrust::get<0>(b)) ? thrust::get<0>(b)
                                                  : thrust::get<0>(a));
    }
  };


  template <typename T>
  struct absolute {
    typedef T result_type;

    __host__ __device__
    T operator() (T a) { return (a < 0) ? -a : a; }
  };


  template <typename T>
  struct square {
    typedef T result_type;

    __host__ __device__
    T operator() (T a) { return a * a; }
  };


  template <typename T>
  struct square_root {
    typedef T result_type;

    __host__ __device__
    T operator() (T a) { return sqrtf(a); }
  };


  template <typename T>
  struct reduce_plus4 {
    typedef thrust::tuple<T, T, T, T> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 const &a, Tuple2 const &b) {
      return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                thrust::get<1>(a) + thrust::get<1>(b),
                                thrust::get<2>(a) + thrust::get<2>(b),
                                thrust::get<3>(a) + thrust::get<3>(b));
    }
  };


  template <typename T>
  struct reduce_plus4_with_dummy {
    /* Add dummy argument to prevent CUDA error:
     *
     *     ../thrust/system/cuda/detail/bulk/algorithm/reduce_by_key.hpp(58): error: ambiguous "?" operation
     */
    typedef thrust::tuple<T, T, T, T, T> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 const &a, Tuple2 const &b) {
      return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                thrust::get<1>(a) + thrust::get<1>(b),
                                thrust::get<2>(a) + thrust::get<2>(b),
                                thrust::get<3>(a) + thrust::get<3>(b),
                                0);
    }
  };


  template <typename T>
  struct plus4 {
    typedef T result_type;

  __host__ __device__
    T operator() (T a, T b, T c, T d) { return a + b + c + d; }
  };


  template <typename T>
  struct plus5 {
    typedef T result_type;

  __host__ __device__
    T operator() (T a, T b, T c, T d, T e) {
      return a + b + c + d + e;
    }
  };


  template <typename T>
  struct plus_tuple2 {
    typedef T result_type;

    template <typename Tuple>
    __host__ __device__
    T operator() (Tuple const &t) {
      return (thrust::get<0>(t) + thrust::get<1>(t));
    }
  };


  template <typename T>
  struct plus_tuple5 {
    typedef T result_type;

    template <typename Tuple>
    __host__ __device__
    T operator() (Tuple const &t) {
      return (thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<2>(t) +
              thrust::get<3>(t) + thrust::get<4>(t));
    }
  };


  template <typename T>
  struct non_positive {
    typedef T result_type;

    template <typename Tuple>
    __host__ __device__
    T operator() (Tuple const &t) { return t <= 0; }
  };


  template <typename T>
  struct negative {
    typedef T result_type;

    template <typename Tuple>
    __host__ __device__
    T operator() (Tuple const &t) { return t < 0; }
  };


  template <typename T>
  struct non_negative {
    typedef T result_type;

    template <typename Tuple>
    __host__ __device__
    T operator() (Tuple const &t) { return t >= 0; }
  };


  template <typename T>
  struct positive {
    typedef T result_type;

    template <typename Tuple>
    __host__ __device__
    T operator() (Tuple const &t) { return t > 0; }
  };


  template <typename T>
  struct less_than_constant {
    typedef T result_type;

    result_type value;

    less_than_constant(T value) : value(value) {}

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) { return v < value; }
  };
}

#endif  // #ifndef ___CYTHRUST__FUNCTIONAL__HPP___
