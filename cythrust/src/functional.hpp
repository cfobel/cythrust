#ifndef ___CYTHRUST__FUNCTIONAL__HPP___
#define ___CYTHRUST__FUNCTIONAL__HPP___

#include <thrust/tuple.h>
#include <math.h>


namespace cythrust {
  template <typename T>
  struct first {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::get<0>(a);
    }
  };

  template <typename T>
  struct second {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::get<1>(a);
    }
  };

  template <typename T>
  struct third {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::get<2>(a);
    }
  };

  template <typename T>
  struct fourth {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::get<3>(a);
    }
  };

  template <typename T>
  struct fifth {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::get<4>(a);
    }
  };

  template <typename T>
  struct sixth {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::get<5>(a);
    }
  };

  template <typename T>
  struct seventh {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::get<6>(a);
    }
  };

  template <typename T>
  struct eighth {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::get<7>(a);
    }
  };

  template <typename T>
  struct ninth {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 a) {
      return thrust::get<8>(a);
    }
  };

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
  struct min2_tuple {
    typedef thrust::tuple<T, T> result_type;

    template <typename T1, typename T2>
    __host__ __device__
    result_type operator() (T1 const &a, T2 const &b) {
      result_type result;

      //     [     a0         a1           b0                  b1       ]
      if (thrust::get<1>(a) < thrust::get<0>(b)) {
        result = a;
      //     [     b0         b1           a0                  a1       ]
      } else if (thrust::get<1>(b) < thrust::get<0>(a)) {
        result = b;
      //     [     a0         b0           a1                  b1       ]
      //     [     a0         b0           b1                  a1       ]
      } else if (thrust::get<0>(a) < thrust::get<0>(b)) {
        result = thrust::make_tuple(thrust::get<0>(a), thrust::get<0>(b));
      //     [     b0         a0           a1                  b1       ]
      //     [     b0         a0           b1                  a1       ]
      } else {
        result = thrust::make_tuple(thrust::get<0>(b), thrust::get<0>(a));
      }
      return result;
    }
  };


  template <typename T>
  struct max2_tuple {
    typedef thrust::tuple<T, T> result_type;

    template <typename T1, typename T2>
    __host__ __device__
    result_type operator() (T1 const &a, T2 const &b) {
      //     [     a0         a1           b0                  b1       ]
      if (thrust::get<1>(a) < thrust::get<0>(b)) {
        return thrust::make_tuple(thrust::get<0>(b), thrust::get<1>(b));
      //     [     b0         b1           a0                  a1       ]
      } else if (thrust::get<1>(b) < thrust::get<0>(a)) {
        return thrust::make_tuple(thrust::get<0>(a), thrust::get<1>(a));
      //     [     a0         b0           a1                  b1       ]
      //     [     b0         a0           a1                  b1       ]
      } else if (thrust::get<1>(a) < thrust::get<1>(b)) {
        return thrust::make_tuple(thrust::get<1>(a), thrust::get<1>(b));
      //     [     a0         b0           b1                  a1       ]
      //     [     b0         a0           b1                  a1       ]
      } else {
        return thrust::make_tuple(thrust::get<1>(b), thrust::get<1>(a));
      }
    }
  };


  template <typename T>
  struct min2max2_tuple {
    typedef thrust::tuple<T, T, T, T> result_type;

    template <typename T1, typename T2>
    __host__ __device__
    result_type operator() (T1 const &a, T2 const &b) {
      /* # `minmax_tuple` #
       *
       * ## Arguments ##
       *
       *  - `a`: 4-tuple.
       *  - `b`: 4-tuple.
       *
       * ## Return ##
       *
       *  - 4-tuple containing:
       *   * Lowest value (minimum)
       *   * Second lowest value
       *   * Second highest value
       *   * Highest value (maximum)
       */
      T min0;
      T min1;
      T max0;
      T max1;

      //     [     a0         a1           b0                  b1       ]
      if (thrust::get<1>(a) < thrust::get<0>(b)) {
        min0 = thrust::get<0>(a);
        min1 = thrust::get<1>(a);
      //     [     b0         b1           a0                  a1       ]
      } else if (thrust::get<1>(b) < thrust::get<0>(a)) {
        min0 = thrust::get<0>(b);
        min1 = thrust::get<1>(b);
      //     [     a0         b0           a1                  b1       ]
      //     [     a0         b0           b1                  a1       ]
      } else if (thrust::get<0>(a) < thrust::get<0>(b)) {
        min0 = thrust::get<0>(a);
        min1 = thrust::get<0>(b);
      //     [     b0         a0           a1                  b1       ]
      //     [     b0         a0           b1                  a1       ]
      } else {
        min0 = thrust::get<0>(b);
        min1 = thrust::get<0>(a);
      }

      //     [     a0         a1           b0                  b1       ]
      if (thrust::get<3>(a) < thrust::get<2>(b)) {
        max0 = thrust::get<2>(b);
        max1 = thrust::get<3>(b);
      //     [     b0         b1           a0                  a1       ]
      } else if (thrust::get<3>(b) < thrust::get<2>(a)) {
        max0 = thrust::get<2>(a);
        max1 = thrust::get<3>(a);
      //     [     a0         b0           a1                  b1       ]
      //     [     b0         a0           a1                  b1       ]
      } else if (thrust::get<3>(a) < thrust::get<3>(b)) {
        max0 = thrust::get<3>(a);
        max1 = thrust::get<3>(b);
      //     [     a0         b0           b1                  a1       ]
      //     [     b0         a0           b1                  a1       ]
      } else {
        max0 = thrust::get<3>(b);
        max1 = thrust::get<3>(a);
      }
      return thrust::make_tuple(min0, min1, max0, max1);
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
          (thrust::get<1>(a) < thrust::get<1>(b)) ? thrust::get<1>(b)
                                                  : thrust::get<1>(a));
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


  template <typename T>
  struct acos_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return acosf(v);
    }
  };


  template <typename T>
  struct asin_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return asinf(v);
    }
  };


  template <typename T>
  struct atan_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return atanf(v);
    }
  };


  template <typename T>
  struct atan2_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return atan2f(v);
    }
  };


  template <typename T>
  struct cos_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return cosf(v);
    }
  };


  template <typename T>
  struct sin_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return sinf(v);
    }
  };


  template <typename T>
  struct tan_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return tanf(v);
    }
  };


  template <typename T>
  struct cosh_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return coshf(v);
    }
  };


  template <typename T>
  struct sinh_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return sinhf(v);
    }
  };


  template <typename T>
  struct tanh_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) {
      return tanhf(v);
    }
  };


  template <typename T>
  struct ceil_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) { return ceilf(v); }
  };


  template <typename T>
  struct floor_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) { return floorf(v); }
  };


  template <typename T>
  struct trunc_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) { return (v >= 0) ? floorf(v) : -floorf(-v); }
  };


  template <typename T>
  struct inv_ {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) { return 1. / v; }
  };


  template <typename T>
  struct sign {
    typedef T result_type;

    template <typename T1>
    __host__ __device__
    result_type operator() (T1 v) { return (v >= 0) ? 1 : -1; }
  };


  template <typename T>
  struct switch_ {
    typedef T result_type;

    template <typename T1, typename T2, typename T3>
    __host__ __device__
    result_type operator() (T1 a, T2 b, T3 c) {
      return (a) ? b : c;
    }
  };
}

#endif  // #ifndef ___CYTHRUST__FUNCTIONAL__HPP___
