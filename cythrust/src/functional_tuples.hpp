#ifndef ___CYTHRUST__FUNCTIONAL_TUPLES__HPP___
#define ___CYTHRUST__FUNCTIONAL_TUPLES__HPP___

#include <thrust/tuple.h>
#include <math.h>


namespace cythrust {
  template <typename F1, typename F2>
  struct reduce2pair {
    typedef typename F1::result_type R1;
    typedef typename F2::result_type R2;
    typedef thrust::tuple<R1, R2> result_type;

    template <typename Pair>
    __host__ __device__
    result_type operator() (Pair p) {
      F1 f1;
      F2 f2;
      return thrust::make_tuple(
        f1(thrust::get<0>(p.first), thrust::get<0>(p.second)),
        f2(thrust::get<1>(p.first), thrust::get<1>(p.second)));
    }
  };

  template <typename F0, typename F1>
  struct reduce2 {
    typedef typename F0::result_type R0;
    typedef typename F1::result_type R1;
    typedef thrust::tuple<R0, R1> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 a, Tuple2 b) {
      F0 f0;
      F1 f1;

      return thrust::make_tuple(
        f0(thrust::get<0>(a), thrust::get<0>(b)),
        f1(thrust::get<1>(a), thrust::get<1>(b)));
    }
  };

  template <typename F0, typename F1, typename F2>
  struct reduce3 {
    typedef typename F0::result_type R0;
    typedef typename F1::result_type R1;
    typedef typename F2::result_type R2;
    typedef thrust::tuple<R0, R1, R2> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 a, Tuple2 b) {
      F0 f0;
      F1 f1;
      F2 f2;

      return thrust::make_tuple(
        f0(thrust::get<0>(a), thrust::get<0>(b)),
        f1(thrust::get<1>(a), thrust::get<1>(b)),
        f2(thrust::get<2>(a), thrust::get<2>(b)));
    }
  };

  template <typename F0, typename F1, typename F2, typename F3>
  struct reduce4 {
    typedef typename F0::result_type R0;
    typedef typename F1::result_type R1;
    typedef typename F2::result_type R2;
    typedef typename F3::result_type R3;
    typedef thrust::tuple<R0, R1, R2, R3> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 a, Tuple2 b) {
      F0 f0;
      F1 f1;
      F2 f2;
      F3 f3;

      return thrust::make_tuple(
        f0(thrust::get<0>(a), thrust::get<0>(b)),
        f1(thrust::get<1>(a), thrust::get<1>(b)),
        f2(thrust::get<2>(a), thrust::get<2>(b)),
        f3(thrust::get<3>(a), thrust::get<3>(b)));
    }
  };

  template <typename F0, typename F1, typename F2, typename F3, typename F4>
  struct reduce5 {
    typedef typename F0::result_type R0;
    typedef typename F1::result_type R1;
    typedef typename F2::result_type R2;
    typedef typename F3::result_type R3;
    typedef typename F4::result_type R4;
    typedef thrust::tuple<R0, R1, R2, R3, R4> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 a, Tuple2 b) {
      F0 f0;
      F1 f1;
      F2 f2;
      F3 f3;
      F4 f4;

      return thrust::make_tuple(
        f0(thrust::get<0>(a), thrust::get<0>(b)),
        f1(thrust::get<1>(a), thrust::get<1>(b)),
        f2(thrust::get<2>(a), thrust::get<2>(b)),
        f3(thrust::get<3>(a), thrust::get<3>(b)),
        f4(thrust::get<4>(a), thrust::get<4>(b)));
    }
  };

  template <typename F0, typename F1, typename F2, typename F3, typename F4, typename F5>
  struct reduce6 {
    typedef typename F0::result_type R0;
    typedef typename F1::result_type R1;
    typedef typename F2::result_type R2;
    typedef typename F3::result_type R3;
    typedef typename F4::result_type R4;
    typedef typename F5::result_type R5;
    typedef thrust::tuple<R0, R1, R2, R3, R4, R5> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 a, Tuple2 b) {
      F0 f0;
      F1 f1;
      F2 f2;
      F3 f3;
      F4 f4;
      F5 f5;

      return thrust::make_tuple(
        f0(thrust::get<0>(a), thrust::get<0>(b)),
        f1(thrust::get<1>(a), thrust::get<1>(b)),
        f2(thrust::get<2>(a), thrust::get<2>(b)),
        f3(thrust::get<3>(a), thrust::get<3>(b)),
        f4(thrust::get<4>(a), thrust::get<4>(b)),
        f5(thrust::get<5>(a), thrust::get<5>(b)));
    }
  };

  template <typename F0, typename F1, typename F2, typename F3, typename F4, typename F5, typename F6>
  struct reduce7 {
    typedef typename F0::result_type R0;
    typedef typename F1::result_type R1;
    typedef typename F2::result_type R2;
    typedef typename F3::result_type R3;
    typedef typename F4::result_type R4;
    typedef typename F5::result_type R5;
    typedef typename F6::result_type R6;
    typedef thrust::tuple<R0, R1, R2, R3, R4, R5, R6> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 a, Tuple2 b) {
      F0 f0;
      F1 f1;
      F2 f2;
      F3 f3;
      F4 f4;
      F5 f5;
      F6 f6;

      return thrust::make_tuple(
        f0(thrust::get<0>(a), thrust::get<0>(b)),
        f1(thrust::get<1>(a), thrust::get<1>(b)),
        f2(thrust::get<2>(a), thrust::get<2>(b)),
        f3(thrust::get<3>(a), thrust::get<3>(b)),
        f4(thrust::get<4>(a), thrust::get<4>(b)),
        f5(thrust::get<5>(a), thrust::get<5>(b)),
        f6(thrust::get<6>(a), thrust::get<6>(b)));
    }
  };

  template <typename F0, typename F1, typename F2, typename F3, typename F4, typename F5, typename F6, typename F7>
  struct reduce8 {
    typedef typename F0::result_type R0;
    typedef typename F1::result_type R1;
    typedef typename F2::result_type R2;
    typedef typename F3::result_type R3;
    typedef typename F4::result_type R4;
    typedef typename F5::result_type R5;
    typedef typename F6::result_type R6;
    typedef typename F7::result_type R7;
    typedef thrust::tuple<R0, R1, R2, R3, R4, R5, R6, R7> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 a, Tuple2 b) {
      F0 f0;
      F1 f1;
      F2 f2;
      F3 f3;
      F4 f4;
      F5 f5;
      F6 f6;
      F7 f7;

      return thrust::make_tuple(
        f0(thrust::get<0>(a), thrust::get<0>(b)),
        f1(thrust::get<1>(a), thrust::get<1>(b)),
        f2(thrust::get<2>(a), thrust::get<2>(b)),
        f3(thrust::get<3>(a), thrust::get<3>(b)),
        f4(thrust::get<4>(a), thrust::get<4>(b)),
        f5(thrust::get<5>(a), thrust::get<5>(b)),
        f6(thrust::get<6>(a), thrust::get<6>(b)),
        f7(thrust::get<7>(a), thrust::get<7>(b)));
    }
  };

  template <typename F0, typename F1, typename F2, typename F3, typename F4, typename F5, typename F6, typename F7, typename F8>
  struct reduce9 {
    typedef typename F0::result_type R0;
    typedef typename F1::result_type R1;
    typedef typename F2::result_type R2;
    typedef typename F3::result_type R3;
    typedef typename F4::result_type R4;
    typedef typename F5::result_type R5;
    typedef typename F6::result_type R6;
    typedef typename F7::result_type R7;
    typedef typename F8::result_type R8;
    typedef thrust::tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8> result_type;

    template <typename Tuple1, typename Tuple2>
    __host__ __device__
    result_type operator() (Tuple1 a, Tuple2 b) {
      F0 f0;
      F1 f1;
      F2 f2;
      F3 f3;
      F4 f4;
      F5 f5;
      F6 f6;
      F7 f7;
      F8 f8;

      return thrust::make_tuple(
        f0(thrust::get<0>(a), thrust::get<0>(b)),
        f1(thrust::get<1>(a), thrust::get<1>(b)),
        f2(thrust::get<2>(a), thrust::get<2>(b)),
        f3(thrust::get<3>(a), thrust::get<3>(b)),
        f4(thrust::get<4>(a), thrust::get<4>(b)),
        f5(thrust::get<5>(a), thrust::get<5>(b)),
        f6(thrust::get<6>(a), thrust::get<6>(b)),
        f7(thrust::get<7>(a), thrust::get<7>(b)),
        f8(thrust::get<8>(a), thrust::get<8>(b)));
    }
  };
}

#endif  // #ifndef ___CYTHRUST__FUNCTIONAL_TUPLES__HPP___
