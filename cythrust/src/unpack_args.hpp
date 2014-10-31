#ifndef ___UNPACK_ARGS__HPP___
#define ___UNPACK_ARGS__HPP___


template <typename Functor>
struct unpack_binary_args {
  typedef typename Functor::result_type result_type;
  Functor functor_;

  __host__ __device__
  unpack_binary_args(Functor functor) : functor_(functor) {}

  template <typename Tuple>
  __host__ __device__
  result_type operator() (Tuple args) {
    return functor_(thrust::get<0>(args), thrust::get<1>(args));
  }
};


template <typename Functor>
struct unpack_ternary_args {
  typedef typename Functor::result_type result_type;
  Functor functor_;

  __host__ __device__
  unpack_ternary_args(Functor functor) : functor_(functor) {}

  template <typename Tuple>
  __host__ __device__
  result_type operator() (Tuple args) {
    return functor_(thrust::get<0>(args), thrust::get<1>(args),
                    thrust::get<2>(args));
  }
};


template <typename Functor>
struct unpack_quaternary_args {
  typedef typename Functor::result_type result_type;
  Functor functor_;

  __host__ __device__
  unpack_quaternary_args(Functor functor) : functor_(functor) {}

  template <typename Tuple>
  __host__ __device__
  result_type operator() (Tuple args) {
    return functor_(thrust::get<0>(args), thrust::get<1>(args),
                    thrust::get<2>(args), thrust::get<3>(args));
  }
};


template <typename Functor>
struct unpack_quinary_args {
  typedef typename Functor::result_type result_type;
  Functor functor_;

  __host__ __device__
  unpack_quinary_args(Functor functor) : functor_(functor) {}

  template <typename Tuple>
  __host__ __device__
  result_type operator() (Tuple args) {
    return functor_(thrust::get<0>(args), thrust::get<1>(args),
                    thrust::get<2>(args), thrust::get<3>(args),
                    thrust::get<4>(args));
  }
};


#endif  // #ifndef ___UNPACK_ARGS__HPP___
