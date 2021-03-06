from cythrust.thrust.tuple cimport (tuple2, tuple3, tuple4, tuple5, tuple6,
                                    tuple7, tuple8, tuple9)


cdef extern from "<thrust/functional.h>" namespace "thrust" nogil:
    cdef cppclass unary_function[Argument, Result]:
        pass
    cdef cppclass binary_function[Argument1, Argument2, Result]:
        pass

    # ## Binary ##
    cdef cppclass plus[T]:
        pass
    cdef cppclass minus[T]:
        pass
    cdef cppclass multiplies[T]:
        pass
    cdef cppclass divides[T]:
        pass
    cdef cppclass modulus[T]:
        pass
    cdef cppclass equal_to[T]:
        pass
    cdef cppclass not_equal_to[T]:
        pass
    cdef cppclass greater[T]:
        pass
    cdef cppclass less[T]:
        pass
    cdef cppclass greater_equal[T]:
        pass
    cdef cppclass less_equal[T]:
        pass
    cdef cppclass logical_and[T]:
        pass
    cdef cppclass logical_or[T]:
        pass
    cdef cppclass bit_and[T]:
        pass
    cdef cppclass bit_or[T]:
        pass
    cdef cppclass bit_xor[T]:
        pass
    cdef cppclass maximum[T]:
        pass
    cdef cppclass minimum[T]:
        pass

    # ## Unary ##
    cdef cppclass negate[T]:
        pass
    cdef cppclass logical_not[T]:
        pass
    cdef cppclass identity[T]:
        pass

    # ## Other ##
    cdef cppclass unary_negate[Predicate]:
        pass
    cdef cppclass binary_negate[Predicate]:
        pass
    cdef cppclass project1st[T1, T2]:
        pass
    cdef cppclass project2nd[T1, T2]:
        pass


cdef extern from "src/functional.hpp" namespace "cythrust" nogil:
    cdef cppclass less_than_constant[T]:
        less_than_constant(T)

    cdef cppclass non_positive[T]:  # <= 0
        pass

    cdef cppclass non_negative[T]:  # >= 0
        pass

    cdef cppclass negative[T]:  # < 0
        pass

    cdef cppclass positive[T]:  # > 0
        pass

    cdef cppclass duplicate[T]:
        pass

    cdef cppclass minmax_tuple[T]:
        pass

    cdef cppclass min2_tuple[T]:
        pass

    cdef cppclass max2_tuple[T]:
        pass

    cdef cppclass min2max2_tuple[T]:
        pass

    cdef cppclass minmax[T]:
        pass

    cdef cppclass absolute[T]:
        pass

    cdef cppclass square[T]:
        pass

    cdef cppclass square_root[T]:
        pass

    cdef cppclass reduce_plus4_with_dummy[T]:
        pass

    cdef cppclass reduce_plus4[T]:
        pass

    cdef cppclass plus4[T]:
        pass

    cdef cppclass plus5[T]:
        pass

    cdef cppclass plus_tuple5[T]:
        pass

    cdef cppclass first[T]:
        T operator()[T1](T1 a)

    cdef cppclass second[T]:
        T operator()[T1](T1 a)

    cdef cppclass third[T]:
        T operator()[T1](T1 a)

    cdef cppclass fourth[T]:
        T operator()[T1](T1 a)

    cdef cppclass fifth[T]:
        T operator()[T1](T1 a)

    cdef cppclass sixth[T]:
        T operator()[T1](T1 a)

    cdef cppclass seventh[T]:
        T operator()[T1](T1 a)

    cdef cppclass eighth[T]:
        T operator()[T1](T1 a)

    cdef cppclass ninth[T]:
        T operator()[T1](T1 a)


cdef extern from "src/unpack_args.hpp":
    cdef cppclass unpack_binary_args[Functor]:
        unpack_binary_args(Functor)

    cdef cppclass unpack_ternary_args[Functor]:
        unpack_ternary_args(Functor)

    cdef cppclass unpack_quaternary_args[Functor]:
        unpack_quaternary_args(Functor)

    cdef cppclass unpack_quinary_args[Functor]:
        unpack_quinary_args(Functor)

    cdef cppclass reverse_divides[T]:
        pass


cdef extern from "src/functional_tuples.hpp" namespace "cythrust" nogil:
    cdef cppclass reduce2pair[F1, F2]:
        pass

    cdef cppclass reduce2[F0, F1]:
        tuple2[T0, T1] operator()[T0, T1](T0 a,T1 b)

    cdef cppclass reduce3[F0, F1, F2]:
        tuple3[T0, T1, T2] operator()[T0, T1, T2](T0 a,T1 b,T2 c)

    cdef cppclass reduce4[F0, F1, F2, F3]:
        tuple4[T0, T1, T2, T3] operator()[T0, T1, T2, T3](T0 a,T1 b,T2 c,T3 d)

    cdef cppclass reduce5[F0, F1, F2, F3, F4]:
        tuple5[T0, T1, T2, T3, T4] operator()[T0, T1, T2, T3, T4](T0 a,T1 b,T2 c,T3 d,T4 e)

    cdef cppclass reduce6[F0, F1, F2, F3, F4, F5]:
        tuple6[T0, T1, T2, T3, T4, T5] operator()[T0, T1, T2, T3, T4, T5](T0 a,T1 b,T2 c,T3 d,T4 e,T5 f)

    cdef cppclass reduce7[F0, F1, F2, F3, F4, F5, F6]:
        tuple7[T0, T1, T2, T3, T4, T5, T6] operator()[T0, T1, T2, T3, T4, T5, T6](T0 a,T1 b,T2 c,T3 d,T4 e,T5 f,T6 g)

    cdef cppclass reduce8[F0, F1, F2, F3, F4, F5, F6, F7]:
        tuple8[T0, T1, T2, T3, T4, T5, T6, T7] operator()[T0, T1, T2, T3, T4, T5, T6, T7](T0 a,T1 b,T2 c,T3 d,T4 e,T5 f,T6 g,T7 h)

    cdef cppclass reduce9[F0, F1, F2, F3, F4, F5, F6, F7, F8]:
        tuple9[T0, T1, T2, T3, T4, T5, T6, T7, T8] operator()[T0, T1, T2, T3, T4, T5, T6, T7, T8](T0 a,T1 b,T2 c,T3 d,T4 e,T5 f,T6 g,T7 h,T8 i)
