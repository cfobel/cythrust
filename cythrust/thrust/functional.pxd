cdef extern from "<thrust/functional.h>" namespace "thrust" nogil:
    cdef cppclass unary_function[Argument, Result]:
        pass
    cdef cppclass binary_function[Argument1, Argument2, Result]:
        pass
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
    cdef cppclass negate[T]:
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
    cdef cppclass logical_not[T]:
        pass
    cdef cppclass bit_and[T]:
        pass
    cdef cppclass bit_or[T]:
        pass
    cdef cppclass bit_xor[T]:
        pass
    cdef cppclass identity[T]:
        pass
    cdef cppclass maximum[T]:
        pass
    cdef cppclass minimum[T]:
        pass
    cdef cppclass project1st[T1, T2]:
        pass
    cdef cppclass project2nd[T1, T2]:
        pass
    cdef cppclass unary_negate[Predicate]:
        pass
    cdef cppclass binary_negate[Predicate]:
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

    cdef cppclass minmax[T]:
        pass

    cdef cppclass absolute[T]:
        pass

    cdef cppclass square[T]:
        pass

    cdef cppclass reduce_plus4[T]:
        pass

    cdef cppclass plus4[T]:
        pass

    cdef cppclass plus5[T]:
        pass

    cdef cppclass plus_tuple5[T]:
        pass


cdef extern from "src/unpack_args.hpp":
    cdef cppclass unpack_binary_args[Functor]:
        unpack_binary_args(Functor)

    cdef cppclass unpack_ternary_args[Functor]:
        unpack_ternary_args(Functor)

    cdef cppclass unpack_quaternary_args[Functor]:
        unpack_quaternary_args(Functor)

    cdef cppclass unpack_quinary_args[Functor]:
        unpack_quinary_args(Functor)
