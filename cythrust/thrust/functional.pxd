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
    cdef cppclass plus_tuple5[T]:
        pass
