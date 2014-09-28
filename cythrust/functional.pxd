from libc.stdint cimport int32_t, uint32_t
from cythrust.thrust.functional cimport unary_function, binary_function, plus


cdef class UnaryIntFunction:
    cdef unary_function[int32_t, int32_t] *func


cdef class BinaryIntFunction:
    cdef binary_function[int32_t, int32_t, int32_t] *func


cdef class PlusInt:
    cdef plus[int32_t] *func

    cdef inline plus[int32_t] *func(self):
        return self.func
