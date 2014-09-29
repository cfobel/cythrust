# distutils: language = c++
# distutils: include_dirs = c++

from libc.stdint cimport int32_t, uint32_t
from cythrust.functional cimport UnaryIntFunction
from cythrust.thrust.functional cimport (unary_function, binary_function,
                                         negate, plus)

cdef class NegateInt(UnaryIntFunction):
    def __cinit__(self):
        self.func = <unary_function[int32_t, int32_t]*>(new negate[int32_t]())

    def __dealloc__(self):
        del self.func

    def data(self):
        return <size_t>self.func


cdef class PlusInt:
    def __cinit__(self):
        self.func = new plus[int32_t]()

    def __dealloc__(self):
        del self.func
