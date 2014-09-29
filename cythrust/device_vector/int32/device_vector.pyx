# cython: embedsignature = True

from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t

from cythrust.thrust.device_vector cimport device_vector


ctypedef int32_t Value
DTYPE = np.int32
ctypedef device_vector[Value] Vector
ctypedef device_vector[Value].iterator Iterator


cdef class DeviceVector:
    cdef Vector *_vector

    def __cinit__(self, size_t size=0):
        self._vector = new Vector(size)

    def resize(self, size_t size):
        '''
        Resize the device vector to `size`.
        '''
        self._vector.resize(size)

    def astype(self, dtype):
        '''
        Return a _copy_ of the device vector as a `numpy` array of type
        `dtype`.
        '''
        return self.as_array().astype(dtype)

    def as_array(self):
        '''
        Return a _copy_ of the device vector as a `numpy` array.
        '''
        cdef int i
        return np.fromiter((deref(self._vector)[i] for i in xrange(self.size)),
                           dtype=DTYPE)

    def __dealloc__(self):
        del self._vector

    property size:
        def __get__(self):
            return self._vector.size()

        def __set__(self, value):
            self._vector.resize(value)
