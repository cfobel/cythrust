cdef extern from "<thrust/tuple.h>" namespace "thrust" nogil:
    #cdef cppclass tuple_element[N, T]
    #cdef struct tuple_size[T]
    #cdef cppclass tuple[T0, T1]
    cdef cppclass tuple[T0, T1]:
        pass
