cdef extern from "<thrust/pair.h>" namespace "thrust" nogil:
    cdef cppclass pair[T1, T2]:
        cppclass first_type:
            pass
        cppclass second_type:
            pass
        first_type first
        second_type second
