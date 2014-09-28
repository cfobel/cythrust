cdef extern from "<thrust/iterator/iterator_traits.h>" namespace "thrust" nogil:
    cdef cppclass iterator_traits[T]:
        cppclass value_type:
            pass
