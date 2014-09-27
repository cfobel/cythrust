from cythrust.iterator_traits cimport iterator_traits


cdef extern from "<thrust/reduce.h>" namespace "thrust" nogil:
    iterator_traits[InputIterator].value_type reduce[InputIterator](InputIterator first, InputIterator last)
