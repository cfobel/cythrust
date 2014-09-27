from cythrust.iterator_traits cimport iterator_traits


cdef extern from "<thrust/reduce.h>" namespace "thrust" nogil:
    iterator_traits[InputIterator].value_type reduce_sum 'thrust::reduce' [InputIterator](InputIterator first, InputIterator last)
    T reduce 'thrust::reduce' [InputIterator, T, BinaryFunction](InputIterator
                                                                 first,
                                                                 InputIterator
                                                                 last, T
                                                                 init_value,
                                                                 BinaryFunction
                                                                 binary_op)
