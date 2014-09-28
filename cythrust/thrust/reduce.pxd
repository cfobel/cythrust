from cythrust.thrust.iterator.iterator_traits cimport iterator_traits
from cythrust.thrust.pair cimport pair


cdef extern from "<thrust/reduce.h>" namespace "thrust" nogil:
    iterator_traits[InputIterator].value_type accumulate 'thrust::reduce' [InputIterator](InputIterator first, InputIterator last)
    T reduce 'thrust::reduce' [InputIterator, T, BinaryFunction](InputIterator
                                                                 first,
                                                                 InputIterator
                                                                 last, T
                                                                 init_value,
                                                                 BinaryFunction
                                                                 binary_op)
    pair[OutputIterator1, OutputIterator2] \
        accumulate_by_key 'thrust::reduce_by_key' \
        [InputIterator1, InputIterator2, OutputIterator1,
         OutputIterator2](InputIterator1 keys_first, InputIterator1 keys_last,
                          InputIterator2 values_first,
                          OutputIterator1 keys_output,
                          OutputIterator2 values_output)

    pair[OutputIterator1, OutputIterator2] reduce_by_key \
        [InputIterator1, InputIterator2, OutputIterator1, OutputIterator2,
         BinaryPredicate, BinaryFunction](InputIterator1 keys_first,
                                          InputIterator1 keys_last,
                                          InputIterator2 values_first,
                                          OutputIterator1 keys_output,
                                          OutputIterator2 values_output,
                                          BinaryPredicate binary_pred,
                                          BinaryFunction binary_op)
