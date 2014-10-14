from cythrust.thrust.pair cimport pair


cdef extern from "src/unique.h" namespace "thrust" nogil:
    int counted_unique [ForwardIterator] (ForwardIterator first,
                                          ForwardIterator last)

    int counted_unique_w_predicate 'thrust::counted_unique' \
        [ForwardIterator, BinaryPredicate] (ForwardIterator first,
                                            ForwardIterator last,
                                            BinaryPredicate binary_pred)


cdef extern from "<thrust/unique.h>" namespace "thrust" nogil:
    ForwardIterator unique [ForwardIterator] (ForwardIterator first,
                                              ForwardIterator last)

    ForwardIterator unique_w_predicate 'thrust::unique' \
        [ForwardIterator, BinaryPredicate] (ForwardIterator first,
                                            ForwardIterator last,
                                            BinaryPredicate binary_pred)

    OutputIterator unique_copy [InputIterator, OutputIterator] \
        (InputIterator first, InputIterator last, OutputIterator result)

    OutputIterator unique_copy_w_predicate 'thrust::unique_copy' \
        [InputIterator, OutputIterator, BinaryPredicate] \
        (InputIterator first, InputIterator last, OutputIterator result,
         BinaryPredicate binary_pred)

    pair[ForwardIterator1, ForwardIterator2] unique_by_key [ForwardIterator1,
                                                            ForwardIterator2] \
        (ForwardIterator1 keys_first, ForwardIterator1 keys_last,
         ForwardIterator2 values_first)

    pair[ForwardIterator1, ForwardIterator2] unique_by_key_with_predicate \
        'thrust::unique_by_key' [ForwardIterator1, ForwardIterator2,
                                 BinaryPredicate] \
        (ForwardIterator1 keys_first, ForwardIterator1 keys_last,
         ForwardIterator2 values_first, BinaryPredicate binary_pred)

    pair[OutputIterator1, OutputIterator2] unique_by_key_copy \
        [InputIterator1, InputIterator2, OutputIterator1, OutputIterator2] \
        (InputIterator1 keys_first, InputIterator1 keys_last,
         InputIterator2 values_first, OutputIterator1 keys_result,
         OutputIterator2 values_result)

    pair[OutputIterator1, OutputIterator2] unique_by_key_copy_with_predicate \
        'thrust::unique_by_key_copy' [InputIterator1, InputIterator2,
                                      OutputIterator1, OutputIterator2,
                                      BinaryPredicate] \
        (InputIterator1 keys_first, InputIterator1 keys_last,
         InputIterator2 values_first, OutputIterator1 keys_result,
         OutputIterator2 values_result, BinaryPredicate binary_pred)
