from cythrust.thrust.pair cimport pair


cdef extern from "<thrust/extrema.h>" namespace "thrust" nogil:
    ForwardIterator min_element [ForwardIterator] (ForwardIterator first,
                                                   ForwardIterator last)

    ForwardIterator min_element_w_predicate 'thrust::min_element' \
        [ForwardIterator, BinaryPredicate] (ForwardIterator first,
                                            ForwardIterator last,
                                            BinaryPredicate comp)

    ForwardIterator max_element [ForwardIterator] (ForwardIterator first,
                                                   ForwardIterator last)

    ForwardIterator max_element_w_predicate 'thrust::max_element' \
        [ForwardIterator, BinaryPredicate] (ForwardIterator first,
                                            ForwardIterator last,
                                            BinaryPredicate comp)

    pair[ForwardIterator, ForwardIterator] minmax_element [ForwardIterator] \
        (ForwardIterator first, ForwardIterator last)

    pair[ForwardIterator, ForwardIterator] minmax_element_w_predicate \
        'thrust::minmax_element' [ForwardIterator, BinaryPredicate] \
        (ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
