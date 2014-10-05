cdef extern from "<thrust/sequence.h>" namespace "thrust" nogil:
    void sequence[ForwardIterator](ForwardIterator first, ForwardIterator last)
    void sequence_w_init 'thrust::sequence' [ForwardIterator, T] \
        (ForwardIterator first, ForwardIterator last, T init)
    void sequence_w_init_step 'thrust::sequence' [ForwardIterator, T] \
        (ForwardIterator first, ForwardIterator last, T init, T step)
