cdef extern from "<thrust/sequence.h>" namespace "thrust":
    void sequence[ForwardIterator](ForwardIterator first, ForwardIterator last)
    void sequence[ForwardIterator, T](ForwardIterator first, ForwardIterator last, T init)
    void sequence[ForwardIterator, T](ForwardIterator first, ForwardIterator last, T init, T step)
