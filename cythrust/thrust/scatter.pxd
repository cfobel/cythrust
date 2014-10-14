cdef extern from "<thrust/scatter.h>" namespace "thrust" nogil:
    void scatter [InputIterator1, InputIterator2, RandomAccessIterator] \
        (InputIterator1 first, InputIterator1 last, InputIterator2 map,
         RandomAccessIterator result)

    void scatter_if [InputIterator1, InputIterator2, InputIterator3,
                     RandomAccessIterator] \
        (InputIterator1 first, InputIterator1 last, InputIterator2 map,
         InputIterator3 stencil, RandomAccessIterator output)

    void scatter_if_w_predicate 'thrust::scatter_if' [InputIterator1,
                                                      InputIterator2,
                                                      InputIterator3,
                                                      RandomAccessIterator,
                                                      Predicate] \
        (InputIterator1 first, InputIterator1 last, InputIterator2 map,
         InputIterator3 stencil, RandomAccessIterator output, Predicate pred)
