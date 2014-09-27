cdef extern from "<thrust/fill.h>" namespace "thrust":
   void fill[ForwardIterator, T](ForwardIterator first, ForwardIterator last,
                                 const T &value)
   OutputIterator fill_n[OutputIterator, Size, T](OutputIterator first,
                                                  Size n, const T &value)
   void uninitialized_fill[ForwardIterator, T](ForwardIterator first,
                                               ForwardIterator last,
                                               const T &x)
