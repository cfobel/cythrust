cdef extern from "<thrust/copy.h>" namespace "thrust" nogil:
   OutputIterator copy[InputIterator, OutputIterator](InputIterator first,
                                                      InputIterator last,
                                                      OutputIterator result)
   OutputIterator copy_n[InputIterator, Size, OutputIterator](InputIterator
                                                              first, Size n,
                                                              OutputIterator
                                                              result)
   OutputIterator copy_if[InputIterator, OutputIterator,
                          Predicate](InputIterator first, InputIterator last,
                                     OutputIterator result, Predicate pred)

   OutputIterator copy_if_w_stencil 'thrust::copy_if' [InputIterator1,
                                                       InputIterator2,
                                                       OutputIterator,
                                                       Predicate] \
       (InputIterator1 first, InputIterator1 last, InputIterator2 stencil,
        OutputIterator result, Predicate pred)
