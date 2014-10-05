cdef extern from "<thrust/transform_scan.h>" namespace "thrust" nogil:
    OutputIterator transform_inclusive_scan [InputIterator, OutputIterator,
                                             UnaryFunction,
                                             AssociativeOperator] \
        (InputIterator first, InputIterator last, OutputIterator result,
         UnaryFunction unary_op, AssociativeOperator binary_op)
    OutputIterator transform_exclusive_scan_w_init \
        'thrust::transform_exclusive_scan' [InputIterator, OutputIterator,
                                            UnaryFunction, T,
                                            AssociativeOperator] \
        (InputIterator first, InputIterator last, OutputIterator result,
         UnaryFunction unary_op, T init, AssociativeOperator binary_op)
