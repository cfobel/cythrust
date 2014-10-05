cdef extern from "<thrust/scan.h>" namespace "thrust" nogil:
    OutputIterator inclusive_scan [InputIterator, OutputIterator] \
        (InputIterator first, InputIterator last, OutputIterator result)
    OutputIterator inclusive_scan_w_op 'thrust::inclusive_scan' [InputIterator,
                                                                 OutputIterator,
                                                                 AssociativeOperator] \
        (InputIterator first, InputIterator last, OutputIterator result,
         AssociativeOperator binary_op)

    OutputIterator exclusive_scan [InputIterator, OutputIterator] \
        (InputIterator first, InputIterator last, OutputIterator result)
    OutputIterator exclusive_scan_w_init 'thrust::exclusive_scan' \
        [InputIterator, OutputIterator, T] (InputIterator first,
                                            InputIterator last,
                                            OutputIterator result, T init)
    OutputIterator exclusive_scan_w_init_op 'thrust::exclusive_scan' \
        [InputIterator, OutputIterator, T, AssociativeOperator] \
        (InputIterator first, InputIterator last, OutputIterator result,
         T init, AssociativeOperator binary_op)
