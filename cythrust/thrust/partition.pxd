cdef extern from "<thrust/partition.h>" namespace "thrust" nogil:
   InputIterator partition[InputIterator, Predicate](InputIterator first,
                                                     InputIterator last,
                                                     Predicate pred)

   InputIterator1 partition_w_stencil 'thrust::partition' [InputIterator1,
                                                           InputIterator2,
                                                           Predicate] \
       (InputIterator1 first, InputIterator1 last, InputIterator2 stencil,
        Predicate pred)

   InputIterator stable_partition[InputIterator, Predicate](InputIterator
                                                            first,
                                                            InputIterator last,
                                                            Predicate pred)

   InputIterator1 stable_partition_w_stencil 'thrust::stable_partition'\
       [InputIterator1, InputIterator2, Predicate] \
       (InputIterator1 first, InputIterator1 last, InputIterator2 stencil,
        Predicate pred)


cdef extern from "src/partition.h" namespace "thrust" nogil:
    int counted_partition[InputIterator, Predicate](InputIterator first,
                                                   InputIterator last,
                                                   Predicate pred)

    int counted_partition_w_stencil 'thrust::counted_partition' \
        [InputIterator1, InputIterator2, Predicate] \
        (InputIterator1 first, InputIterator1 last, InputIterator2 last,
         Predicate pred)

    int counted_stable_partition 'thrust::counted_stable_partition' \
        [InputIterator, Predicate](InputIterator first, InputIterator last,
                                   Predicate pred)

    int counted_stable_partition_w_stencil 'thrust::counted_stable_partition' \
        [InputIterator1, InputIterator2, Predicate] \
        (InputIterator1 first, InputIterator1 last, InputIterator2 last,
         Predicate pred)
