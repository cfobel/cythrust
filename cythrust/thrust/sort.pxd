cdef extern from "<thrust/sort.h>" namespace "thrust" nogil:
    void sort[RandomAccessIterator](RandomAccessIterator first,
                                    RandomAccessIterator last)
    void sort_by_op 'thrust::sort' [RandomAccessIterator, StrictWeakOrdering] \
        (RandomAccessIterator first, RandomAccessIterator last,
         StrictWeakOrdering comp)
    void stable_sort[RandomAccessIterator](RandomAccessIterator first,
                                           RandomAccessIterator last)
    void stable_sort_by_op 'thrust::stable_sort' [RandomAccessIterator,
                                                  StrictWeakOrdering] \
        (RandomAccessIterator first, RandomAccessIterator last,
         StrictWeakOrdering comp)
    void sort_by_key [RandomAccessIterator1, RandomAccessIterator2] \
        (RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
         RandomAccessIterator2 values_first)
    void sort_by_key_by_op 'thrust::sort_by_key' [RandomAccessIterator1,
                                                  RandomAccessIterator2,
                                                  StrictWeakOrdering] \
        (RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
         RandomAccessIterator2 values_first, StrictWeakOrdering comp)
