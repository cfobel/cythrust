cdef extern from "<thrust/iterator/zip_iterator.h>" namespace "thrust" nogil:
    cdef cppclass zip_iterator[IteratorTuple]:
        zip_iterator[IteratorTuple] operator++()
        zip_iterator[IteratorTuple] operator--()
        zip_iterator[IteratorTuple] operator+(size_t)
        zip_iterator[IteratorTuple] operator-(size_t)
        zip_iterator[IteratorTuple] operator-(zip_iterator[IteratorTuple])

    zip_iterator[IteratorTuple] make_zip_iterator[IteratorTuple](IteratorTuple
                                                                 t)
