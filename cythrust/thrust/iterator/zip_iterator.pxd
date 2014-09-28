cdef extern from "<thrust/iterator/zip_iterator.h>" namespace "thrust" nogil:
    cdef cppclass zip_iterator[IteratorTuple]:
        pass

    zip_iterator[IteratorTuple] make_zip_iterator[IteratorTuple](IteratorTuple
                                                                 t)
