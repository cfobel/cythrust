cdef extern from "<thrust/iterator/permutation_iterator.h>" namespace "thrust" nogil:
    cdef cppclass permutation_iterator[ElementIterator, IndexIterator]:
        pass

    permutation_iterator[ElementIterator, IndexIterator] make_permutation_iterator[ElementIterator, IndexIterator](ElementIterator e, IndexIterator i)
