cdef extern from "<thrust/iterator/discard_iterator.h>" namespace "thrust" nogil:
    cdef cppclass discard_iterator 'thrust::discard_iterator<>':
        pass

    discard_iterator make_discard_iterator()
