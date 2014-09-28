cdef extern from "<thrust/iterator/counting_iterator.h>" namespace "thrust" nogil:
    cdef cppclass counting_iterator[Incrementable]:
        pass

    counting_iterator[Incrementable] make_counting_iterator[Incrementable](Incrementable x)
