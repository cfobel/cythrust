cdef extern from "<thrust/iterator/counting_iterator.h>" namespace "thrust" nogil:
    cdef cppclass counting_iterator[Incrementable]:
        counting_iterator()
        counting_iterator(Incrementable start)

        counting_iterator[Incrementable] operator++()
        counting_iterator[Incrementable] operator--()
        counting_iterator[Incrementable] operator+(size_t)
        counting_iterator[Incrementable] operator-(size_t)

    counting_iterator[Incrementable] make_counting_iterator[Incrementable](Incrementable x)
