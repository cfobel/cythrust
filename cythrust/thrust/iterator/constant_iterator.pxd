cdef extern from "<thrust/iterator/constant_iterator.h>" namespace "thrust" nogil:
    cdef cppclass constant_iterator[Value]:
        constant_iterator()
        constant_iterator(Value)

        constant_iterator[Value] operator++()
        constant_iterator[Value] operator--()
        constant_iterator[Value] operator+(size_t)
        constant_iterator[Value] operator-(size_t)

    constant_iterator[Value] make_constant_iterator[Value](Value x)
