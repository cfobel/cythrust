cdef extern from "src/repeated_range_iterator.h" namespace "thrust" nogil:
    cdef cppclass repeated_range[Iterator, T]:
        cppclass iterator:
            pass

        repeated_range(Iterator first, T repeats)
        repeated_range[Iterator, T].iterator begin()
