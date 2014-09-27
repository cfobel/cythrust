cdef extern from "<thrust/iterator/transform_iterator.h>" namespace "thrust" nogil:
    cdef cppclass transform_iterator[AdaptableUnaryFunction, Iterator]:
        pass

    transform_iterator[AdaptableUnaryFunction, Iterator] make_transform_iterator[AdaptableUnaryFunction, Iterator](Iterator it, AdaptableUnaryFunction fun)
