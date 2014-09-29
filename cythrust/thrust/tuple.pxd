cdef extern from "<thrust/tuple.h>" namespace "thrust" nogil:
    cdef cppclass tuple2 "thrust::tuple" [T0, T1]:
        pass

    cdef cppclass tuple3 "thrust::tuple" [T0, T1, T2]:
        pass

    cdef cppclass tuple4 "thrust::tuple" [T0, T1, T2, T3]:
        pass

    cdef cppclass tuple5 "thrust::tuple" [T0, T1, T2, T3, T4]:
        pass

    tuple2[T0, T1] make_tuple2 "thrust::make_tuple" [T0, T1](T0 a, T1 b)
    tuple3[T0, T1, T2] make_tuple3 "thrust::make_tuple" [T0, T1, T2](T0 a, T1 b, T2 c)
    tuple4[T0, T1, T2, T3] make_tuple4 "thrust::make_tuple" [T0, T1, T2, T3](T0 a, T1 b, T2 c, T3 d)
    tuple5[T0, T1, T2, T3, T4] make_tuple5 "thrust::make_tuple" [T0, T1, T2, T3, T4](T0 a, T1 b, T2 c, T3 d, T4 e)