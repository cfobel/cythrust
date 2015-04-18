cdef extern from "<thrust/tuple.h>" namespace "thrust" nogil:
    cdef cppclass tuple2 "thrust::tuple" [T0, T1]:
        tuple2()
        tuple2(T0 a, T1 b)

    cdef cppclass tuple3 "thrust::tuple" [T0, T1, T2]:
        tuple3()
        tuple3(T0 a, T1 b, T2 c)

    cdef cppclass tuple4 "thrust::tuple" [T0, T1, T2, T3]:
        tuple4()
        tuple4(T0 a, T1 b, T2 c, T3 d)

    cdef cppclass tuple5 "thrust::tuple" [T0, T1, T2, T3, T4]:
        tuple5()
        tuple5(T0 a, T1 b, T2 c, T3 d, T4 e)

    cdef cppclass tuple6 "thrust::tuple" [T0, T1, T2, T3, T4, T5]:
        tuple6()
        tuple6(T0 a, T1 b, T2 c, T3 d, T4 e, T5 f)

    cdef cppclass tuple7 "thrust::tuple" [T0,T1,T2,T3,T4,T5,T6]:
        tuple7()
        tuple7(T0 a,T1 b,T2 c,T3 d,T4 e,T5 f,T6 g)

    cdef cppclass tuple8 "thrust::tuple" [T0,T1,T2,T3,T4,T5,T6,T7]:
        tuple8()
        tuple8(T0 a,T1 b,T2 c,T3 d,T4 e,T5 f,T6 g,T7 h)

    cdef cppclass tuple9 "thrust::tuple" [T0,T1,T2,T3,T4,T5,T6,T7,T8]:
        tuple9()
        tuple9(T0 a,T1 b,T2 c,T3 d,T4 e,T5 f,T6 g,T7 h,T8 i)

    tuple2[T0, T1] make_tuple2 "thrust::make_tuple" [T0, T1](T0 a, T1 b)
    tuple3[T0, T1, T2] make_tuple3 "thrust::make_tuple" [T0, T1, T2](T0 a, T1 b, T2 c)
    tuple4[T0, T1, T2, T3] make_tuple4 "thrust::make_tuple" [T0, T1, T2, T3](T0 a, T1 b, T2 c, T3 d)
    tuple5[T0, T1, T2, T3, T4] make_tuple5 "thrust::make_tuple" [T0, T1, T2, T3, T4](T0 a, T1 b, T2 c, T3 d, T4 e)
    tuple6[T0, T1, T2, T3, T4, T5] make_tuple6 "thrust::make_tuple" [T0, T1, T2, T3, T4, T5](T0 a, T1 b, T2 c, T3 d, T4 e, T5 f)
    tuple7[T0, T1, T2, T3, T4, T5, T6] make_tuple7 "thrust::make_tuple" [T0, T1, T2, T3, T4, T5, T6](T0 a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g)
    tuple8[T0, T1, T2, T3, T4, T5, T6, T7] make_tuple8 "thrust::make_tuple" [T0, T1, T2, T3, T4, T5, T6, T7] (T0 a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h)
    tuple9[T0, T1, T2, T3, T4, T5, T6, T7, T8] make_tuple9 "thrust::make_tuple" [T0, T1, T2, T3, T4, T5, T6, T7, T8] (T0 a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i)
