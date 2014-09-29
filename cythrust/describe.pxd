from libcpp.string cimport string
from libc.stdint cimport (int8_t, uint8_t, int16_t, uint16_t, int32_t,
                          uint32_t, int64_t, uint64_t)


cdef extern from "src/Describe.hpp" nogil:
    cdef cppclass _DescribeSummary 'DescribeSummary' [Real]:
        Real sum
        Real mean
        Real std
        Real min
        Real max
        Real count
        Real non_zero_count

        void dump()
        void dump_si()
        string str()
        string str_si()

    double get_std_dev(int n, double sum_x_squared, double av_x)

    _DescribeSummary[float] describe_f[ValueIterator](ValueIterator start,
                                                      ValueIterator end)
    _DescribeSummary[double] describe_d[ValueIterator](ValueIterator start,
                                                       ValueIterator end)



ctypedef fused array_t:
    int8_t[:]
    uint8_t[:]
    int16_t[:]
    uint16_t[:]
    int32_t[:]
    uint32_t[:]
    int64_t[:]
    uint64_t[:]
    float[:]
    double[:]


cdef class DescribeSummary:
    cdef _DescribeSummary[float] *data


cdef from_cdescribe(_DescribeSummary[float] other)
