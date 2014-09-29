# distutils: language = c++
cdef class DescribeSummary:
    def __cinit__(self):
        self.data = new _DescribeSummary[float]()

    def __dealloc__(self):
        del self.data

    property sum:
        def __get__(self):
            return self.data.sum

        def __set__(self, value):
            self.data.sum = value

    property mean:
        def __get__(self):
            return self.data.mean

        def __set__(self, value):
            self.data.mean = value

    property std:
        def __get__(self):
            return self.data.std

        def __set__(self, value):
            self.data.std = value

    property min:
        def __get__(self):
            return self.data.min

        def __set__(self, value):
            self.data.min = value

    property max:
        def __get__(self):
            return self.data.max

        def __set__(self, value):
            self.data.max = value

    property count:
        def __get__(self):
            return self.data.count

        def __set__(self, value):
            self.data.count = value

    property non_zero_count:
        def __get__(self):
            return self.data.non_zero_count

        def __set__(self, value):
            self.data.non_zero_count = value

    def str_si(self):
        return self.data.str_si()

    def str(self):
        return self.data.str()


cdef from_cdescribe(_DescribeSummary[float] other):
    result = DescribeSummary()

    result.sum = other.sum
    result.mean = other.mean
    result.std = other.std
    result.min = other.min
    result.max = other.max
    result.count = other.count
    result.non_zero_count = other.non_zero_count

    return result


def describe(array_t u):
    cdef _DescribeSummary[float] info = describe_f(&u[0],
                                                   &u[0] + <size_t>u.size)
    return from_cdescribe(info)
