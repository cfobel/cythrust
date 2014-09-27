from cython.operator cimport dereference as deref, preincrement as inc
from cython cimport typeof

from cythrust.device_vector cimport device_vector
from cythrust.fill cimport fill, fill_n, uninitialized_fill
from cythrust.copy cimport copy, copy_n
from cythrust.sequence cimport sequence
from cythrust.tuple cimport tuple2, tuple3, tuple4, make_tuple2
from cythrust.permutation_iterator cimport make_permutation_iterator
from cythrust.counting_iterator cimport make_counting_iterator, counting_iterator
from cythrust.discard_iterator cimport make_discard_iterator, discard_iterator
from cythrust.transform_iterator cimport make_transform_iterator
from cythrust.zip_iterator cimport make_zip_iterator
from cythrust.functional cimport negate, identity, plus, multiplies
from cythrust.reduce cimport reduce_sum, reduce as reduce_
from cythrust.iterator_traits cimport iterator_traits


ctypedef int Value


def test():
    cdef device_vector[Value] *v_ptr
    u_ptr = new device_vector[Value](10)
    v_ptr = new device_vector[Value](10)

    sequence(u_ptr.begin(), u_ptr.end())
    fill(v_ptr.begin(), v_ptr.end(), 1)
    copy(u_ptr.begin(), u_ptr.end(), v_ptr.begin())

    cdef int i

    for i in xrange(v_ptr.size()):
        print deref(v_ptr)[i]

    cdef tuple2[Value, Value] test = make_tuple2[Value, Value](1, 2)
    cdef counting_iterator[long] c = make_counting_iterator(42)

    fill(v_ptr.begin(), v_ptr.end(), 1)
    copy_n(c, v_ptr.size(), v_ptr.begin())

    print ''
    print '----------------------------------------'
    print ''

    for i in xrange(10):
        print deref(v_ptr)[i]

    copy_n(
        make_permutation_iterator(v_ptr.begin(), make_counting_iterator(0)),
        v_ptr.size(), u_ptr.begin())

    copy(v_ptr.begin(), v_ptr.end(), make_discard_iterator())

    cdef negate[Value] n

    copy_n(
        make_transform_iterator(v_ptr.begin(), n),
        v_ptr.size(), u_ptr.begin())

    copy_n(
        make_zip_iterator(
            make_tuple2(v_ptr.begin(), v_ptr.begin())),
        v_ptr.size(),
        make_zip_iterator(
            make_tuple2(u_ptr.begin(), u_ptr.begin())))

    print ''
    print '----------------------------------------'
    print ''

    cdef int v_sum = 0

    for i in xrange(10):
        print deref(u_ptr)[i]
        v_sum += deref(v_ptr)[i]

    print ''
    print '----------------------------------------'
    print ''
    print v_sum
    print <Value>reduce_sum(v_ptr.begin(), v_ptr.end())

    cdef Value temp = <Value>reduce_sum(v_ptr.begin(), v_ptr.end())
    cdef identity[float] to_float

    print <float>reduce_sum(make_transform_iterator(v_ptr.begin(), to_float),
                            make_transform_iterator(v_ptr.end(), to_float))

    cdef multiplies[float] multiply

    sequence(v_ptr.begin(), v_ptr.end(), 1)

    print reduce_(make_transform_iterator(v_ptr.begin(), to_float),
                  make_transform_iterator(v_ptr.end(), to_float),
                  1, multiply)

    del v_ptr
    del u_ptr
