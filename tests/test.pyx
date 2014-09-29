# distutils: language = c++
from cython.operator cimport dereference as deref, preincrement as inc
from cython cimport typeof

from cythrust.thrust.device_vector cimport device_vector
from cythrust.thrust.fill cimport fill, fill_n, uninitialized_fill
from cythrust.thrust.copy cimport copy, copy_n
from cythrust.thrust.sequence cimport sequence
from cythrust.thrust.tuple cimport tuple2, tuple3, tuple4, make_tuple2
from cythrust.thrust.iterator.permutation_iterator cimport make_permutation_iterator
from cythrust.thrust.iterator.counting_iterator cimport make_counting_iterator, counting_iterator
from cythrust.thrust.iterator.discard_iterator cimport make_discard_iterator, discard_iterator
from cythrust.thrust.iterator.transform_iterator cimport make_transform_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.thrust.functional cimport (negate, identity, plus, multiplies, equal_to,
                                  greater)
from cythrust.thrust.reduce cimport (accumulate, reduce as reduce_, accumulate_by_key,
                              reduce_by_key)
from cythrust.thrust.iterator.iterator_traits cimport iterator_traits
from cythrust.thrust.pair cimport pair
from cythrust.thrust.sort cimport sort_by_key, sort_by_key_by_op


ctypedef int Value
ctypedef device_vector[Value].iterator ValueIterator


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
    print <Value>accumulate(v_ptr.begin(), v_ptr.end())

    cdef Value temp = <Value>accumulate(v_ptr.begin(), v_ptr.end())
    cdef identity[float] to_float

    print <float>accumulate(make_transform_iterator(v_ptr.begin(), to_float),
                            make_transform_iterator(v_ptr.end(), to_float))

    cdef plus[Value] plus_func
    cdef multiplies[float] multiply
    cdef equal_to[Value] eq

    sequence(v_ptr.begin(), v_ptr.end(), 1)
    sequence(u_ptr.begin(), u_ptr.end(), 1)

    print reduce_(make_transform_iterator(v_ptr.begin(), to_float),
                  make_transform_iterator(v_ptr.end(), to_float),
                  1, multiply)

    accumulate_by_key(v_ptr.begin(), v_ptr.end(), u_ptr.begin(), v_ptr.begin(),
                      u_ptr.begin())
    cdef pair[ValueIterator, ValueIterator] result = \
        reduce_by_key(v_ptr.begin(), v_ptr.end(), u_ptr.begin(), v_ptr.begin(),
                      u_ptr.begin(), eq, plus_func)

    cdef int reduce_count = (<ValueIterator>result.first - v_ptr.begin())

    print ''
    print '----------------------------------------'
    print ''
    print 'reduce_count:', reduce_count
    print ''
    print '----------------------------------------'
    print ''

    for i in xrange(reduce_count):
        print '%d, ' % deref(v_ptr)[i],
    print ''
    for i in xrange(reduce_count):
        print '%d, ' % deref(u_ptr)[i],
    print ''

    print ''
    print '----------------------------------------'
    print ''

    cdef greater[Value] greater_than
    sort_by_key_by_op(v_ptr.begin(), v_ptr.end(), u_ptr.begin(), greater_than)

    for i in xrange(reduce_count):
        print '%d, ' % deref(v_ptr)[i],
    print ''
    for i in xrange(reduce_count):
        print '%d, ' % deref(u_ptr)[i],
    print ''

    print ''
    print '----------------------------------------'
    print ''

    sort_by_key(v_ptr.begin(), v_ptr.end(), u_ptr.begin())

    for i in xrange(reduce_count):
        print '%d, ' % deref(v_ptr)[i],
    print ''
    for i in xrange(reduce_count):
        print '%d, ' % deref(u_ptr)[i],
    print ''

    del v_ptr
    del u_ptr
