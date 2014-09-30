#distutils: language = c++
#cython: embedsignature = True
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t
from cythrust.device_vector.int32.device_vector cimport DeviceVector as DeviceVectorInt32
from cythrust.thrust.copy cimport copy_n, copy
from cythrust.thrust.fill cimport fill_n
from cythrust.thrust.transform cimport transform, transform2
from cythrust.thrust.functional cimport (plus, negate, plus_tuple5, plus5,
                                         unpack_quinary_args)
from cythrust.thrust.iterator.transform_iterator cimport make_transform_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.thrust.tuple cimport make_tuple2, make_tuple5


def test_1(DeviceVectorInt32 u, DeviceVectorInt32 v, DeviceVectorInt32 w,
           DeviceVectorInt32 x, DeviceVectorInt32 y, DeviceVectorInt32 z):
    cdef plus[int32_t] plus_int32
    copy(u._vector.begin(), u._vector.end(), z._vector.begin())
    transform2(z._vector.begin(), z._vector.end(), v._vector.begin(),
               z._vector.begin(), plus_int32)
    transform2(z._vector.begin(), z._vector.end(), w._vector.begin(),
               z._vector.begin(), plus_int32)
    transform2(z._vector.begin(), z._vector.end(), x._vector.begin(),
               z._vector.begin(), plus_int32)
    transform2(z._vector.begin(), z._vector.end(), y._vector.begin(),
               z._vector.begin(), plus_int32)


def test_2(DeviceVectorInt32 u, DeviceVectorInt32 v, DeviceVectorInt32 w,
           DeviceVectorInt32 x, DeviceVectorInt32 y, DeviceVectorInt32 z):
    cdef plus5[int32_t] plus5_int32
    cdef unpack_quinary_args[plus5[int32_t]] *tuple_plus5_int32 = \
        new unpack_quinary_args[plus5[int32_t]](plus5_int32)

    transform(
        make_zip_iterator(make_tuple5(u._vector.begin(), v._vector.begin(),
                                      w._vector.begin(), x._vector.begin(),
                                      y._vector.begin())),
        make_zip_iterator(make_tuple5(u._vector.end(), v._vector.end(),
                                      w._vector.end(), x._vector.end(),
                                      y._vector.end())),
        z._vector.begin(), deref(tuple_plus5_int32))


def test_1(DeviceVectorInt32 u, DeviceVectorInt32 v, DeviceVectorInt32 w,
           DeviceVectorInt32 x, DeviceVectorInt32 y, DeviceVectorInt32 z):
    cdef plus[int32_t] plus_int32
    copy(u._vector.begin(), u._vector.end(), z._vector.begin())
    transform2(z._vector.begin(), z._vector.end(), v._vector.begin(),
               z._vector.begin(), plus_int32)
    transform2(z._vector.begin(), z._vector.end(), w._vector.begin(),
               z._vector.begin(), plus_int32)
    transform2(z._vector.begin(), z._vector.end(), x._vector.begin(),
               z._vector.begin(), plus_int32)
    transform2(z._vector.begin(), z._vector.end(), y._vector.begin(),
               z._vector.begin(), plus_int32)


def test_2(DeviceVectorInt32 u, DeviceVectorInt32 v, DeviceVectorInt32 w,
           DeviceVectorInt32 x, DeviceVectorInt32 y, DeviceVectorInt32 z):
    cdef plus5[int32_t] plus5_int32
    cdef unpack_quinary_args[plus5[int32_t]] *tuple_plus5_int32 = \
        new unpack_quinary_args[plus5[int32_t]](plus5_int32)

    transform(
        make_zip_iterator(make_tuple5(u._vector.begin(), v._vector.begin(),
                                      w._vector.begin(), x._vector.begin(),
                                      y._vector.begin())),
        make_zip_iterator(make_tuple5(u._vector.end(), v._vector.end(),
                                      w._vector.end(), x._vector.end(),
                                      y._vector.end())),
        z._vector.begin(), deref(tuple_plus5_int32))


def test_1_cpp(int32_t[:] u, int32_t[:] v, int32_t[:] w, int32_t[:] x,
               int32_t[:] y, int32_t[:] z):
    cdef size_t N = u.size
    cdef int i

    for i in xrange(N):
        z[i] = u[i]

    for i in xrange(N):
        z[i] += v[i]

    for i in xrange(N):
        z[i] += w[i]

    for i in xrange(N):
        z[i] += x[i]

    for i in xrange(N):
        z[i] += y[i]


def test_2_cpp(int32_t[:] u, int32_t[:] v, int32_t[:] w, int32_t[:] x,
               int32_t[:] y, int32_t[:] z):
    cdef size_t N = u.size
    cdef int i

    for i in xrange(N):
        z[i] = u[i] + v[i] + w[i] + x[i] + y[i]
