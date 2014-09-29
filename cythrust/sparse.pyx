# distutils: language = c++
from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np
from libc.stdint cimport (uint32_t, int32_t, int64_t, uint64_t, INT32_MIN,
                          INT64_MIN, INT32_MAX, UINT32_MAX, INT64_MAX,
                          UINT64_MAX)

from cythrust.thrust.pair cimport pair
from cythrust.thrust.sort cimport sort_by_key
from cythrust.thrust.reduce cimport (accumulate_by_key, reduce_by_key)
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.thrust.tuple cimport make_tuple2
from cythrust.thrust.functional cimport equal_to, minimum, maximum


def sort_coo(coo, axis=0):
    # `scipy.sparse.coo_matrix` uses `np.int32` data-type for row and column
    # index values.
    cdef np.ndarray primary_keys
    cdef np.ndarray secondary_keys
    if axis == 0:
        primary_keys = coo.row
        secondary_keys = coo.col
    elif axis == 1:
        primary_keys = coo.col
        secondary_keys = coo.row
    else:
        raise ValueError('`axis` must be either 0 or 1, not %s.' % axis)

    cdef np.ndarray values = coo.data
    cdef int32_t *primary_first = <int32_t *>&primary_keys.data[0]
    cdef int32_t *primary_last = primary_first + <size_t>primary_keys.size
    cdef int32_t *secondary_first = <int32_t *>&secondary_keys.data[0]
    cdef int32_t *secondary_last = (secondary_first +
                                    <size_t>secondary_keys.size)

    if values.dtype.type == np.uint32:
        sort_by_key(
            make_zip_iterator(make_tuple2(primary_first, secondary_first)),
            make_zip_iterator(make_tuple2(primary_last, secondary_last)),
            <uint32_t *>&values.data[0])
        return

    raise ValueError('Unsupported data type: %s' % (values.dtype.type))


def sum_coo(coo, axis=0):
    # `scipy.sparse.coo_matrix` uses `np.int32` data-type for row and column
    # index values.
    cdef np.ndarray keys
    if axis == 0:
        keys = coo.row
    elif axis == 1:
        keys = coo.col
    else:
        raise ValueError('`axis` must be either 0 or 1, not %s.' % axis)

    cdef np.ndarray values = coo.data
    cdef int32_t *keys_first = <int32_t *>&keys.data[0]
    cdef int32_t *keys_last = keys_first + <size_t>keys.size
    cdef int count = 0

    if values.dtype.type == np.uint32:
        count = (<int32_t *>accumulate_by_key(keys_first, keys_last,
                                              <uint32_t *>&values.data[0],
                                              keys_first,
                                              <uint32_t *>&values.data[0])
                 .first - keys_first)
        return count

    raise ValueError('Unsupported data type: %s' % (values.dtype.type))


def min_coo(coo, axis=0):
    # `scipy.sparse.coo_matrix` uses `np.int32` data-type for row and column
    # index values.
    cdef np.ndarray keys
    if axis == 0:
        keys = coo.row
    elif axis == 1:
        keys = coo.col
    else:
        raise ValueError('`axis` must be either 0 or 1, not %s.' % axis)

    cdef np.ndarray values = coo.data
    cdef int32_t *keys_first = <int32_t *>&keys.data[0]
    cdef int32_t *keys_last = keys_first + <size_t>keys.size
    cdef int count = 0
    cdef equal_to[int32_t] eq
    cdef minimum[uint32_t] _minimum

    if values.dtype.type == np.uint32:
        count = (<int32_t *>reduce_by_key(keys_first, keys_last,
                                          <uint32_t *>&values.data[0],
                                          keys_first,
                                          <uint32_t *>&values.data[0],
                                          eq, _minimum)
                 .first - keys_first)
        return count

    raise ValueError('Unsupported data type: %s' % (values.dtype.type))


def max_coo(coo, axis=0):
    # `scipy.sparse.coo_matrix` uses `np.int32` data-type for row and column
    # index values.
    cdef np.ndarray keys
    if axis == 0:
        keys = coo.row
    elif axis == 1:
        keys = coo.col
    else:
        raise ValueError('`axis` must be either 0 or 1, not %s.' % axis)

    cdef np.ndarray values = coo.data
    cdef int32_t *keys_first = <int32_t *>&keys.data[0]
    cdef int32_t *keys_last = keys_first + <size_t>keys.size
    cdef int count = 0
    cdef equal_to[int32_t] eq
    cdef maximum[uint32_t] _maximum

    if values.dtype.type == np.uint32:
        count = (<int32_t *>reduce_by_key(keys_first, keys_last,
                                          <uint32_t *>&values.data[0],
                                          keys_first,
                                          <uint32_t *>&values.data[0],
                                          eq, _maximum)
                 .first - keys_first)
        return count

    raise ValueError('Unsupported data type: %s' % (values.dtype.type))
