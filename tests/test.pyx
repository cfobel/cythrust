from cython.operator cimport dereference as deref, preincrement as inc

from cythrust.device_vector cimport device_vector
from cythrust.fill cimport fill, fill_n, uninitialized_fill
from cythrust.copy cimport copy, copy_n
from cythrust.sequence cimport sequence


def test():
    cdef device_vector[int] *v_ptr
    u_ptr = new device_vector[int](10)
    v_ptr = new device_vector[int](10)

    sequence(u_ptr.begin(), u_ptr.end())
    fill(v_ptr.begin(), v_ptr.end(), 1)
    copy(u_ptr.begin(), u_ptr.end(), v_ptr.begin())

    cdef int i

    for i in xrange(v_ptr.size()):
        print deref(v_ptr)[i]

    del v_ptr
    del u_ptr
