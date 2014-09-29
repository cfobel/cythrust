# distutils: language = c++
cimport cython
from cython.operator cimport dereference as deref
from libc.stdint cimport (uint32_t, int32_t, int64_t, uint64_t, INT32_MIN,
                          INT64_MIN, INT32_MAX, UINT32_MAX, INT64_MAX,
                          UINT64_MAX)
import numpy as np
cimport numpy as np
from cythrust.thrust.reduce cimport (accumulate as _accumulate,
                                     reduce as _reduce)
from cythrust.thrust.functional cimport (plus, multiplies, minus, divides,
                                         minimum, maximum)


cdef union Iterator:
    int32_t *int32
    uint32_t *uint32
    int64_t *int64
    uint64_t *uint64
    float *float32
    double *float64


def accumulate(np.ndarray a):
    if a.ndim > 1:
        raise ValueError('Only single dimension arrays are supported.')

    cdef Iterator first
    cdef Iterator last

    if a.dtype.type == np.int32:
        first.int32 = <int32_t *>a.data
        last.int32 = first.int32 + <size_t>a.size
        return <int32_t>_accumulate(first.int32, last.int32)
    elif a.dtype.type == np.uint32:
        first.uint32 = <uint32_t *>a.data
        last.uint32 = first.uint32 + <size_t>a.size
        return <uint32_t>_accumulate(first.uint32, last.uint32)
    elif a.dtype.type == np.int64:
        first.int64 = <int64_t *>a.data
        last.int64 = first.int64 + <size_t>a.size
        return <int64_t>_accumulate(first.int64, last.int64)
    elif a.dtype.type == np.uint64:
        first.uint64 = <uint64_t *>a.data
        last.uint64 = first.uint64 + <size_t>a.size
        return <uint64_t>_accumulate(first.uint64, last.uint64)
    elif a.dtype.type == np.float32:
        first.float32 = <float *>a.data
        last.float32 = first.float32 + <size_t>a.size
        return <float>_accumulate(first.float32, last.float32)
    elif a.dtype.type == np.float64:
        first.float64 = <double *>a.data
        last.float64 = first.float64 + <size_t>a.size
        return <double>_accumulate(first.float64, last.float64)

    raise ValueError('Unsupported data type: %s' % (a.dtype.type))


cdef enum BinaryOpCode:
    PLUS, MINUS, MULTIPLIES, DIVIDES, MODULUS, NEGATE, EQUAL_TO, NOT_EQUAL_TO,
    GREATER, LESS, GREATER_EQUAL, LESS_EQUAL, LOGICAL_AND, LOGICAL_OR,
    LOGICAL_NOT, BIT_AND, BIT_OR, BIT_XOR, MINIMUM, MAXIMUM


@cython.internal
cdef class _BinaryOps:
    cdef:
        readonly int PLUS
        readonly int MINUS
        readonly int MULTIPLIES
        readonly int DIVIDES
        readonly int MODULUS
        readonly int NEGATE
        readonly int EQUAL_TO
        readonly int NOT_EQUAL_TO
        readonly int GREATER
        readonly int LESS
        readonly int GREATER_EQUAL
        readonly int LESS_EQUAL
        readonly int LOGICAL_AND
        readonly int LOGICAL_OR
        readonly int LOGICAL_NOT
        readonly int BIT_AND
        readonly int BIT_OR
        readonly int BIT_XOR
        readonly int MINIMUM
        readonly int MAXIMUM

    def __cinit__(self):
        self.PLUS = PLUS
        self.MINUS = MINUS
        self.MULTIPLIES = MULTIPLIES
        self.DIVIDES = DIVIDES
        self.MODULUS = MODULUS
        self.NEGATE = NEGATE
        self.EQUAL_TO = EQUAL_TO
        self.NOT_EQUAL_TO = NOT_EQUAL_TO
        self.GREATER = GREATER
        self.LESS = LESS
        self.GREATER_EQUAL = GREATER_EQUAL
        self.LESS_EQUAL = LESS_EQUAL
        self.LOGICAL_AND = LOGICAL_AND
        self.LOGICAL_OR = LOGICAL_OR
        self.LOGICAL_NOT = LOGICAL_NOT
        self.BIT_AND = BIT_AND
        self.BIT_OR = BIT_OR
        self.BIT_XOR = BIT_XOR
        self.MINIMUM = MINIMUM
        self.MAXIMUM = MAXIMUM


BINARY_OPS = _BinaryOps()
BINARY_OP_NAME_BY_TYPE = {BINARY_OPS.PLUS: 'PLUS',
                          BINARY_OPS.MINUS: 'MINUS',
                          BINARY_OPS.MULTIPLIES: 'MULTIPLIES',
                          BINARY_OPS.DIVIDES: 'DIVIDES',
                          BINARY_OPS.MODULUS: 'MODULUS',
                          BINARY_OPS.NEGATE: 'NEGATE',
                          BINARY_OPS.EQUAL_TO: 'EQUAL_TO',
                          BINARY_OPS.NOT_EQUAL_TO: 'NOT_EQUAL_TO',
                          BINARY_OPS.GREATER: 'GREATER',
                          BINARY_OPS.LESS: 'LESS',
                          BINARY_OPS.GREATER_EQUAL: 'GREATER_EQUAL',
                          BINARY_OPS.LESS_EQUAL: 'LESS_EQUAL',
                          BINARY_OPS.LOGICAL_AND: 'LOGICAL_AND',
                          BINARY_OPS.LOGICAL_OR: 'LOGICAL_OR',
                          BINARY_OPS.LOGICAL_NOT: 'LOGICAL_NOT',
                          BINARY_OPS.BIT_AND: 'BIT_AND',
                          BINARY_OPS.BIT_OR: 'BIT_OR',
                          BINARY_OPS.BIT_XOR: 'BIT_XOR',
                          BINARY_OPS.MINIMUM: 'MINIMUM',
                          BINARY_OPS.MAXIMUM: 'MAXIMUM'}


cdef union Plus:
    plus[int32_t] int32
    plus[uint32_t] uint32
    plus[int64_t] int64
    plus[uint64_t] uint64
    plus[float] float32
    plus[double] float64


cdef union Multiplies:
    multiplies[int32_t] int32
    multiplies[uint32_t] uint32
    multiplies[int64_t] int64
    multiplies[uint64_t] uint64
    multiplies[float] float32
    multiplies[double] float64


cdef union Minus:
    minus[int32_t] int32
    minus[uint32_t] uint32
    minus[int64_t] int64
    minus[uint64_t] uint64
    minus[float] float32
    minus[double] float64


cdef union Minimum:
    minimum[int32_t] int32
    minimum[uint32_t] uint32
    minimum[int64_t] int64
    minimum[uint64_t] uint64
    minimum[float] float32
    minimum[double] float64


cdef union Maximum:
    maximum[int32_t] int32
    maximum[uint32_t] uint32
    maximum[int64_t] int64
    maximum[uint64_t] uint64
    maximum[float] float32
    maximum[double] float64


cdef union BinaryOp:
    Plus plus_
    Multiplies multiplies_
    Minus minus_
    Minimum minimum_
    Maximum maximum_


def reduce(np.ndarray a, BinaryOpCode op, init_value=None):
    if a.ndim > 1:
        raise ValueError('Only single dimension arrays are supported.')

    cdef Iterator first
    cdef Iterator last
    cdef BinaryOp operation

    if a.dtype.type == np.int32:
        first.int32 = <int32_t *>a.data
        last.int32 = first.int32 + <size_t>a.size

        if op == BINARY_OPS.PLUS:
            if init_value is None:
                init_value = 0
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.plus_.int32)
        elif op == BINARY_OPS.MULTIPLIES:
            if init_value is None:
                init_value = 1
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.multiplies_.int32)
        elif op == BINARY_OPS.MINUS:
            if init_value is None:
                init_value = 0
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.minus_.int32)
        elif op == BINARY_OPS.MINIMUM:
            if init_value is None:
                init_value = INT32_MAX
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.minimum_.int32)
        elif op == BINARY_OPS.MAXIMUM:
            if init_value is None:
                init_value = INT32_MIN
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.maximum_.int32)
    elif a.dtype.type == np.uint32:
        first.uint32 = <uint32_t *>a.data
        last.uint32 = first.uint32 + <size_t>a.size

        if op == BINARY_OPS.PLUS:
            if init_value is None:
                init_value = 0
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.plus_.uint32)
        elif op == BINARY_OPS.MULTIPLIES:
            if init_value is None:
                init_value = 1
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.multiplies_.uint32)
        elif op == BINARY_OPS.MINUS:
            if init_value is None:
                raise ValueError('Initial value must be provided for unsigned '
                                 'minus reductions.')
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.minus_.uint32)
        elif op == BINARY_OPS.MINIMUM:
            if init_value is None:
                init_value = UINT32_MAX
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.minimum_.uint32)
        elif op == BINARY_OPS.MAXIMUM:
            if init_value is None:
                init_value = 0
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.maximum_.uint32)

    raise ValueError('Unsupported data type: %s' % (a.dtype.type))


def reduce(np.ndarray a, BinaryOpCode op, init_value=None):
    if a.ndim > 1:
        raise ValueError('Only single dimension arrays are supported.')

    cdef Iterator first
    cdef Iterator last
    cdef BinaryOp operation

    if a.dtype.type == np.int32:
        first.int32 = <int32_t *>a.data
        last.int32 = first.int32 + <size_t>a.size

        if op == BINARY_OPS.PLUS:
            if init_value is None:
                init_value = 0
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.plus_.int32)
        elif op == BINARY_OPS.MULTIPLIES:
            if init_value is None:
                init_value = 1
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.multiplies_.int32)
        elif op == BINARY_OPS.MINUS:
            if init_value is None:
                init_value = 0
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.minus_.int32)
        elif op == BINARY_OPS.MINIMUM:
            if init_value is None:
                init_value = INT32_MAX
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.minimum_.int32)
        elif op == BINARY_OPS.MAXIMUM:
            if init_value is None:
                init_value = INT32_MIN
            return <int32_t>_reduce(first.int32, last.int32, <int32_t>init_value,
                                    operation.maximum_.int32)
    elif a.dtype.type == np.uint32:
        first.uint32 = <uint32_t *>a.data
        last.uint32 = first.uint32 + <size_t>a.size

        if op == BINARY_OPS.PLUS:
            if init_value is None:
                init_value = 0
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.plus_.uint32)
        elif op == BINARY_OPS.MULTIPLIES:
            if init_value is None:
                init_value = 1
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.multiplies_.uint32)
        elif op == BINARY_OPS.MINUS:
            if init_value is None:
                raise ValueError('Initial value must be provided for unsigned '
                                 'minus reductions.')
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.minus_.uint32)
        elif op == BINARY_OPS.MINIMUM:
            if init_value is None:
                init_value = UINT32_MAX
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.minimum_.uint32)
        elif op == BINARY_OPS.MAXIMUM:
            if init_value is None:
                init_value = 0
            return <uint32_t>_reduce(first.uint32, last.uint32,
                                     <uint32_t>init_value,
                                     operation.maximum_.uint32)

    raise ValueError('Unsupported data type: %s' % (a.dtype.type))
