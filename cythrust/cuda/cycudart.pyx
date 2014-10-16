# distutils: language = c++
# cython: embedsignature = True
from libc.stdint cimport uint64_t
from cython.operator import address as addr
from collections import OrderedDict


CUDA_ERROR = OrderedDict([
    (cudaSuccess, 'cudaSuccess'),
    (cudaErrorAddressOfConstant, 'cudaErrorAddressOfConstant'),
    (cudaErrorApiFailureBase, 'cudaErrorApiFailureBase'),
    (cudaErrorCudartUnloading, 'cudaErrorCudartUnloading'),
    (cudaErrorInitializationError, 'cudaErrorInitializationError'),
    (cudaErrorInvalidChannelDescriptor, 'cudaErrorInvalidChannelDescriptor'),
    (cudaErrorInvalidConfiguration, 'cudaErrorInvalidConfiguration'),
    (cudaErrorInvalidDevice, 'cudaErrorInvalidDevice'),
    (cudaErrorInvalidDeviceFunction, 'cudaErrorInvalidDeviceFunction'),
    (cudaErrorInvalidDevicePointer, 'cudaErrorInvalidDevicePointer'),
    (cudaErrorInvalidFilterSetting, 'cudaErrorInvalidFilterSetting'),
    (cudaErrorInvalidHostPointer, 'cudaErrorInvalidHostPointer'),
    (cudaErrorInvalidMemcpyDirection, 'cudaErrorInvalidMemcpyDirection'),
    (cudaErrorInvalidNormSetting, 'cudaErrorInvalidNormSetting'),
    (cudaErrorInvalidPitchValue, 'cudaErrorInvalidPitchValue'),
    (cudaErrorInvalidResourceHandle, 'cudaErrorInvalidResourceHandle'),
    (cudaErrorInvalidSymbol, 'cudaErrorInvalidSymbol'),
    (cudaErrorInvalidTexture, 'cudaErrorInvalidTexture'),
    (cudaErrorInvalidTextureBinding, 'cudaErrorInvalidTextureBinding'),
    (cudaErrorInvalidValue, 'cudaErrorInvalidValue'),
    (cudaErrorLaunchFailure, 'cudaErrorLaunchFailure'),
    (cudaErrorLaunchOutOfResources, 'cudaErrorLaunchOutOfResources'),
    (cudaErrorLaunchTimeout, 'cudaErrorLaunchTimeout'),
    (cudaErrorMapBufferObjectFailed, 'cudaErrorMapBufferObjectFailed'),
    (cudaErrorMemoryAllocation, 'cudaErrorMemoryAllocation'),
    (cudaErrorMemoryValueTooLarge, 'cudaErrorMemoryValueTooLarge'),
    (cudaErrorMissingConfiguration, 'cudaErrorMissingConfiguration'),
    (cudaErrorMixedDeviceExecution, 'cudaErrorMixedDeviceExecution'),
    (cudaErrorNotReady, 'cudaErrorNotReady'),
    (cudaErrorNotYetImplemented, 'cudaErrorNotYetImplemented'),
    (cudaErrorPriorLaunchFailure, 'cudaErrorPriorLaunchFailure'),
    (cudaErrorStartupFailure, 'cudaErrorStartupFailure'),
    (cudaErrorSynchronizationError, 'cudaErrorSynchronizationError'),
    (cudaErrorTextureFetchFailed, 'cudaErrorTextureFetchFailed'),
    (cudaErrorTextureNotBound, 'cudaErrorTextureNotBound'),
    (cudaErrorUnknown, 'cudaErrorUnknown'),
    (cudaErrorUnmapBufferObjectFailed, 'cudaErrorUnmapBufferObjectFailed')])


CU_ERROR = OrderedDict([
    (CUDA_SUCCESS, 'CUDA_SUCCESS'),
    (CUDA_ERROR_INVALID_VALUE, 'CUDA_ERROR_INVALID_VALUE'),
    (CUDA_ERROR_INVALID_DEVICE, 'CUDA_ERROR_INVALID_DEVICE')])


def set_device(int device_id):
    cdef int result = cudaSetDevice(device_id)
    if result != cudaSuccess:
        raise RuntimeError('cudaSetDevice failed with error code: %d' %
                           CUDA_ERROR[result])


def free(uint64_t address):
    cdef int result = cudaFree(<void *>address)
    if result != cudaSuccess:
        raise RuntimeError('cudaFree failed with error code: %d' %
                           CUDA_ERROR[result])


def mem_get_info():
    cdef size_t free_mem
    cdef size_t total_mem

    cdef int result = cuMemGetInfo(addr(free_mem), addr(total_mem))
    if result != cudaSuccess:
        raise RuntimeError('cuMemGetInfo failed with error code: %s' %
                           CU_ERROR[result])
    return free_mem, total_mem


cpdef cuda_init(int device_id):
    cdef int result = cuInit(device_id)
    if result != cudaSuccess:
        raise RuntimeError('cuInit failed with error code: %s' %
                           CU_ERROR[result])


cpdef get_device_count():
    cdef int device_count = 0
    cdef int result = cuDeviceGetCount(addr(device_count))
    if result != cudaSuccess:
        raise RuntimeError('cuDeviceGetCount failed with error code: %s' %
                           CU_ERROR[result])
    return device_count


cdef class Env:
    def __cinit__(self, int flags=0):
        cuda_init(flags)

    property device_count:
        def __get__(self):
            return get_device_count()


cdef class Context:
    cdef CUdevice dev
    cdef CUcontext ctx
    cdef int device_count

    def __cinit__(self, int device_id):
        cuDeviceGet(addr(self.dev), device_id)
        cuCtxCreate(addr(self.ctx), 0, self.dev)

    def mem_get_info(self):
        return mem_get_info()

    def __dealloc__(self):
        cuCtxDetach(self.ctx)
