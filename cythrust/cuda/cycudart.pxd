cdef extern from "<cuda_runtime_api.h>" nogil:
    enum cudaError:
        cudaSuccess
        cudaErrorAddressOfConstant
        cudaErrorApiFailureBase
        cudaErrorCudartUnloading
        cudaErrorInitializationError
        cudaErrorInvalidChannelDescriptor
        cudaErrorInvalidConfiguration
        cudaErrorInvalidDevice
        cudaErrorInvalidDeviceFunction
        cudaErrorInvalidDevicePointer
        cudaErrorInvalidFilterSetting
        cudaErrorInvalidHostPointer
        cudaErrorInvalidMemcpyDirection
        cudaErrorInvalidNormSetting
        cudaErrorInvalidPitchValue
        cudaErrorInvalidResourceHandle
        cudaErrorInvalidSymbol
        cudaErrorInvalidTexture
        cudaErrorInvalidTextureBinding
        cudaErrorInvalidValue
        cudaErrorLaunchFailure
        cudaErrorLaunchOutOfResources
        cudaErrorLaunchTimeout
        cudaErrorMapBufferObjectFailed
        cudaErrorMemoryAllocation
        cudaErrorMemoryValueTooLarge
        cudaErrorMissingConfiguration
        cudaErrorMixedDeviceExecution
        cudaErrorNotReady
        cudaErrorNotYetImplemented
        cudaErrorPriorLaunchFailure
        cudaErrorStartupFailure
        cudaErrorSynchronizationError
        cudaErrorTextureFetchFailed
        cudaErrorTextureNotBound
        cudaErrorUnknown
        cudaErrorUnmapBufferObjectFailed

    cudaError cudaSetDevice(int)
    cudaError cudaFree(void *)


cdef extern from "<cuda.h>" nogil:
    enum cudaError_enum:
        CUDA_SUCCESS
        CUDA_ERROR_INVALID_VALUE
        CUDA_ERROR_INVALID_DEVICE

    enum CUresult:
        pass

    cdef cppclass CUdevice:
        pass

    cdef cppclass CUcontext:
        pass

    cudaError cuMemGetInfo(size_t *, size_t *)
    cudaError cuInit(int)
    cudaError cuDeviceGetCount(int *)
    cudaError cuDeviceGet(CUdevice *, int)
    cudaError cuCtxCreate(CUcontext *, int, CUdevice)
    cudaError cuCtxDetach(CUcontext)
