cdef extern from "src/UniformRandomGenerator.hpp" nogil:
    cdef cppclass UniformRandomGeneratorBase [T, Float]:
        UniformRandomGeneratorBase()
        UniformRandomGeneratorBase(T seed)

    cdef cppclass ParkMillerRNGBase [T, Float]:
        ParkMillerRNGBase ()
        ParkMillerRNGBase (T seed)

    cdef cppclass SimpleRNG [IntT, FloatT]:
        pass
