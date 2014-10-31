#ifndef ___UNIFORM_RANDOM_GENERATOR__HPP___
#define ___UNIFORM_RANDOM_GENERATOR__HPP___

#include <limits>
#include <iostream>
#include <stdint.h>

#ifndef __host__
#define CUDA_HOST_DEVICE
#else
#define CUDA_HOST_DEVICE  __host__ __device__
#endif  // #ifndef __host__

template <class T, typename Float=float>
class UniformRandomGeneratorBase {
protected:
    /* Portable random number generator defined below.  Taken from ANSI C by  *
    * K & R.  Not a great generator, but fast, and good enough for my needs. */

    static const T IA = 1103515245u;
    static const T IC = 12345u;
    static const T IM = 2147483648u;

    T current_random;
public:
    typedef Float float_type;
    typedef T type;

  CUDA_HOST_DEVICE
    UniformRandomGeneratorBase() : current_random(0) {}
    CUDA_HOST_DEVICE
    UniformRandomGeneratorBase(T seed) : current_random(seed) {}

    CUDA_HOST_DEVICE
    T current_value() const {
        return current_random;
    }

    CUDA_HOST_DEVICE
    void seed(T seed) {
        current_random = (T)seed;
    }

    CUDA_HOST_DEVICE
    T irand() {
        return irand(std::numeric_limits<T>::max());
    }

    CUDA_HOST_DEVICE
    T irand(T imax) {
        /* Creates a random integer between 0 and imax, inclusive.
         * _i.e. `[0..imax]`_ */
        T ival;

        current_random = current_random * IA + IC;	/* Use overflow to wrap */
        ival = current_random & (IM - 1);	/* Modulus */
        ival = (T)((float_type)ival * (float_type)(imax + 0.999) /
                   (float_type)IM);

        return (ival);
    }

    CUDA_HOST_DEVICE
    float_type frand(void) {
        /* Creates a random `float_type` between 0 and 1.  i.e. [0..1).        */
        float_type fval;
        T ival;

        current_random = current_random * IA + IC;	/* Use overflow to wrap */
        ival = current_random & (IM - 1);	/* Modulus */
        fval = (float_type)ival / (float_type)IM;

#if 0
        using namespace std;

        cout << _("[UniformRandomGeneratorBase] frand=%.2f") % fval << endl;
#endif
        return (fval);
    }
};


template <class int_t, class real_t>
class ParkMillerRNGBase {
protected:
    static int_t const a = 16807;      //ie 7**5
    static int_t const m = 2147483647; //ie 2**31-1
    int_t seed;
    real_t const reciprocal_m;
    real_t const reciprocal_m_sub_1;
public:
    typedef real_t result_type;

    CUDA_HOST_DEVICE
    ParkMillerRNGBase() : seed(0),
        reciprocal_m(1.0 / m), reciprocal_m_sub_1(1.0 / (m - 1)) {}

    CUDA_HOST_DEVICE
    ParkMillerRNGBase(int_t seed) : seed(seed),
        reciprocal_m(1.0 / m), reciprocal_m_sub_1(1.0 / (m - 1)) {}

    CUDA_HOST_DEVICE
    void set_seed(int_t i_seed) {
        seed = i_seed;
    }

    CUDA_HOST_DEVICE
    int_t MOD(real_t value, real_t divisor, real_t divisor_inv) {
        return value - (int_t)floor(value * divisor_inv) * divisor;
    }

    CUDA_HOST_DEVICE
    int_t current_value() {
        return seed;
    }

    CUDA_HOST_DEVICE
    int_t get_value() {
        real_t temp = seed * a;
        seed = MOD(temp, m, reciprocal_m);
        return seed;
    }

    CUDA_HOST_DEVICE
    int_t rand_int(int_t max_value) {
        return get_value() % (max_value + 1);
    }

    CUDA_HOST_DEVICE
    real_t rand_real() {
        return get_value() * reciprocal_m_sub_1;
    }

    CUDA_HOST_DEVICE
    result_type operator() (int_t i_seed) {
      set_seed(i_seed);
      /* Prime the generator by ignoring the first value, otherwise the value
       * returned is ~0. */
      get_value();
      return get_value() / (m - 1.);
    }
};


template <typename IntT, typename RealT>
struct SimpleRNG {
protected:
    static const IntT a = 16807;      //ie 7**5
    static const IntT m = 2147483647; //ie 2**31-1
public:
    typedef RealT result_type;

    CUDA_HOST_DEVICE
    result_type operator() (IntT seed) {
      RealT temp;

      /* Prime the generator by ignoring the first value, otherwise the value
       * returned is ~0. */
      temp = seed * a;
      seed = static_cast<IntT>(temp) % m;

      temp = seed * a;
      seed = static_cast<IntT>(temp) % m;

      return seed / (m - 1.);
    }
};


#endif
