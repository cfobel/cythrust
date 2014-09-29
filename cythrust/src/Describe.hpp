#ifndef ___DESCRIBE__HPP___
#define ___DESCRIBE__HPP___

#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include "src/si_prefix.hpp"


#ifdef __CUDACC__
static const double POSITIVE_INFINITY_REAL = std::numeric_limits<double>::max();
static const double NEGATIVE_INFINITY_REAL = std::numeric_limits<double>::min();
#else
static const double POSITIVE_INFINITY_REAL = 9e99;
static const double NEGATIVE_INFINITY_REAL = -9e99;
#endif  // #ifdef __CUDACC__


#ifdef __CUDACC__
__host__ __device__
#endif
inline double get_std_dev(int n, double sum_x_squared, double av_x) {
  /* # `get_std_dev` #
   *
   * From [VRP][1]:
   *
   * > Returns the standard deviation of data set x.  There are n sample
   * > points, sum_x_squared is the summation over n of x^2 and av_x is the
   * > average x. All operations are done in double precision, since round off
   * > error can be a problem in the initial temp. std_dev calculation for big
   * > circuits.
   *
   * [1]: https://code.google.com/p/vtr-verilog-to-routing/ */

  double std_dev;

  if(n <= 1) {
    std_dev = 0.;
  } else {
    std_dev = (sum_x_squared - n * av_x * av_x) / (double)(n - 1);
  }

  if(std_dev > 0.) {
    /* Very small variances sometimes round negative */
    std_dev = sqrt(std_dev);
  } else {
    std_dev = 0.;
  }

  return (std_dev);
}


template <typename Real=float>
struct DescribeSummary {
  Real sum;
  Real mean;
  Real std;
  Real min;
  Real max;
  Real count;
  Real non_zero_count;

  std::string str() const {
    std::stringstream s;
    s << "|Stat             | Value           |" << std::endl;
    s << "|-----------------|----------------:|" << std::endl;
    const size_t value_width = 17;
    s << "|      $\\sum{x}$  |" << std::setw(value_width) << sum << "|"
      << std::endl;
    s << "|      $\\mu_{x}$  |" << std::setw(value_width) << mean << "|"
      << std::endl;
    s << "|   $\\sigma_{x}$  |" << std::setw(value_width) << std << "|"
      << std::endl;
    s << "|            min  |" << std::setw(value_width) << min << "|"
      << std::endl;
    s << "|            max  |" << std::setw(value_width) << max << "|"
      << std::endl;
    s << "|          count  |" << std::setw(value_width) << count << "|"
      << std::endl;
    s << "| non-zero count  |" << std::setw(value_width) << non_zero_count
      << "|" << std::endl;
    return s.str();
  }

  void dump() const {
    std::cout << str();
  }

  std::string str_si(int precision=2) const {
    std::stringstream s;
    s << "|Stat             | Value           |" << std::endl;
    s << "|-----------------|----------------:|" << std::endl;
    const size_t value_width = 17;
    s << "|      $\\sum{x}$  |" << std::setw(value_width)
              << si_prefix::format(sum, precision)
              << "|" << std::endl;
    s << "|      $\\mu_{x}$  |" << std::setw(value_width)
              << si_prefix::format(mean, precision)
              << "|" << std::endl;
    s << "|   $\\sigma_{x}$  |" << std::setw(value_width)
              << si_prefix::format(std, precision)
              << "|" << std::endl;
    s << "|            min  |" << std::setw(value_width)
              << si_prefix::format(min, precision)
              << "|" << std::endl;
    s << "|            max  |" << std::setw(value_width)
              << si_prefix::format(max, precision)
              << "|" << std::endl;
    s << "|          count  |" << std::setw(value_width)
              << si_prefix::format(count, precision)
              << "|" << std::endl;
    s << "| non-zero count  |" << std::setw(value_width)
              << si_prefix::format(non_zero_count, precision)
              << "|" << std::endl;
    return s.str();
  }

  void dump_si(int precision=2) const {
    std::cout << str_si(precision);
  }
};


template <typename Input, typename Real=float>
struct describe_value
  : public thrust::unary_function<
      Input, thrust::tuple<Real,  /* $v$ */
                           Real,  /* $v^2$ */
                           Real,  /* $min{v}$ */
                           Real,  /* $max{v}$ */
                           Real, Real> > {
  typedef thrust::tuple<Real,  /* $v$ */
                        Real,  /* $v^2$ */
                        Real,  /* $min{v}$ */
                        Real,  /* $max{v}$ */
                        Real, Real> result_type;
  typedef DescribeSummary<Real> summary_type;

  __host__ __device__
  static summary_type summarize(result_type const &result) {
    Real _mean = thrust::get<0>(result) / thrust::get<4>(result);
    Real _std = get_std_dev(thrust::get<4>(result), thrust::get<1>(result),
                            _mean);
    summary_type summary = {thrust::get<0>(result), _mean, _std,
                            thrust::get<2>(result), thrust::get<3>(result),
                            thrust::get<4>(result), thrust::get<5>(result)};
    return summary;
  }

  __host__ __device__
  static result_type init() {
    return result_type(0, 0, POSITIVE_INFINITY_REAL, NEGATIVE_INFINITY_REAL,
                       0, 0);
  }

  __host__ __device__
  result_type operator() (Input v) {
    return result_type(v, v * v, v, v, 1, v != 0);
  }
};


template <typename T>
struct DescribeReduce : public thrust::binary_function<T, T, T> {
  __host__ __device__
  T operator() (T a, T b) {
    return T(
      thrust::get<0>(a) + thrust::get<0>(b),
      thrust::get<1>(a) + thrust::get<1>(b),
      ((thrust::get<2>(a) < thrust::get<2>(b)) ? thrust::get<2>(a) :
       thrust::get<2>(b)),
      ((thrust::get<3>(a) > thrust::get<3>(b)) ? thrust::get<3>(a) :
       thrust::get<3>(b)),
      thrust::get<4>(a) + thrust::get<4>(b),
      thrust::get<5>(a) + thrust::get<5>(b));
  }
};


template <typename ValueIterator, typename Real>
DescribeSummary<Real> describe(ValueIterator start, ValueIterator end) {
  typedef describe_value<Real, Real> DescribeValue;
  return DescribeValue::summarize(
    thrust::reduce(
      thrust::make_transform_iterator(start, DescribeValue()),
      thrust::make_transform_iterator(end, DescribeValue()),
      DescribeValue::init(),
      DescribeReduce<typename DescribeValue::result_type>()));
}


template <typename ValueIterator>
DescribeSummary<float> describe_f(ValueIterator start, ValueIterator end) {
  return describe<ValueIterator, float>(start, end);
}


template <typename ValueIterator>
DescribeSummary<double> describe_d(ValueIterator start, ValueIterator end) {
  return describe<ValueIterator, double>(start, end);
}


#endif  // #ifndef ___DESCRIBE__HPP___
