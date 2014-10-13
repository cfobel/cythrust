#ifndef ___CYTHRUST__PARTITION__H___
#define ___CYTHRUST__PARTITION__H___

#include <thrust/partition.h>

namespace thrust {

template<typename ForwardIterator, typename Predicate>
int counted_partition(ForwardIterator first, ForwardIterator last,
                      Predicate pred) {
  return static_cast<int>((partition(first, last, pred) - first));
}


template<typename ForwardIterator, typename InputIterator, typename Predicate>
int counted_partition(ForwardIterator first, ForwardIterator last,
                      InputIterator stencil, Predicate pred) {
  return static_cast<int>((partition(first, last, stencil, pred) - first));
}


template<typename ForwardIterator, typename Predicate>
int counted_stable_partition(ForwardIterator first, ForwardIterator last,
                      Predicate pred) {
  return static_cast<int>((stable_partition(first, last, pred) - first));
}


template<typename ForwardIterator, typename InputIterator, typename Predicate>
int counted_stable_partition(ForwardIterator first, ForwardIterator last,
                      InputIterator stencil, Predicate pred) {
  return static_cast<int>((stable_partition(first, last, stencil, pred) -
                          first));
}


}

#endif  // #ifndef ___CYTHRUST__PARTITION__H___
