#ifndef ___CYTHRUST__UNIQUE__H___
#define ___CYTHRUST__UNIQUE__H___

#include <thrust/partition.h>

namespace thrust {


template<typename ForwardIterator>
int counted_unique(ForwardIterator first, ForwardIterator last) {
  return static_cast<int>((unique(first, last) - first));
}


template<typename ForwardIterator, typename Predicate>
int counted_unique(ForwardIterator first, ForwardIterator last,
                   Predicate pred) {
  return static_cast<int>((unique(first, last, pred) - first));
}


}

#endif  // #ifndef ___CYTHRUST__UNIQUE__H___
