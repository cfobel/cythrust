cdef extern from "<thrust/replace.h>" namespace "thrust" nogil:
    void replace[ForwardIterator, T](ForwardIterator first,
                                     ForwardIterator last, T old_value,
                                     T new_value)

    #void replace_if [ForwardIterator, Predicate, T](ForwardIterator first,
                                                    #ForwardIterator last,
                                                    #Predicate pred,
                                                    #T new_value)

    void replace_if_w_stencil 'thrust::replace_if' [ForwardIterator,
                                                    InputIterator, Predicate,
                                                    T] \
        (ForwardIterator first, ForwardIterator last, InputIterator stencil,
         Predicate pred, T new_value)

    OutputIterator replace_copy [InputIterator, OutputIterator, T] \
        (InputIterator first, InputIterator last, OutputIterator result,
         T old_value, T new_value)

    OutputIterator replace_copy_if [InputIterator, OutputIterator, Predicate, T] \
        (InputIterator first, InputIterator last, OutputIterator result,
         Predicate pred, T new_value)

    #OutputIterator replace_copy_if [InputIterator1, InputIterator2, OutputIterator,
                                    #Predicate, T] \
        #(InputIterator1 first, InputIterator1 last, InputIterator2 stencil,
         #OutputIterator result, Predicate pred, T new_value)
