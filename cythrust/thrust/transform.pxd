cdef extern from "<thrust/transform.h>" namespace "thrust" nogil:
    # See Thrust [`transform` documentation][1].
    #
    # [1]: http://thrust.github.io/doc/transform_8h.html
    OutputIterator transform[InputIterator, OutputIterator, UnaryFunction] \
        (InputIterator first, InputIterator last, OutputIterator result,
         UnaryFunction op)

    # Pass each pair of elements from two "zipped" input iterators as arguments to
    # the provided binary function, and write the result to the corresponding
    # element of the output iterator.
    OutputIterator transform2 'thrust::transform' [InputIterator1,
                                                   InputIterator2,
                                                   OutputIterator,
                                                   BinaryFunction] \
        (InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
         OutputIterator result, BinaryFunction op)

    # Pass each pair of elements from two "zipped" input iterators as arguments to
    # the provided binary function, and write the result to the corresponding
    # element of the output iterator.
    #ForwardIterator transform_if[InputIterator, ForwardIterator, UnaryFunction,
                                 #Predicate] \
        #(InputIterator first, InputIterator last, ForwardIterator result,
         #UnaryFunction op, Predicate pred)
    #ForwardIterator transform_if[InputIterator1, InputIterator2,
                                 #ForwardIterator, UnaryFunction, Predicate] \
        #(InputIterator1 first, InputIterator1 last, InputIterator2 stencil,
         #ForwardIterator result, UnaryFunction op, Predicate pred)
    #ForwardIterator transform_if[InputIterator1, InputIterator2,
                                 #InputIterator3, ForwardIterator,
                                 #BinaryFunction, Predicate] \
        #(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
         #InputIterator3 stencil, ForwardIterator result,
         #BinaryFunction binary_op, Predicate pred)
