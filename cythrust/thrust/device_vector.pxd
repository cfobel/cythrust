cdef extern from "<thrust/device_vector.h>" namespace "thrust" nogil:
    cdef cppclass device_vector[T]:
        cppclass value_type:
            pass
        cppclass iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_t)
            iterator operator-(size_t)
            int operator-(iterator)
            bint operator==(iterator)
            bint operator!=(iterator)
            bint operator<(iterator)
            bint operator>(iterator)
            bint operator<=(iterator)
            bint operator>=(iterator)
        cppclass reverse_iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_t)
            iterator operator-(size_t)
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator>=(reverse_iterator)
        #cppclass const_iterator(iterator):
        #    pass
        #cppclass const_reverse_iterator(reverse_iterator):
        #    pass
        device_vector() except +
        device_vector(device_vector&) except +
        device_vector(size_t) except +
        device_vector(size_t, T&) except +
        #device_vector[input_iterator](input_iterator, input_iterator)
        T& operator[](size_t)
        ##vector& operator=(vector&)
        #bint operator==(vector&, vector&)
        #bint operator!=(vector&, vector&)
        #bint operator<(vector&, vector&)
        #bint operator>(vector&, vector&)
        #bint operator<=(vector&, vector&)
        #bint operator>=(vector&, vector&)
        #void assign(size_t, T&)
        #void assign[input_iterator](input_iterator, input_iterator)
        #T& at(size_t)
        #T& back()
        iterator begin()
        ##const_iterator begin()
        #size_t capacity()
        #void clear()
        #bint empty()
        iterator end()
        ##const_iterator end()
        #iterator erase(iterator)
        #iterator erase(iterator, iterator)
        #T& front()
        #iterator insert(iterator, T&)
        #void insert(iterator, size_t, T&)
        #void insert(iterator, iterator, iterator)
        size_t max_size()
        #void pop_back()
        #void push_back(T&)
        #reverse_iterator rbegin()
        ##const_reverse_iterator rbegin()
        #reverse_iterator rend()
        ##const_reverse_iterator rend()
        #void reserve(size_t)
        void resize(size_t)
        #void resize(size_t, T&)
        size_t size()
        #void swap(vector&)

        ##C++0x methods
        #T* data()
        #void shrink_to_fit()
