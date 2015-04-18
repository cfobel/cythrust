# coding: utf-8
import sys
import hashlib
from collections import OrderedDict, Container
from ConfigParser import NoOptionError, NoSectionError

import pkg_resources
import numpy as np
from Cybuild import Context as _Context, NvccBuilder
from path_helpers import path
import jinja2
import functools32
import theano.tensor as T
try:
    import pandas as pd
except:
    pass
from .template import (SORT_TEMPLATE, BASE_TEMPLATE, REDUCE_BY_KEY_TEMPLATE,
                       COUNT_TEMPLATE, TRANSFORM_SETUP_TEMPLATE,
                       TRANSFORM_TEMPLATE, SCATTER_SETUP_TEMPLATE,
                       SCATTER_TEMPLATE, REDUCE_SETUP_TEMPLATE,
                       REDUCE_TEMPLATE)


NP_TYPE_TO_CTYPE = OrderedDict([('int8', 'int8_t'),
                                ('uint8', 'uint8_t'),
                                ('int16', 'int16_t'),
                                ('uint16', 'uint16_t'),
                                ('int32', 'int32_t'),
                                ('uint32', 'uint32_t'),
                                ('int64', 'int64_t'),
                                ('uint64', 'uint64_t'),
                                ('float32', 'float'),
                                ('float', 'double'),
                                ('float64', 'double')])

PANDAS_TO_THRUST = {'sum': 'plus', 'product': 'multiplies',
                    'min': 'minimum', 'max': 'maximum'}

NAMED_POSITIONS = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth',
                   'seventh', 'eighth', 'ninth']

def type_info(transform):
    dtype = transform.dfg.operation_graph.owner.out.dtype
    try:
        type_info = np.iinfo(dtype)
    except ValueError:
        type_info = np.finfo(dtype)
    return type_info

TRANSFORM_IDENTITIES = {'plus': 0, 'multiplies': 1,
                        'minimum': lambda t: type_info(t).max,
                        'maximum': lambda t: type_info(t).min}


class Functor(object):
    def __init__(self, code, func):
        self.code = code
        self.func = func
        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def get_includes():
    return [pkg_resources.resource_filename('cythrust', '')]


class Transform(object):
    """
    Generate the following sources files defining a Thrust transform iterator
    for the specified operation graph:

     - C++ header containing transform iterator class with `begin` method to
       return the starting iterator of the transform output.
     - Cython header *(i.e., `.pxd`)* containing definition of class defined in
       C++ header.  This definition provides a convenient way to use the C++
       transform class in Cython code.
     - Python and Cython `__init__` files to make the specified `output_dir` a
       module.

    __NB__ The `get_includes` method returns the path of the module where the
    C++ `.hpp` file resides.  This makes it easy to include the header path
    when compiling Cython code that uses the transform class.

    Example
    -------

    >>> from cythrust import DeviceDataFrame
    >>> import pandas as pd
    >>> import theano.tensor as T
    >>> import theano
    >>>
    >>> N = 10
    >>> host_df = pd.DataFrame({'a': np.arange(N, dtype='uint8'),
    ...                         'b': np.arange(N, dtype='uint8')},
    ...                        columns=list('ab'))
    >>> df = DeviceDataFrame(host_df)
    >>> a, b = df.tensor(['a', 'b'])
    >>>
    >>> # `build_transform` automatically adds the `Foo` module parent directory
    >>> # to the Python path so we can import from it.
    >>> foo_transform = df._context.build_transform(2 * a, 'Foo')
    >>> foo = df.inline_func(df.columns,
    ...     setup='''
    ... from Foo.Foo cimport Foo
    ... from cythrust.thrust.copy cimport copy_n''',
    ...     include_dirs=foo_transform.get_includes(),
    ...     code = '''
    ...     cdef Foo *op
    ...     op = new Foo(<Foo.a_t>a._begin)
    ...     cdef size_t N = a._end - a._begin
    ...     copy_n(op.begin(), N, b._begin)
    ...     return N''') # doctest:+ELLIPSIS
    ...
    >>> print foo(df['a'], df['b'])
    10
    >>> print foo_transform.__doc__
    (TensorConstant{2} * a)
    >>> print df.df[:].T
       0  1  2  3  4   5   6   7   8   9
    a  0  1  2  3  4   5   6   7   8   9
    b  0  2  4  6  8  10  12  14  16  18
    """
    def __init__(self, operation_graph, functor_name, output_dir):
        import theano
        from theano_helpers import DataFlowGraph, ThrustCode

        self.dfg = DataFlowGraph(operation_graph)
        self.thrust_code = ThrustCode(self.dfg)

        code = self.thrust_code.header_code(functor_name)
        hash_label = hashlib.sha1(code).hexdigest()[:12]

        output_dir.makedirs_p()

        py_init_path = output_dir.joinpath('__init__.py')
        cy_init_path = output_dir.joinpath('__init__.pxd')
        header_path = output_dir.joinpath('%s.hpp' % hash_label)
        cyheader_path = output_dir.joinpath('%s.pxd' % functor_name)

        with header_path.open('wb') as output:
            output.write(code)
        with cyheader_path.open('wb') as output:
            output.write(self.thrust_code.cython_header_code(
                functor_name, '"%s.hpp"' % hash_label))
        py_init_path.touch()
        cy_init_path.touch()

        self.__doc__ = str(theano.pp(operation_graph))
        self.output_dir = path(output_dir).expand().abspath()
        self.functor_name = functor_name

        if not self.output_dir.parent in sys.path:
            sys.path.insert(0, str(self.output_dir.parent))

    def get_includes(self):
        return [self.output_dir]


class Context(_Context):
    '''
    Sub-class `Cybuild` context to build dynamic Cython extensions with
    `thrust`-specific settings.
    '''
    template_path = path(pkg_resources
                         .resource_filename('cythrust',
                                            'template'))
    LIB_PATH = path('~/.cache/cythrust/lib').expand()
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))

    def __init__(self, device_system='THRUST_DEVICE_SYSTEM_CPP', tag='',
                 include_dirs=None, **kwargs):
        '''
        Parameters
        ----------

        `device_system` : `str`, optional
         * A Thrust [device backend][1].

        `tag` : `str`, optional
         * Optional tag string to differentiate between variants using the
           same Thrust device backend.
         * __NB__, Compile arguments may be passed through the remaining `kwargs`.

        [1]: https://github.com/thrust/thrust/wiki/Device-Backends
        '''
        if device_system == 'THRUST_DEVICE_SYSTEM_CUDA':
            if 'builder' not in kwargs:
                kwargs['builder'] = NvccBuilder()
        super(Context, self).__init__(**kwargs)
        self.tag = tag
        self.device_system = device_system
        if include_dirs is None:
            include_dirs = self.include_dirs
        self.kwargs = {'pyx_kwargs': {'cplus': True},
                       'include_dirs': include_dirs + get_includes()}
        if self.device_system in ('THRUST_DEVICE_SYSTEM_CPP',
                                  'THRUST_DEVICE_SYSTEM_TBB',
                                  'THRUST_DEVICE_SYSTEM_OMP'):
            self.kwargs.update({'define_macros': [('THRUST_DEVICE_SYSTEM',
                                                   device_system)],
                                'extra_compile_args': ['-O3']})
        elif self.device_system == 'THRUST_DEVICE_SYSTEM_CUDA':
            self.kwargs['preargs'] = ['-O3', '-D' + self.device_system]
        if self.device_system == 'THRUST_DEVICE_SYSTEM_TBB':
            self.kwargs['extra_link_args'] = ['-ltbb']
        elif self.device_system == 'THRUST_DEVICE_SYSTEM_OMP':
            self.kwargs['extra_compile_args'] += ['-fopenmp']
            self.kwargs['extra_link_args'] = ['-fopenmp']

    @property
    def include_dirs(self):
        try:
            value = self.config.get('paths', 'include_dirs')
            if isinstance(value, str):
                value = eval(value)
            return [str(path(d).expand()) for d in value]
        except NoSectionError:
            self.config.add_section('paths')
        except NoOptionError:
            pass
        self.include_dirs = []
        return self.include_dirs

    @include_dirs.setter
    def include_dirs(self, value):
        self.config.set('paths', 'include_dirs', value)

    def build_pyx(self, *args, **kwargs):
        _kwargs = self.kwargs.copy()
        _kwargs.update(kwargs)

        return super(Context, self).build_pyx(*args, **_kwargs)

    def build_transform(self, operation_graph, transform_name,
                        module_name=None):
        '''
         - Generate C++ header file with Thrust transform iterator definition
           which implements the specified operation graph.
         - Generate Cython header file *(i.e., `.pxd`)* which includes
           transform class definition.
         - Add files to a directory with the specified `module_name` within the
           current context library path root.  This should place the module on
           the Python path.
        '''
        if module_name is None:
            module_name = transform_name
        output_dir = self.LIB_PATH.joinpath(module_name)
        transform = Transform(operation_graph, transform_name, output_dir)
        return transform

    def inline_pyx_module(self, *args, **kwargs):
        _kwargs = self.kwargs.copy()
        _kwargs.update(kwargs)

        return super(Context, self).inline_pyx_module(*args, **_kwargs)

    def from_array(self, array, dtype=None):
        if dtype is None:
            dtype = array.dtype
        vector_class = self.get_device_vector_class(dtype)
        return vector_class.from_array(array)

    def clear_cache(self):
        self.get_device_vector_class.cache_clear()
        self.get_device_vector_module.cache_clear()
        self.get_device_vector_view_class.cache_clear()

    @functools32.lru_cache()
    def get_device_vector_class(self, dtype):
        vector_module_name =  self.get_device_vector_module(dtype)
        exec('from %s import DeviceVector as _vector_class'
             % vector_module_name)
        return _vector_class

    @functools32.lru_cache()
    def get_device_vector_view_class(self, dtype):
        vector_module_name =  self.get_device_vector_module(dtype)
        exec('from %s import DeviceVectorView as _view_class'
             % vector_module_name)
        return _view_class

    @functools32.lru_cache()
    def get_device_vector_module(self, dtype):
        if isinstance(dtype, np.dtype):
            dtype = dtype.type
        np_dtype = 'np.' + dtype.__name__
        c_dtype = NP_TYPE_TO_CTYPE[dtype.__name__]
        system_key = self.device_system.split('_')[-1].lower()
        if self.tag:
            system_key += '_%s' % self.tag
        full_module_name = '.'.join(['device_vector', system_key,
                                     np_dtype[3:]])
        try:
            exec('import %s' % full_module_name)
            return full_module_name
        except ImportError:
            pass

        pyx_dir = self.LIB_PATH.joinpath('device_vector',
                                         system_key, np_dtype[3:])
        pyx_dir.makedirs_p()

        for d in (pyx_dir.parent.parent.parent,
                  pyx_dir.parent.parent, pyx_dir.parent, pyx_dir):
            d.joinpath('__init__.pxd').write_bytes('')
            d.joinpath('__init__.py').write_bytes('')

        pyx_template_path = self.template_path.joinpath(
            'device_vector.pyxt')
        pyx_source_path = pyx_dir.joinpath(pyx_template_path.name[:-1])

        with pyx_source_path.open('wb') as output:
            template = jinja2.Template(pyx_template_path.bytes())
            output.write(template.render(C_DTYPE=c_dtype,
                                         NP_DTYPE=np_dtype))

        pxd_template_path = self.template_path.joinpath(
            'device_vector.pxdt')
        pxd_source_path = pyx_dir.joinpath(pxd_template_path.name[:-1])

        with pxd_source_path.open('wb') as output:
            template = jinja2.Template(pxd_template_path.bytes())
            output.write(template.render(C_DTYPE=c_dtype,
                                         NP_DTYPE=np_dtype))

        pyx_dir.joinpath('__init__.py').write_bytes(
            'from device_vector import *')

        module_dir, module_name = self.build_pyx(pyx_source_path,
                                                 module_dir=pyx_dir)
        return full_module_name


class DeviceViewGroup(object):
    '''
    Base class to group together references to device vector views that belong
    to a common `Context`.  This is useful, for example, to create `Transform`
    instances using views from different columns of `DeviceDataFrame` and
    `DeviceVectorCollection` instances.

    This class also acts as a mixin for the `DeviceDataFrame` and
    `DeviceVectorCollection` classes, to provide shared functionality.

    Note that a `DeviceViewGroup` does *not necessarily* own the underlying
    device vectors.
    '''
    TRANSFORM_CACHE = {}

    @classmethod
    def from_device_vectors(self, device_vectors):
        group = DeviceViewGroup()
        group._context = device_vectors._context
        group._view_dict = device_vectors.v.copy()
        if len(set([v.size for v in group._view_dict.itervalues()])) > 1:
            # Device vector views are of different sizes.
            group._jagged = True
        else:
            # All vector views are of the same length.
            group._jagged = True
        return group

    @property
    def jagged(self):
        return self._jagged

    @property
    def df(self):
        return self.as_dataframe()

    def as_dataframe(self):
        return pd.DataFrame(OrderedDict([(k, v[:])
                                         for k, v in self._view_dict
                                         .iteritems()]),
                            index=range(*self.index_bounds()))

    def reset_views(self):
        for column in self.columns:
            self.v[column].first_i = 0
            self.v[column].last_i = self.d[column].size - 1

    def index_bounds(self):
        sample_view = self._view_dict.values()[0]
        first_i = sample_view.first_i
        last_i = sample_view.last_i
        return first_i, last_i + 1

    def inline_func(self, columns, code='', setup='', context=None,
                    verbose=False, include_dirs=None, **kwargs):
        '''
        Return a dynamically compiled Cython function, based on the
        `BASE_TEMPLATE` template.

        Argument types are implied by the data-type of each column, where each
        column is represented by the corresponding `DeviceVectorView` type.

        Arguments
        ---------

         - `columns` : `list`-like
          * Column names to pass as arguments to generated function.
         - `code` : `str` *(optional)*
          * Code to insert in function body.  May contain Jinja
            template language.
         - `setup` : `str` *(optional)*
          * Set-up code to insert at start of template.  May contain Jinja
            template language.
         - `context` : `dict` *(optional)*
          * Context to add to Jinja template context.
         - `verbose` : `bool` *(optional)*
          * If `True`, generated code will be printed.
         - `kwargs` : `dict` *(optional)*
          * Additional keyword arguments to pass along to `inline_pyx_module`.
        '''
        template = jinja2.Template('\n'.join([setup, BASE_TEMPLATE, code]))

        if context is None:
            context = {}
        all_code = template.render(
            df=self, module_names=self.get_vector_module(columns),
            dtypes=self.get_dtype(columns), view_names=columns,
            **context)
        if include_dirs is None:
            include_dirs = self._context.kwargs['include_dirs']
        else:
            include_dirs += self._context.kwargs['include_dirs']
        module_path, module_name = self._context.inline_pyx_module(
            all_code, include_dirs=include_dirs, **kwargs)
        if verbose:
            print all_code
        exec('import %s' % module_name)
        return Functor(all_code, eval('%s.__foo__' % module_name))

    def __getitem__(self, key):
        '''
        Return the columns corresponding to the provided key(s) as a
        `DeviceViewGroup`.
        '''
        group = DeviceViewGroup()
        group._context = self._context
        views = self._get_scalar_or_list(key, lambda name, column: (name,
                                                                    column),
                                         include_name=True)
        if isinstance(views, tuple):
            views = [views]
        group._view_dict = OrderedDict(views)
        return group

    @property
    def columns(self):
        return self._view_dict.keys()

    @property
    def v(self):
        return self._view_dict

    def _get_scalar_or_list(self, column, func, include_name=False):
        if not isinstance(column, str) and isinstance(column, Container):
            if include_name:
                return [func(c, self.v[c]) for c in column]
            else:
                return [func(self.v[c]) for c in column]
        else:
            if include_name:
                return func(column, self.v[column])
            else:
                return func(self.v[column])

    def get_vector_class(self, column):
        return self._get_scalar_or_list(column,
                                        lambda x:
                                        self._context
                                        .get_device_vector_class(x.dtype))

    def get_vector_view_class(self, column):
        return self._get_scalar_or_list(column,
                                        lambda x:
                                        self._context
                                        .get_device_vector_view_class(x.dtype))

    def get_vector_module(self, column):
        return self._get_scalar_or_list(column,
                                        lambda x:
                                        self._context
                                        .get_device_vector_module(x.dtype))

    def tensor(self, column):
        import theano.tensor as T

        return self._get_scalar_or_list(column, lambda column, view:
                                        T.vector(column,
                                                 dtype=view.dtype.__name__),
                                        include_name=True)

    def get_dtype(self, column):
        return self._get_scalar_or_list(column, lambda x: x.dtype)

    def get_ctype(self, column):
        return self._get_scalar_or_list(column, lambda x: x.ctype)

    @functools32.lru_cache()
    def get_sort_func(self, key_columns, value_columns=None, stable=False):
        '''
        Dynamically compile a Thrust Cython sort function based on the types of
        the specified key and value columns for sorting.

        __NB,__ Results of this function are cached to improve runtime
        performance of repeated calls for the same columns/column types.
        '''
        if value_columns is None:
            value_columns = []
        key_modules = tuple(self.get_vector_module(key_columns))
        value_modules = tuple(self.get_vector_module(value_columns))
        key_dtypes = tuple(self.get_dtype(key_columns))
        value_dtypes = tuple(self.get_dtype(value_columns))

        return get_sort_func(self._context, key_modules, key_dtypes,
                             value_modules, value_dtypes, stable=stable)

    def sort(self, column=None, key=None, stable=False):
        '''
        Sort values in specified column(s) by the values specified key
        column(s).

        If no column or key is specified, the rows across all columns will be
        sorted, based on the order of the columns as they appear in the
        `columns` attribute.

        If no `key` columns are specified, the values in the `column` column(s)
        are sorted while leaving all other columns *untouched*.
        '''
        if column is None:
            if key is not None:
                raise ValueError('If column is not specified, `key` must not '
                                 'be specified, either.')
            columns = tuple(self.columns)
        elif isinstance(column, str):
            columns = (column, )
        else:
            columns = tuple(column)

        if key is None:
            # If no columns were specified to sort by, interpret `column` as
            # key columns to sort by.
            key = columns
            columns = tuple()
        elif isinstance(key, str):
            key = (key, )
        else:
            key = tuple(key)

        sort_func = self.get_sort_func(key_columns=key, value_columns=columns,
                                       stable=stable)
        sort_func(*self[key + columns])

    def as_arrays(self):
        return OrderedDict([(k, v[:]) for k, v in self._view_dict.iteritems()])

    def views(self):
        return self._view_dict.values()

    @property
    def sizes(self):
        return OrderedDict([(k, v.size)
                            for k, v in self._view_dict.iteritems()])

    @property
    def size(self):
        sizes = self.sizes.values()
        assert(min(sizes) == max(sizes))
        return sizes[0]

    @property
    def vector_sizes(self):
        return OrderedDict([(k, d.size)
                            for k, d in self._data_dict.iteritems()])

    def groupby(self, key_columns, sort=True):
        return GroupBy(self, key_columns)

    def transform(self, transform_dict, out=None):
        '''
        Return result from applying specified transforms.

        Use `TRANSFORM_CACHE` to reuse previously compiled functions for the
        same dictionary transforms.
        '''
        out, group, foo = self.get_transform_function(transform_dict, out)
        foo(*group._view_dict.values())
        return out

    def get_transform_function(self, transform_dict, out):
        transform_tuple = tuple(transform_dict.items())

        if out is None or transform_tuple not in self.TRANSFORM_CACHE:
            transforms = OrderedDict([
                (k, self._context.build_transform(v, '%s%s' % (k, hash(v))))
                for k, v in transform_dict.iteritems()])

        if out is None:
            out = DeviceDataFrame(OrderedDict([
                (k, np.zeros(self.size, dtype=t.thrust_code.output_nodes
                             .sort_index().iloc[-1].dtype))
                for k, t in transforms.iteritems()]),
                context=self._context)

        group = join(self, out)

        if transform_tuple in self.TRANSFORM_CACHE:
            foo = self.TRANSFORM_CACHE[transform_tuple]
        else:
            setup = (jinja2.Template(TRANSFORM_SETUP_TEMPLATE)
                    .render(transforms=transforms.values(),
                            out_views=transforms.keys()))
            code = (jinja2.Template(TRANSFORM_TEMPLATE)
                    .render(transforms=transforms.values(),
                            out_views=transforms.keys()))

            try:
                foo = group.inline_func(group._view_dict.keys(),
                    setup=setup,
                    include_dirs=
                    np.concatenate([t.get_includes()
                                    for t in transforms.values()]).tolist(),
                    code=code)
            except:
                print 50 * '='
                print setup
                print 50 * '-'
                print code
                print 50 * '='
                raise
            self.TRANSFORM_CACHE[transform_tuple] = foo
        return out, group, foo

    def scatter(self, out_size, in_operations, out_operations):
        '''
        For each sequence defined in list of `in_operations`, scatter values to
        respective sequence defined in `out_operations`.

        Arguments
        ---------

         - `out_size`: Number of values to scatter.  This is required, since it
           is simpler than inferring from sequences.
           * __Warning__: An incorrect value for `out_size` may lead to a
             segmentation fault.
         - `in_operations`: List of (or single) theano operations, which may
           involve column tensors of the view group.
         - `out_operations`: List of (or single) theano operations, where each
           operation represents the output sequence for the respective sequence
           in `in_operations`.
         -

        Examples
        --------

        The following examples are valid, assuming columns `c` and `d` are the
        same length.  Note that

            c, d = cd.tensor(['c', 'd'])

            ## Working ##
            cd.scatter(cd.size, T.arange(11, ddf_d.size + 11), d.take(T.arange(0)))  # d[:] = np.arange(11, d.size + 11)
            cd.scatter(cd.size, ddf_d.size - 1 - T.arange(ddf_d.size), d.take(T.arange(0)))  # d[:] = np.arange(ddf_d.size - 1, 0, -1)
            cd.scatter(cd.size, c - c + 4.3, d.take(T.arange(0)))  # d[:] = 4.3
            cd.scatter(cd.size, c.take(T.arange(0)), d.take(T.arange(0)))  # d[:] = c[:]


        TODO
        ----

        The following examples currently result in compilation errors:

            ## Compile error ##
            cd.scatter(cd.size, c, d)  # d[:] = c[:]
            cd.scatter(cd.size, 11, d)  # d[:] = 11
            cd.scatter(cd.size, T.constant(11), d)  # d[:] = 11

        Fix `theano_helpers` to support the above examples.

         - Add support for single column tensors as valid operations.
         - Add support for constant values as valid operations.
        '''
        try:
            assert(len(in_operations) > 0)
            assert(len(in_operations) == len(out_operations))
        except TypeError:
            # At least one of `in_operations`, `out_operations` is not a
            # iterable. Assume a single operator was passed.
            in_operations = [in_operations]
            out_operations = [out_operations]
        transforms_in = [self._context.build_transform(t, 'scatter_in%s' %
                                                       hash(t))
                         for t in in_operations]
        transforms_out = [self._context.build_transform(t, 'scatter_out%s' %
                                                        hash(t))
                          for t in out_operations]

        setup = (jinja2.Template(SCATTER_SETUP_TEMPLATE)
                 .render(transforms_in=transforms_in,
                         transforms_out=transforms_out))
        code = (jinja2.Template(SCATTER_TEMPLATE)
                .render(transforms_in=transforms_in,
                        transforms_out=transforms_out))

        include_dirs = np.concatenate([t.get_includes()
                                       for t in transforms_in +
                                       transforms_out]).tolist()

        scatter_func = self.inline_func(self.columns,
                                        include_dirs=include_dirs, setup=setup,
                                        code=code,
                                        context=dict(preargs='uint32_t N, '))
        scatter_func(out_size, *self.v.values())
        return self

    def min(self, **kwargs):
        return self._reduce_wrapper('minimum', **kwargs)

    def max(self, **kwargs):
        return self._reduce_wrapper('maximum', **kwargs)

    def sum(self, **kwargs):
        return self._reduce_wrapper('plus', **kwargs)

    def product(self, **kwargs):
        return self._reduce_wrapper('multiplies', **kwargs)

    def _reduce_wrapper(self, reduce_op, **kwargs):
        result = self.reduce(reduce_op, **kwargs)
        if kwargs.get('operations', None) is None:
            return pd.Series(result, index=self.columns)
        else:
            return result

    def get_transforms(self, columns=None, operations=None):
        if operations is None:
            if columns is None:
                columns = self.columns
            tensors = self.tensor(columns)

            # Create identity operation for each column, which iterate over the
            # values of the corresponding column.
            # TODO: Fix support for bare column tensors.  This may require
            # changes to `theano_helpers`.
            operations = [t.take(T.arange((1 << 32) - 1)) for t in tensors]
        return [self._context.build_transform(t, 'reduce%s' % hash(t))
                for t in operations]

    def reduce(self, reduce_ops=None, operations=None, transforms=None,
               init_values=None, size=None, **kwargs):
        if transforms is None:
            transforms = self.get_transforms(operations=operations)
        elif isinstance(transforms, list) and not transforms:
            transforms.extend(self.get_transforms(operations=operations))

        if isinstance(reduce_ops, str):
            reduce_ops = [reduce_ops] * len(transforms)
        elif reduce_ops is None:
            reduce_ops = ['plus'] * len(transforms)

        if init_values is None:
            init_values = [TRANSFORM_IDENTITIES[k]
                           if not callable(TRANSFORM_IDENTITIES[k])
                           else TRANSFORM_IDENTITIES[k](transforms[i])
                           for i, k in enumerate(reduce_ops)]
        if size is None:
            size = self.size

        # Cast arguments as tuples since cached `get_reduce_func` function
        # requires *hashable* arguments.
        reduce_func = self.get_reduce_func(tuple(reduce_ops),
                                           tuple(transforms),
                                           tuple(init_values), **kwargs)

        return reduce_func(size, *self.v.values())

    @functools32.lru_cache()
    def get_reduce_func(self, reduce_ops, transforms, init_values, **kwargs):
        '''
        __NB__ `reduce_ops`, `transforms`, and `init_values` must all be
        *hashable* types.  This is a requirement for using
        `functools32.lru_cache`.
        '''
        setup = (jinja2.Template(REDUCE_SETUP_TEMPLATE)
                .render(transforms=transforms,
                        reduce_ops=reduce_ops,
                        init_values=init_values,
                        named_positions=NAMED_POSITIONS))

        code = (jinja2.Template(REDUCE_TEMPLATE)
                .render(transforms=transforms,
                        reduce_ops=reduce_ops,
                        init_values=init_values,
                        named_positions=NAMED_POSITIONS))

        include_dirs = np.concatenate([t.get_includes()
                                       for t in transforms]).tolist()

        return self.inline_func(self.columns, include_dirs=include_dirs,
                                setup=setup, code=code,
                                context=dict(preargs='size_t N, '), **kwargs)


class DeviceVectorCollection(DeviceViewGroup):
    '''
    A container of `cythrust.device_vector.DeviceVector` instances.
    Can be initialized by:
        - A dictionary-like container of `numpy.ndarray` instances.
        - A `pandas.DataFrame`.

    The types of the device vector instances will be inferred based on the
    types of the input arrays.


    Known issues
    ------------

     - Only numeric types are accepted.
    '''
    def __init__(self, data=None, context=None):
        if context is None:
            context = Context()
        self._context = context
        if isinstance(data, dict):
            self._data_dict = OrderedDict([(k, self._context.from_array(v))
                                           for k, v in data.iteritems()])
        elif isinstance(data, pd.DataFrame):
            self._data_dict = OrderedDict([(c, self._context
                                            .from_array(data[c].values))
                                           for c in data.columns])
        elif data is None:
            self._data_dict = OrderedDict()
        elif data is not None:
            raise ValueError('Unsupported input data type.')

        self._view_dict = OrderedDict([(k, v.view())
                                       for k, v in self._data_dict
                                       .iteritems()])

    def reorder(self, column_names):
        if set(column_names).intersection(self.columns) != set(self.columns):
            raise ValueError('All column names must be provided in reordered '
                             'list.')
        if len(set(column_names)) != len(column_names):
            raise ValueError('Each column name must appear _exactly once_ in '
                             'the reordered  list.')
        self._data_dict = OrderedDict([(k, self._data_dict[k])
                                       for k in column_names])
        self._view_dict = OrderedDict([(k, self._view_dict[k])
                                       for k in column_names])

    def add(self, column_name, column_data, dtype=None):
        if dtype is None:
            column_data = column_data.astype(dtype)
        self._data_dict[column_name] = self._context.from_array(column_data)
        self._view_dict[column_name] = self._data_dict[column_name].view()

    def drop(self, column_name):
        del self._view_dict[column_name]
        del self._data_dict[column_name]

    def base(self):
        result = DeviceVectorCollection()
        result._data_dict = self._data_dict
        result._view_dict = OrderedDict([(k, v.view())
                                         for k, v in result._data_dict
                                         .iteritems()])
        return result

    @property
    def d(self):
        return self._data_dict


class DeviceDataFrame(DeviceVectorCollection):
    '''
    A container of `cythrust.device_vector.DeviceVector` instances.
    Can be initialized by:
        - A dictionary-like container of `numpy.ndarray` instances.
        - A `pandas.DataFrame`.

    The types of the device vector instances will be inferred based on the
    types of the input arrays.


    Known issues
    ------------

     - Only numeric types are accepted.
    '''
    def __init__(self, data=None, context=None):
        if isinstance(data, dict):
            assert(len(set([v.size for v in data.itervalues()])) == 1)
        super(DeviceDataFrame, self).__init__(data, context)

    def copy(self):
        return DeviceDataFrame(self.as_arrays(), context=self._context)

    def add(self, column_name, column_data=None, dtype=None):
        if dtype is None and column_data is None:
            raise ValueError('At least one of `column_data` or `dtype` must '
                             'be provided.')
        if len(self._data_dict):
            size = self._data_dict.values()[0].size
        else:
            size = None
        if column_data is None:
            if size is None:
                raise ValueError('Size cannot be inferred when the data frame '
                                 'does not have any existing columns.')
            column_data = np.zeros(size, dtype=dtype)
        elif dtype is not None:
            column_data = column_data.astype(dtype)
        self._data_dict[column_name] = self._context.from_array(column_data)
        if len(self._view_dict):
            sample_view = self._view_dict.values()[0]
            start = sample_view.first_i
            end = sample_view.last_i
        else:
            start = 0
            end = self._data_dict.values()[0].size - 1
        self._view_dict[column_name] = self._data_dict[column_name].view(
            first_i=start, last_i=end)

    def __setitem__(self, key_or_slice, value):
        if isinstance(value, pd.DataFrame):
            missing_columns = (set(value.columns)
                               .difference(self._view_dict.keys()))
            if missing_columns:
                raise KeyError('The following columns are not '
                               'present in the device frame: %s' %
                               ', '.join(missing_columns))

        # A slice was provided, so we expect to either:
        #
        #  - Broadcasting a single value, or
        #  - Setting several rows with values from a `pandas.DataFrame`.
        if isinstance(value, pd.DataFrame):
            for column in value.columns:
                self._view_dict[column][key_or_slice] = value[column]
        else:
            for v in self._view_dict.itervalues():
                v[key_or_slice] = value

    def _in_bounds(self, i):
        lbound, ubound = self.index_bounds()
        return (i >= lbound and i < ubound and lbound < ubound)

    @property
    def size(self):
        sizes = self.sizes
        min_size = min(sizes.values())
        max_size = max(sizes.values())
        assert(min_size == max_size)
        return max_size

    @property
    def vector_size(self):
        sizes = self.vector_sizes
        min_size = min(sizes.values())
        max_size = max(sizes.values())
        assert(min_size == max_size)
        return max_size

    def view(self, start, end=None):
        '''
        Return a view corresponding to the specified slice of the views
        dictionary items.

        *N.B.,* No data is copied.
        '''
        if end is None:
            start = 0
            end = start
        if start < 0:
            start = self.size + start
        if end < 0:
            end = self.size + end + 1
        elif end > self.size:
            end = self.size
        view = DeviceDataFrame()
        view._data_dict = self._data_dict
        if len(self._view_dict):
            sample_view = self._view_dict.values()[0]
            start += sample_view.first_i
            end += sample_view.first_i
            end = min(sample_view.last_i + 1, end)
        view._view_dict = OrderedDict([(k, v.view(start, end - 1))
                                        for k, v in
                                        self._data_dict.iteritems()])
        return view

    def base(self):
        result = DeviceDataFrame()
        result._data_dict = self._data_dict
        result._view_dict = OrderedDict([(k, v.view())
                                         for k, v in result._data_dict
                                         .iteritems()])
        return result


class GroupBy(object):
    def __init__(self, views, key_columns, sort=True, stable=False):
        self.views = views
        self.key_views = views[key_columns]
        self.value_views = views[[c for c in self.views.columns
                                  if c not in key_columns]]
        key_columns = self.key_views.columns
        value_columns = self.value_views.columns
        key_modules = tuple(self.key_views.get_vector_module(key_columns))
        key_dtypes = tuple(self.key_views.get_dtype(key_columns))
        value_modules = tuple(self.value_views.get_vector_module(value_columns))
        value_dtypes = tuple(self.value_views.get_dtype(value_columns))

        self.sort_func = get_sort_func(views._context, key_modules, key_dtypes,
                                       value_modules, value_dtypes, stable=stable)

        if sort:
            self.sort()

    def sort(self):
        self.sort_func(*(self.key_views.v.values() + self.value_views.v.values()))

    def get_count_func(self, out):
        key_columns = self.key_views.columns
        key_modules = tuple(self.key_views.get_vector_module(key_columns))
        key_dtypes = tuple(self.key_views.get_dtype(key_columns))
        key_ctypes = tuple(self.key_views.get_ctype(key_columns))
        count_modules = (out.get_vector_module('count'), )
        count_dtypes = (out.get_dtype('count'), )
        count_ctypes = (out.get_ctype('count'), )

        return get_count_func(self.views._context, key_modules, key_dtypes,
                              count_modules, count_dtypes, key_ctypes,
                              count_ctypes)

    def get_reduce_func(self, reduce_ops, value_columns=None):
        # TODO: Optionally accept `out` kwarg to infer iterator and module
        # types from output device data frame.
        key_columns = self.key_views.columns
        if value_columns is None:
            value_columns = self.value_views.columns
        key_modules = tuple(self.key_views.get_vector_module(key_columns))
        key_dtypes = tuple(self.key_views.get_dtype(key_columns))
        key_ctypes = tuple(self.key_views.get_ctype(key_columns))
        value_modules = tuple(self.value_views.get_vector_module(value_columns))
        value_dtypes = tuple(self.value_views.get_dtype(value_columns))
        value_ctypes = tuple(self.value_views.get_ctype(value_columns))
        assert(len(reduce_ops) == len(value_columns))

        return get_reduce_func(self.views._context, key_modules, key_dtypes,
                               value_modules, value_dtypes, key_ctypes,
                               value_ctypes, tuple(reduce_ops))

    def ref_agg(self, reduce_op):
        '''
        Perform reduction using `pandas.DataFrame.agg`.
        '''
        ref_result = (self.views.df.groupby(self.key_views.columns)
                      .agg(reduce_op))
        # When applying multiple reduce operations, `pandas` creates a
        # multi-level index for result.  To match output from `GroupBy.agg`,
        # flatten multi-level index in result by appending the first level as a
        # prefix in the form `<value column name>_<operation name>`.
        if not isinstance(reduce_op, str):
            ref_result.columns = ['_'.join(col).strip()
                                  for col in ref_result.columns.values]
        return ref_result.reset_index()

    def agg(self, reduce_op, out=None, bounds_check=True):
        '''
        Perform reduction using `cythrust.thrust.reduce_by_key`.
        '''
        if isinstance(reduce_op, str):
            value_columns = self.value_views.v.keys()
            reduce_ops = [reduce_op, ] * len(value_columns)
            reduce_op = (reduce_op, )
        else:
            value_columns = [column
                             for column in self.value_views.v.keys()
                             for op in reduce_op]
            reduce_ops = np.tile(reduce_op, len(self.value_views.v)).tolist()

        # __NB__ Key columns and value columns are all unique
        # (i.e., a key column will not have the same name as
        # any value column).
        if out is None:
            out_vectors = OrderedDict(
                [(column, np.zeros(view.size, dtype=view.dtype))
                 for column, view in self.key_views.v.iteritems()] +
                [('%s_%s' % (column, op),
                  np.zeros(view.size, dtype=view.dtype))
                 for column, view in self.value_views.v.iteritems()
                 for op in reduce_op])
            out = DeviceDataFrame(out_vectors, context=self.views._context)
        elif bounds_check:
            # Check to make sure that the views are large enough.
            if hasattr(out, 'vector_size'):
                assert(out.vector_size >= self.views.size)
            else:
                for view in out.v.itervalues():
                    assert(view.size >= self.views.size)

        func = self.get_reduce_func([PANDAS_TO_THRUST[op]
                                     for op in reduce_ops],
                                    value_columns=value_columns)
        in_views = self.key_views.v.values() + [self.value_views.v[c]
                                                for c in value_columns]
        out_views = out.v.values()

        reduced_key_count = func(*(in_views + out_views))

        for view in out.v.itervalues():
            view.last_i = reduced_key_count - 1
        return out

    def ref_count(self):
        '''
        Perform count using `pandas.DataFrame.agg`.
        '''
        first_col = self.value_views.columns[0]
        ref_out = (self.views.df.groupby(self.key_views.columns)
                   .agg({first_col: 'count'})
                   .rename(columns={first_col: 'count'}))
        return ref_out.reset_index()

    def count(self, out=None, bounds_check=True):
        '''
        Perform count by key using `cythrust.thrust.reduce_by_key`.
        '''
        if out is None:
            out_vectors = OrderedDict(
                [(column, np.zeros(view.size, dtype=view.dtype))
                 for column, view in self.key_views.v.iteritems()] +
                [('count', np.zeros(self.key_views.size, dtype='uint32'))])
            out = DeviceDataFrame(out_vectors, context=self.views._context)
        elif bounds_check:
            # Check to make sure that the views are large enough.
            if hasattr(out, 'vector_size'):
                assert(out.vector_size >= self.views.size)
            else:
                for view in out.v.itervalues():
                    assert(view.size >= self.views.size)

        func = self.get_count_func(out)
        in_views = self.key_views.v.values()
        out_views = out.v.values()

        reduced_key_count = func(*(in_views + out_views))

        for view in out.v.itervalues():
            view.last_i = reduced_key_count - 1
        return out


@functools32.lru_cache()
def get_sort_func(context, key_modules, key_dtypes, value_modules=None,
                  value_dtypes=None, stable=False):
    if value_modules is None or value_dtypes is None:
        value_modules = []
        value_dtypes = []

    template = jinja2.Template(SORT_TEMPLATE)
    code = template.render(key_modules=key_modules, key_dtypes=key_dtypes,
                           value_modules=value_modules,
                           value_dtypes=value_dtypes, stable=stable)
    try:
        module_path, module_name = context.inline_pyx_module(code)
    except:
        print code
        raise
    exec('from %s import sort_func as __sort_func__' % module_name)
    return __sort_func__


@functools32.lru_cache()
def get_reduce_func(context, key_modules, key_dtypes, value_modules,
                    value_dtypes, key_ctypes, value_ctypes,
                    reduce_ops, key_out_modules=None, key_out_dtypes=None,
                    value_out_modules=None, value_out_dtypes=None):
    # By default, use input key/value types for output keys/values.
    if key_out_modules is None:
        key_out_modules = key_modules
    if key_out_dtypes is None:
        key_out_dtypes = key_dtypes
    if value_out_modules is None:
        value_out_modules = value_modules
    if value_out_dtypes is None:
        value_out_dtypes = value_dtypes

    template = jinja2.Template(REDUCE_BY_KEY_TEMPLATE)
    code = template.render(key_modules=key_modules,
                           key_dtypes=key_dtypes,
                           value_modules=value_modules,
                           value_dtypes=value_dtypes,
                           key_ctypes=key_ctypes,
                           value_ctypes=value_ctypes,
                           reduce_ops=reduce_ops,
                           key_out_modules=key_out_modules,
                           key_out_dtypes=key_out_dtypes,
                           value_out_modules=value_out_modules,
                           value_out_dtypes=value_out_dtypes)
    try:
        module_path, module_name = context.inline_pyx_module(code)
    except:
        print code
        raise
    exec('from %s import reduce_by_key_func as __reduce_func__' % module_name)
    return __reduce_func__


@functools32.lru_cache()
def get_count_func(context, key_modules, key_dtypes, value_out_modules,
                   value_out_dtypes, key_ctypes, value_out_ctypes,
                   key_out_modules=None, key_out_dtypes=None):
    # By default, use input key/value types for output keys/values.
    if key_out_modules is None:
        key_out_modules = key_modules
    if key_out_dtypes is None:
        key_out_dtypes = key_dtypes

    template = jinja2.Template(COUNT_TEMPLATE)
    code = template.render(key_modules=key_modules,
                           key_dtypes=key_dtypes,
                           key_ctypes=key_ctypes,
                           value_out_modules=value_out_modules,
                           value_out_dtypes=value_out_dtypes,
                           value_out_ctypes=value_out_ctypes,
                           key_out_modules=key_out_modules,
                           key_out_dtypes=key_out_dtypes)
    try:
        module_path, module_name = context.inline_pyx_module(code)
    except:
        print code
        raise
    exec('from %s import count_by_key_func as __count_func__' % module_name)
    return __count_func__


def join(left_group, right_group):
    group = DeviceViewGroup()
    assert(left_group._context == right_group._context)
    group._context = left_group._context
    group._view_dict = OrderedDict(left_group.v.items() +
                                   right_group.v.items())
    return group
