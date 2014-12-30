# coding: utf-8
from collections import OrderedDict, Container
import sys
from ConfigParser import NoOptionError, NoSectionError

import pkg_resources
import numpy as np
from Cybuild import Context as _Context, NvccBuilder
from path_helpers import path
import jinja2
import functools32
try:
    import pandas as pd
except:
    pass
from .template import SORT_TEMPLATE


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


def get_includes():
    return [pkg_resources.resource_filename('cythrust', '')]


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
                                  'THRUST_DEVICE_SYSTEM_TBB'):
            self.kwargs.update({'define_macros': [('THRUST_DEVICE_SYSTEM',
                                                   device_system)],
                                'extra_compile_args': ['-O3']})
        elif self.device_system == 'THRUST_DEVICE_SYSTEM_CUDA':
            self.kwargs['preargs'] = ['-O3', '-D' + self.device_system]
        if self.device_system == 'THRUST_DEVICE_SYSTEM_TBB':
            self.kwargs['extra_link_args'] = ['-ltbb']

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


class DeviceVectorCollection(object):
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

    def as_arrays(self):
        return OrderedDict([(k, v[:]) for k, v in self._view_dict.iteritems()])

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

    def get_dtype(self, column):
        return self._get_scalar_or_list(column, lambda x: x.dtype)

    def get_ctype(self, column):
        return self._get_scalar_or_list(column, lambda x: x.ctype)

    def _get_scalar_or_list(self, column, func):
        if not isinstance(column, str) and isinstance(column, Container):
            return [func(self.v[c]) for c in column]
        else:
            return func(self.v[column])

    @property
    def d(self):
        return self._data_dict

    @property
    def v(self):
        return self._view_dict

    def __getitem__(self, key):
        return self._get_scalar_or_list(key, lambda c: c)

    @property
    def columns(self):
        return self._view_dict.keys()

    @property
    def size(self):
        return OrderedDict([(k, v.size)
                            for k, v in self._view_dict.iteritems()])


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

    def as_dataframe(self):
        return pd.DataFrame(OrderedDict([(k, v[:])
                                         for k, v in self._view_dict
                                         .iteritems()]),
                            index=range(*self.index_bounds()))

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

    def index_bounds(self):
        sample_view = self._view_dict.values()[0]
        first_i = sample_view.first_i
        last_i = sample_view.last_i
        return first_i, last_i + 1

    def _in_bounds(self, i):
        lbound, ubound = self.index_bounds()
        return (i >= lbound and i < ubound and lbound < ubound)

    @property
    def df(self):
        return self.as_dataframe()

    @property
    def size(self):
        return self._view_dict.values()[0].size

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

    def views(self):
        return self._view_dict.values()

    def base(self):
        result = DeviceDataFrame()
        result._data_dict = self._data_dict
        result._view_dict = OrderedDict([(k, v.view())
                                         for k, v in result._data_dict
                                         .iteritems()])
        return result

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
