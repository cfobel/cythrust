# coding: utf-8
from collections import OrderedDict

import numpy as np
import pandas as pd
import cythrust.device_vector as dv


class DeviceDataFrame(object):
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
    def __init__(self, data=None):
        if isinstance(data, dict):
            assert(len(set([v.size for v in data.itervalues()])) == 1)
            self._data_dict = OrderedDict([(k, dv.from_array(v))
                                           for k, v in data.iteritems()])
        elif isinstance(data, pd.DataFrame):
            self._data_dict = OrderedDict([(c, dv.from_array(data[c]
                                                             .values))
                                           for c in data.columns])
        elif data is None:
            self._data_dict = OrderedDict()
        elif data is not None:
            raise ValueError('Unsupported input data type.')

        self._view_dict = OrderedDict([(k, dv.view_from_vector(v))
                                       for k, v in self._data_dict
                                       .iteritems()])

    def add(self, column_name, column_data=None, dtype=None):
        if dtype is None and column_data is None:
            raise ValueError('At least one of `column_data` or `dtype` must '
                             'be provided.')
        size = self._data_dict.values()[0].size
        if column_data is None:
            column_data = np.zeros(size, dtype=dtype)
        elif dtype is None:
            column_data = column_data.astype(dtype)
        self._data_dict[column_name] = dv.from_array(column_data)
        sample_view = self._view_dict.values()[0]
        self._view_dict[column_name] = dv.view_from_vector(
            self._data_dict[column_name], first_i=sample_view.first_i,
            last_i=sample_view.last_i)

    def drop(self, column_name):
        del self._view_dict[column_name]
        del self._data_dict[column_name]

    def base(self):
        result = DeviceDataFrame()
        result._data_dict = self._data_dict
        result._view_dict = OrderedDict([(k, dv.view_from_vector(v))
                                         for k, v in result._data_dict
                                         .iteritems()])
        return result

    @property
    def d(self):
        return self._data_dict

    @property
    def v(self):
        return self._view_dict

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

    def __getitem__(self, key):
        # TODO: Return view?
        return self.as_dataframe()[key]

    @property
    def size(self):
        return self._view_dict.values()[0].size

    def view(self, start, end=None):
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
        print start, end
        view._view_dict = OrderedDict([(k, dv.view_from_vector(v, start,
                                                               end - 1))
                                        for k, v in
                                        self._data_dict.iteritems()])
        return view

    def views(self):
        return self._view_dict.values()
