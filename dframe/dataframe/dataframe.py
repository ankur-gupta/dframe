from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from builtins import range

from prettytable import PrettyTable
from collections import OrderedDict
import numpy as np
import pandas as pd
import csv

from dframe.array import Array, to_best_dtype
from dframe.errors import InternalError
from dframe.compat import Iterable
from dframe.dtypes import is_integer, is_float, is_string, is_bool, infer_dtype
from dframe.scalar import (is_scalar, get_length, is_list_unique, is_list_same,
                           is_iterable_string, is_iterable_integer)


def _get_generic_names(ncol):
    return ['C{}'.format(i) for i in range(ncol)]


class DataFrame(object):
    _print_max_nrows = 60

    def __init__(self, data={}):
        if isinstance(data, dict):
            self._init_from_dict(data)
        elif isinstance(data, self.__class__):
            # This does not copy the underlying data a la Python and pandas.
            self._data = data._data
            self._names = data._names
            self._names_to_index = data._names_to_index
            self._ncol = data._ncol
            self._nrow = data._nrow
        else:
            # Default constructor only supports dict-like objects
            # Use alternate constructors for numpy/list/tuple, etc.
            msg = ('{} construction from {} is not supported; see '
                   'alternate constructors DataFrame.from_*')
            msg = msg.format(self.__class__.__name__, type(data))
            raise NotImplementedError(msg)

    @classmethod
    def from_dict(cls, data):
        assert isinstance(data, dict)
        return cls(data)

    @classmethod
    def from_items(cls, items):
        # Ensure names are unique
        if is_list_unique([name for name, _ in items]):
            return cls(OrderedDict(items))
        else:
            raise ValueError('duplicate column names found')

    @classmethod
    def from_rows(cls, list_of_rows, names=None):
        nrow = get_length(list_of_rows)
        ncol_per_row = [get_length(row) for row in list_of_rows]
        if is_list_same(ncol_per_row):
            if len(ncol_per_row) == 0:
                ncol = 0
            else:
                ncol = ncol_per_row[0]
            if names is None:
                names = _get_generic_names(ncol)
            if ncol == get_length(names):
                items = [
                    (names[j], [list_of_rows[i][j] for i in range(nrow)])
                    for j in range(ncol)]
                return cls.from_items(items)
            else:
                msg = 'number of names do not match the number of columns'
                raise ValueError(msg)
        else:
            msg = 'all rows do not have the same number of columns'
            raise ValueError(msg)

    @classmethod
    def from_columns(cls, list_of_columns, names=None):
        ncol = get_length(list_of_columns)
        if names is None:
            names = _get_generic_names(ncol)
        if ncol == get_length(names):
            items = [(names[j], list_of_columns[j]) for j in range(ncol)]
            return cls.from_items(items)
        else:
            msg = 'number of names do not match the number of columns'
            raise ValueError(msg)

    @classmethod
    def from_numpy(cls, array, names=None):
        assert isinstance(array, np.ndarray)
        if len(array.shape) == 2:
            columns = [array[:, j] for j in range(array.shape[1])]
            return cls.from_columns(columns, names)
        else:
            msg = 'numpy array dimensions must be exactly 2'
            raise ValueError(msg)

    @classmethod
    def from_pandas(cls, df):
        if isinstance(df, pd.Series):
            msg = ('pandas Series is not supported, please use pandas '
                   'DataFrame instead')
            raise ValueError(msg)
        else:
            assert isinstance(df, pd.DataFrame)
        items = [(name, df[name]) for name in df]
        return cls.from_items(items)

    @classmethod
    def from_shape(cls, shape, names=None):
        if get_length(shape) == 2:
            if is_iterable_integer(shape):
                if (shape[0] >= 0) and (shape[1] >= 0):
                    if names is None:
                        names = _get_generic_names(shape[1])
                    if shape[1] == get_length(names):
                        items = [(names[j], [None for i in range(shape[0])])
                                 for j in range(shape[1])]
                        return cls.from_items(items)
                    else:
                        msg = ('number of names do not match the number '
                               'of columns')
                        raise ValueError(msg)
                else:
                    msg = 'shape elements must be non-negative'
                    raise ValueError(msg)
            else:
                msg = 'shape elements must be integer'
                raise ValueError(msg)
        else:
            msg = 'shape must have exactly two elements'
            raise ValueError(msg)

    @classmethod
    def from_csv(cls, path, infer_dtypes=True, header=True, delimiter=','):
        # FIXME: Handle missing data!
        with open(path, 'r') as f:
            r = csv.reader(f, delimiter=delimiter)
            rows = [row for row in r]
        if header:
            df = cls.from_rows(rows[1:], rows[0])
        else:
            df = cls.from_rows(rows)
        if infer_dtypes:
            for j in range(df.ncol):
                df[j] = to_best_dtype(df[j])
        return df

    def _init_from_dict(self, data):
        scalarity_per_value = [is_scalar(value) for value in data.values()]
        if all(scalarity_per_value):
            # Box all scalar values
            _data = [Array(value) for value in data.values()]
        elif any(scalarity_per_value):
            # At least one value is scalar but all values are not scalars
            # Allocate a list and put non-scalar values inside.
            _data = [None] * len(data)
            length_per_value = [None] * len(data)
            for i, value in enumerate(data.values()):
                if not scalarity_per_value[i]:
                    _data[i] = Array(value)
                    length_per_value[i] = len(_data[i])

            # All non-scalar columns must have the same length or
            # we raise a ValueError
            length_non_scalars = set(
                [length for length, scalarity in
                 zip(length_per_value, scalarity_per_value)
                 if not scalarity])
            if length(length_non_scalars) > 1:
                msg = 'columns do not have the same length'
                raise ValueError(msg)
            elif len(length_non_scalars) == 0:
                msg = 'you found a bug, please report it'
                raise InternalError(msg)
            else:
                length = list(length_non_scalars)[0]
        else:
            # All values are non-scalars. No need to box them.
            _data = [Array(value) for value in data.values()]

        # Ensure dict keys are string types
        if not is_iterable_string(data.keys()):
            msg = 'non string names are not allowed'
            raise ValueError(msg)
        else:
            _names = data.keys()

        # Ensure all columns have the same length
        if not is_list_same([len(column) for column in _data]):
            msg = 'columns do not have the same lengths'
            raise ValueError(msg)

        # Set curated internal vars
        self._data = Array(_data)
        self._names = Array(_names)

        # Update all other fields
        self._update_nrow_ncol()
        self._update_names_to_index()

    def _update_nrow_ncol(self):
        column_lengths = set(map(len, self._data))
        if len(column_lengths) > 1:
            msg = ('found different column lengths which is a bug, '
                   'please report it')
            raise InternalError(msg)
        elif len(column_lengths) == 0:
            self._nrow = 0
        else:
            self._nrow = list(column_lengths)[0]
        self._ncol = len(self._data)

    def _update_names_to_index(self):
        self._names_to_index = {name: index
                                for index, name in enumerate(self._names)}

    def __len__(self):
        return self._ncol

    def __iter__(self):
        for name in self._names:
            yield name

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.ncol == 0:
            out = 'DataFrame with {} rows and {} columns'
            return out.format(self.nrow, self.ncol)
        else:
            headers = self.names
            headers.insert(0, '')
            table = PrettyTable(headers)
            if self.nrow <= self._print_max_nrows:
                for i, row in enumerate(self.rows()):
                    row.insert(0, i)
                    table.add_row(row)
                return str(table)
            else:
                for i in range(self._print_max_nrows // 2):
                    row = self[i, :].rows()[0]
                    row.insert(0, i)
                    table.add_row(row)
                divider = ['...' for _ in range(self.ncol)]
                divider.insert(0, '')
                table.add_row(divider)
                for i in range(self._print_max_nrows // 2, 0, -1):
                    row = self[self.nrow - i, :].rows()[0]
                    row.insert(0, self.nrow - i)
                    table.add_row(row)
                out = '{}\n\n[{} rows x {} columns]'
                return out.format(str(table), self.nrow, self.ncol)

    @property
    def ncol(self):
        return self._ncol

    @property
    def nrow(self):
        return self._nrow

    @property
    def shape(self):
        return (self._nrow, self._ncol)

    @property
    def names(self):
        # Copy and provide to user to avoid accidental changes.
        # Use Array over list so user can use array functions.
        return Array(self._names)

    @names.setter
    def names(self, value):
        if len(self._names) == get_length(value):
            if is_iterable_string(value):
                self._names = Array(value)
                self._update_names_to_index()
            else:
                msg = 'non string names are not allowed'
                raise ValueError(msg)
        else:
            msg = 'number of names must match the number of columns'
            raise ValueError(msg)

    @property
    def dtypes(self):
        return Array([column.dtype for column in self._data])

    def keys(self):
        # Copy and provide to user to avoid accidental changes.
        # Use Array over list so user can use array functions.
        return Array(self._names)

    def values(self):
        return [column for column in self._data]

    def items(self):
        return [(key, value) for key, value in zip(self.keys(), self.values())]

    def rows(self):
        return [[self._data[col_index][row_index]
                for col_index in range(self._ncol)]
                for row_index in range(self._nrow)]

    def columns(self):
        return self.values()

    def head(self, nrows=6):
        return self[:nrows, :]

    def __getitem__(self, key):
        if is_float(key):
            msg = 'float index is not supported; please cast to int'
            raise KeyError(msg)
        elif is_bool(key):
            msg = 'logical indexing must provide a list of full length'
            raise KeyError(msg)
        elif is_integer(key):
            return self._data[key]
        elif is_string(key):
            return self._data[self._names_to_index[key]]
        elif isinstance(key, slice):
            return DataFrame.from_items(zip(self._names[key], self._data[key]))
        elif isinstance(key, list):
            if is_iterable_string(key):
                key = [self._names_to_index[k] for k in key]
            return DataFrame.from_items(zip(self._names[key], self._data[key]))
        elif isinstance(key, tuple):
            # Dual Indexing. Select both rows and columns.
            if len(key) == 2:
                rowkey = key[0]
                colkey = key[1]
                if isinstance(colkey, tuple):
                    colkey = list(colkey)
                if is_integer(rowkey):
                    row_items = [
                        (name, [value[rowkey]])
                        for name, value in zip(self._names, self._data)]
                else:
                    row_items = [
                        (name, value[rowkey])
                        for name, value in zip(self._names, self._data)]
                selection = DataFrame.from_items(row_items)[colkey]

                if is_integer(rowkey):
                    if is_integer(colkey) or is_string(colkey):
                        selection = selection[0]
                return selection
            else:
                msg = 'tuple indexing must have exactly 2 elements'
                raise KeyError(msg)
        elif isinstance(key, Iterable):
            return DataFrame.from_items(zip(self._names[key], self._data[key]))
        else:
            # Catchall for all other addresses
            msg = 'address must be int, string, list, slice, or a 2-tuple'
            raise KeyError(msg)

    def _create_array(self, value):
        if is_scalar(value):
            return Array([value] * self._nrow)
        else:
            return Array(value)

    def _setitem_using_int_key(self, key, value):
        assert is_integer(key)
        tmp = self._create_array(value)
        if len(tmp) == self._nrow:
            self._data[key] = tmp
        else:
            msg = 'value does not have match existing number of rows = {}'
            raise ValueError(msg.format(self._nrow))

    def _append_new_column(self, name, value):
        assert is_string(name)
        assert name not in self._names
        tmp = self._create_array(value)
        if len(tmp) == self._nrow:
            # FIXME: Override Array.append by checking for type
            self._names.append(name)
            self._data.append(tmp)
            self._update_nrow_ncol()
            self._update_names_to_index()
        else:
            msg = 'value does not have match existing number of rows = {}'
            raise ValueError(msg.format(self._nrow))

    def _parse_colkey(self, colkey):
        if isinstance(colkey, slice):
            colkey = range(*colkey.indices(len(self)))
        if is_iterable_string(colkey):
            colkey = [self._names_to_index[k] for k in colkey]
        if infer_dtype(colkey) is bool:
            if any([k is None for k in colkey]):
                msg = 'logical index contains missing values (None(s))'
                raise IndexError(msg)
            else:
                if get_length(colkey) == len(self):
                    colkey = [i for i, k in enumerate(colkey) if k]
                else:
                    msg = 'logical index does not match number of columns'
                    raise IndexError(msg)
        valid = range(self._ncol)
        if not all([k in valid for k in colkey]):
            msg = 'invalid column key'
            raise KeyError(msg)
        return colkey

    def _setitem_using_list_of_int_key_numpy_value(self, key, value):
        assert isinstance(key, list)
        assert is_iterable_integer(key)
        assert isinstance(value, np.array)
        if len(value.shape) == 2:
            if len(key) == value.shape[1]:
                if self._nrow == value.shape[0]:
                    for i, k in enumerate(key):
                        self._setitem_using_int_key(k, value[:, i])
                else:
                    msg = 'key and value do not have the same number of rows'
                    raise ValueError(msg)
            else:
                msg = 'key and value do not have the same number of columns'
                raise ValueError(msg)
        else:
            msg = 'key and value do not have the same number of dimensions'
            raise ValueError(msg)

    def _setitem_using_list_of_int_key_pandas_value(self, key, value):
        assert isinstance(key, list)
        assert is_iterable_integer(key)
        assert isinstance(value, pd.DataFrame)
        if len(value.shape) == 2:
            if len(key) == value.shape[1]:
                if self._nrow == value.shape[0]:
                    for i, k in enumerate(key):
                        self._setitem_using_int_key(k, value.iloc[:, i])
                else:
                    msg = 'key and value do not have the same number of rows'
                    raise ValueError(msg)
            else:
                msg = 'key and value do not have the same number of columns'
                raise ValueError(msg)
        else:
            msg = 'pandas DataFrame value of shape = {} cannot be assigned'
            raise NotImplementedError(msg.format(value.shape))

    def _setitem_elements_using_list_of_int_key_numpy_value(
            self, rowkey, colkey, value):
        assert isinstance(colkey, list)
        assert is_iterable_integer(colkey)
        assert isinstance(value, np.array)
        if len(value.shape) == 2:
            if len(colkey) == value.shape[1]:
                for i, k in enumerate(colkey):
                    self._data[k][rowkey] = value[:, i]
            else:
                msg = 'key and value do not have the same number of columns'
                raise ValueError(msg)
        else:
            msg = 'key and value do not have the same number of dimensions'
            raise ValueError(msg)

    def _setitem_elements_using_list_of_int_key_pandas_value(
            self, rowkey, colkey, value):
        assert isinstance(colkey, list)
        assert is_iterable_integer(colkey)
        assert isinstance(value, pd.DataFrame)
        if len(value.shape) == 2:
            if len(colkey) == value.shape[1]:
                for i, k in enumerate(colkey):
                    self._data[k][rowkey] = value.iloc[:, i]
            else:
                msg = 'key and value do not have the same number of columns'
                raise ValueError(msg)
        else:
            msg = 'key and value do not have the same number of dimensions'
            raise ValueError(msg)

    def _setitem_using_rowkey_colkey(self, rowkey, colkey, value):
        if is_float(colkey):
            msg = 'float index is not supported; please cast to int'
            raise KeyError(msg)
        elif is_bool(colkey):
            msg = 'logical indexing must provide a list of full length'
            raise KeyError(msg)
        elif is_integer(colkey):
            self._data[colkey][rowkey] = value
        elif is_string(colkey):
            colkey = self._names_to_index[colkey]
            self._data[colkey][rowkey] = value
        elif isinstance(colkey, (slice, list)):
            colkey = self._parse_colkey(colkey)
            if is_scalar(value):
                for k in colkey:
                    self._data[k][rowkey] = value
            elif isinstance(value, np.array):
                self._setitem_elements_using_list_of_int_key_numpy_value(
                    rowkey, colkey, value)
            elif isinstance(value, pd.Series):
                msg = ('pandas Series is not supported, please use pandas '
                       'DataFrame instead')
                raise ValueError(msg)
            elif isinstance(value, pd.DataFrame):
                self._setitem_elements_using_list_of_int_key_pandas_value(
                    rowkey, colkey, value)
            elif isinstance(value, Iterable):
                if len(colkey) == len(value):
                    for k, v in zip(colkey, value):
                        self._data[k][rowkey] = v
                else:
                    msg = ('key and value do not have the same number '
                           'of columns')
                    raise ValueError(msg)
            else:
                msg = 'cannot assign {} type value'.format(type(value))
                raise ValueError(msg)
        else:
            # Catchall for all other addresses
            msg = 'column key must be int, string, list, or slice'
            raise KeyError(msg)

    def __setitem__(self, key, value):
        if is_float(key):
            msg = 'float index is not supported; please cast to int'
            raise KeyError(msg)
        elif is_bool(key):
            msg = 'logical indexing must provide a list of full length'
            raise KeyError(msg)
        elif is_integer(key):
            self._setitem_using_int_key(key, value)
        elif is_string(key):
            if key in self._names:
                self._setitem_using_int_key(self._names_to_index[key], value)
            else:
                self._append_new_column(key, value)
        elif isinstance(key, (slice, list)):
            key = self._parse_colkey(key)
            if is_scalar(value):
                for k in key:
                    self._setitem_using_int_key(k, value)
            elif isinstance(value, np.array):
                self._setitem_using_list_of_int_key_numpy_value(key, value)
            elif isinstance(value, pd.Series):
                msg = ('pandas Series is not supported, please use pandas '
                       'DataFrame instead')
                raise ValueError(msg)
            elif isinstance(value, pd.DataFrame):
                self._setitem_using_list_of_int_key_pandas_value(key, value)
            elif isinstance(value, Iterable):
                if len(key) == len(value):
                    for k, v in zip(key, value):
                        self._setitem_using_int_key(k, v)
                else:
                    msg = ('key and value do not have the same number '
                           'of columns')
                    raise ValueError(msg)
            else:
                msg = 'cannot assign {} type value'.format(type(value))
                raise ValueError(msg)
        elif isinstance(key, tuple):
            # Dual Indexing. Set both rows and columns.
            if len(key) == 2:
                rowkey = key[0]
                colkey = key[1]
                self._setitem_using_rowkey_colkey(rowkey, colkey, value)
            else:
                msg = 'tuple indexing must have exactly 2 elements'
                raise KeyError(msg)
        else:
            # Catchall for all other addresses
            msg = 'key must be int, string, list, slice, or a 2-tuple'
            raise KeyError(msg)

    def _delitem_colkey(self, colkey):
        colkey = self._parse_colkey(colkey)
        del self._data[colkey]
        del self._names[colkey]
        self._update_nrow_ncol()
        self._update_names_to_index()

    def _delitem_rowkey(self, rowkey):
        for j in range(len(self)):
            del self._data[j][rowkey]
        self._update_nrow_ncol()

    def __delitem__(self, key):
        if is_float(key):
            msg = 'float index is not supported; please cast to int'
            raise KeyError(msg)
        elif is_bool(key):
            msg = 'logical indexing must provide a list of full length'
            raise KeyError(msg)
        elif is_integer(key):
            key = [key]
            self._delitem_colkey(key)
        elif is_string(key):
            key = [self._names_to_index[key]]
            self._delitem_colkey(key)
        elif isinstance(key, (slice, list)):
            self._delitem_colkey(key)
        elif isinstance(key, tuple):
            # Dual Indexing. Set both rows and columns.
            if len(key) == 2:
                rowkey = key[0]
                colkey = key[1]
                if isinstance(colkey, tuple):
                    colkey = list(colkey)
                if colkey == slice(None):
                    # Form: del df[<something>, :]
                    self._delitem_rowkey(rowkey)
                else:
                    # colkey is not `:`
                    if rowkey == slice(None):
                        # Form: del df[:, <something-but-not-:>]
                        del self[colkey]
                    else:
                        # Neither colkey nor rowkey is `:`
                        msg = 'either row key or column key must be :'
                        raise KeyError(msg)
            else:
                msg = 'tuple indexing must have exactly 2 elements'
                raise KeyError(msg)

    def reset_names(self):
        self.names = _get_generic_names(self.ncol)

    def rename(self, rename_dict):
        '''
            Rename the columns of the DataFrame object. Renaming happens
            in place.

            Args
            -----
            rename_dict (dict): a dictionary of the form
                {'existing_column_name': 'new_column_name', ... }. Keys of
                `rename_dict` are the existing column names. Values of
                `rename_dict` are the intended new column names.

            Returns
            --------
            Nothing. Renaming happens in place.
        '''
        assert isinstance(rename_dict, dict)
        updated_names = list(self._names)
        for current, new in rename_dict.items():
            updated_names[self._names_to_index[current]] = new
        if is_iterable_string(updated_names):
            if is_list_unique(updated_names):
                self._names = Array(updated_names)
                self._update_names_to_index()
                if set(self._names_to_index.keys()) != set(self._names):
                    msg = ('renaming violated internal consistency ',
                           'this is a bug, please report it')
                    raise InternalError(msg)
            else:
                msg = 'renaming cannot create duplicate names'
                raise ValueError(msg)
        else:
            msg = 'non string names are not allowed'
            raise ValueError(msg)

    def equals(self, other):
        if self is other:
            return True
        else:
            if type(self) is type(other):
                if self.shape == other.shape:
                    if self._names.equals(other._names):
                        if self._data.equals(other._data):
                            if self._print_max_nrows == other._print_max_nrows:
                                return True
                            else:
                                return False
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False












