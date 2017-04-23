import numpy as np
import pandas as pd
import csv
from collections import OrderedDict
from prettytable import PrettyTable

from _column import Column
from utils import is_list_unique, is_scalar, is_string_type, \
    is_iterable_string_type, is_iterable_int_type, is_iterable_bool_type, \
    get_length, is_list_same, to_best_dtype


def _get_generic_names(ncol):
    return ['C{}'.format(i) for i in range(ncol)]


class DataFrame(object):
    _data = None
    _names = None
    _names_to_index = None
    _ncol = None
    _nrow = None

    # --------------------------------------------------------------------
    # Constructors
    # --------------------------------------------------------------------
    def __init__(self, data=None):
        if data is None:
            data = {}
        if isinstance(data, dict):
            self._init_from_dict(data)
        elif isinstance(data, self.__class__):
            self._data = data._data
            self._names = data._names
            self._names_to_index = data._names_to_index
            self._ncol = data._ncol
            self._nrow = data._nrow
        else:
            # FIXME: list/tuple, numpy array (vector, matrix)
            msg = '{} construction from {} is not supported'
            msg = msg.format(self.__class__.__name__, type(data))
            raise NotImplementedError(msg)

    @classmethod
    def from_dict(cls, data):
        assert isinstance(data, dict)
        return cls(data)

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
                msg = 'number of names must match the number of columns'
                raise ValueError(msg)
        else:
            msg = 'all rows must have the same number of columns'
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
            msg = 'number of names must match the number of columns'
            raise ValueError(msg)

    @classmethod
    def from_items(cls, items):
        # Ensure names are unique
        if is_list_unique([name for name, _ in items]):
            return cls(OrderedDict(items))
        else:
            raise ValueError('provided column names must be unique')

    @classmethod
    def from_numpy(cls, array, names=None):
        # FIX: Handle missing values. Convert them to None.
        assert isinstance(array, np.ndarray)
        if len(array.shape) == 0:
            msg = 'internal error; numpy array has a zero-length shape tuple'
            raise ValueError(msg)
        elif len(array.shape) == 1:
            return cls.from_columns([array], names)
        elif len(array.shape) == 2:
            columns = [array[:, j] for j in range(array.shape[1])]
            return cls.from_columns(columns, names)
        else:
            msg = 'numpy array dimensions must be less than or equal to 2'
            raise ValueError(msg)

    @classmethod
    def from_pandas(cls, df):
        '''
            Create a dframe.DataFrame from pandas.DataFrame object.

            Args
            -----
                df (pandas.DataFrame): input dataframe


            Returns
            --------
                dframe.DataFrame

            Notes
            ------
            1. pandas.DataFrame allows duplicate column names,  which is not
            allowed in dframe.DataFrame. Attempt to convert a
            pandas.DataFrame with duplicate names into a dframe.DataFrame
            will fail. Please rename pandas.DataFrame columns to have unique
            names.

            2. pandas.Series objects are not supported. Please convert
            pandas.Series into a pandas.DataFrame object and then use
            this function.
        '''
        if isinstance(df, pd.Series):
            msg = ('pandas Series is not supported, please use pandas '
                   'DataFrame instead')
            raise ValueError(msg)
        else:
            assert isinstance(df, pd.DataFrame)
        # FIX: Handle missing values. Convert them to None.
        items = [(name, df[name]) for name in df]
        return cls.from_items(items)

    @classmethod
    def from_shape(cls, shape, names=None):
        if get_length(shape) == 2:
            if is_iterable_int_type(shape):
                if (shape[0] >= 0) and (shape[1] >= 0):
                    if names is None:
                        names = _get_generic_names(shape[1])
                    if shape[1] == get_length(names):
                        items = [(names[j], [None for i in range(shape[0])])
                                 for j in range(shape[1])]
                        return cls.from_items(items)
                    else:
                        msg = ('number of names must match the number of '
                               'columns')
                        raise ValueError(msg)
                else:
                    msg = 'shape elements must be non-negative'
                    raise ValueError(msg)
            else:
                msg = 'shape elements must be int'
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
            for j, value in enumerate(df.values()):
                df[j] = to_best_dtype(df[j])
        return df

    def _init_from_dict(self, data):
        # Behavior depends on whether values are scalar or not.
        scalarity_per_value = [is_scalar(value) for value in data.values()]
        if all(scalarity_per_value):
            # Simply box all scalar values
            _data = [Column([value]) for value in data.values()]
        elif any(scalarity_per_value):
            # At least one value is scalar but all values are not scalars
            # Allocate a list and put non-scalar values inside.
            _data = [None] * len(data)
            length_per_value = [None] * len(data)
            for i, value in enumerate(data.values()):
                if not scalarity_per_value[i]:
                    _data[i] = Column(value)
                    length_per_value[i] = len(_data[i])

            # All non-scalar columns must have the same length or
            # we raise a ValueError
            length_non_scalars = set(
                [length for length, scalarity in
                 zip(length_per_value, scalarity_per_value)
                 if not scalarity])
            if length(length_non_scalars) > 1:
                msg = 'provided columns must have the same lengths'
                raise ValueError(msg)
            elif len(length_non_scalars) == 0:
                msg = 'you found a bug, please report it'
                raise ValueError(msg)
            else:
                length = list(length_non_scalars)[0]
        else:
            # All values are non-scalars. No need to box them.
            _data = [Column(value) for value in data.values()]

        # Ensure dict keys are string types
        if not is_iterable_string_type(data.keys()):
            msg = 'provided column names must all be string types'
            raise ValueError(msg)
        else:
            _names = data.keys()

        # Set curated internal vars
        self._data = _data
        self._names = _names

        # Update all other fields
        self._update_nrow_ncol()
        self._update_names_to_index()

    def _update_nrow_ncol(self):
        column_lengths = set(map(len, self._data))
        if len(column_lengths) > 1:
            msg = ('found different column lengths which is a a bug, '
                   'please report it')
            raise ValueError(msg)
        elif len(column_lengths) == 0:
            self._nrow = 0
        else:
            self._nrow = list(column_lengths)[0]

        self._ncol = len(self._data)

    def _update_names_to_index(self):
        self._names_to_index = {name: index
                                for index, name in enumerate(self._names)}

    # --------------------------------------------------------------------
    # Customary functions
    # --------------------------------------------------------------------
    def __len__(self):
        return self._ncol

    def __iter__(self):
        for name in self._names:
            yield name

    def __repr__(self):
        # # FIXME: pandas has __str__ and __repr__ both returning the same
        # # result which is the dataframe in table format (a la R).
        # # Decide if you want to do that.
        # msg = '{cls}(nrow={nrow}, ncol={ncol})'
        # msgdict = {'cls': self.__class__.__name__,
        #            'nrow': self.nrow, 'ncol': self.ncol}
        # return msg.format(**msgdict)
        return str(self)

    def __str__(self):
        if self.ncol == 0:
            out = 'DataFrame with {} rows and {} columns'
            return out.format(self.nrow, self.ncol)
        else:
            headers = self.names
            headers.insert(0, '')
            table = PrettyTable(headers)
            for i, row in enumerate(self.rows()):
                row.insert(0, i)
                table.add_row(row)
            return str(table)

    # --------------------------------------------------------------------
    # Shape information extraction functions
    # --------------------------------------------------------------------
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
        # We copy and provide to the user so user's edits cannot
        # disturb our internal data model.
        return list(self._names)

    @names.setter
    def names(self, value):
        if len(self._names) == get_length(value):
            if is_iterable_string_type(value):
                self._names = value
                self._update_names_to_index()
            else:
                msg = 'all names must be strings'
                raise ValueError(msg)
        else:
            msg = ('number of names provided must match the number '
                   'of columns')
            raise ValueError(msg)

    @property
    def dtypes(self):
        return [column.dtype for column in self._data]

    def keys(self):
        # We copy and provide to the user so user's edits cannot
        # disturb our internal data model.
        return list(self._names)

    def values(self):
        return [column.data for column in self._data]

    def items(self):
        return [(key, value) for key, value in zip(self.keys(), self.values())]

    def rows(self):
        return [[self._data[col_index][row_index].data[0]
                for col_index in xrange(self._ncol)]
                for row_index in xrange(self._nrow)]

    def columns(self):
        return self.values()

    # --------------------------------------------------------------------
    # Shape information extraction functions
    # --------------------------------------------------------------------
    def head(self, nrows=6):
        return self[:nrows, :]

    # --------------------------------------------------------------------
    # Get Indexing
    # --------------------------------------------------------------------
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key].data
        elif is_string_type(key):
            return self._data[self._names_to_index[key]].data
        elif isinstance(key, slice):
            selection = [
                (name, value)
                for name, value in zip(self._names[key], self._data[key])]
            return DataFrame.from_items(selection)
        elif isinstance(key, list):
            # FIXME: Can I use self._convert_list_col_address_to_list_of_int()?
            if is_iterable_int_type(key):
                selection = [(self._names[k], self._data[k]) for k in key]
            elif is_iterable_string_type(key):
                selection = [(self._names[self._names_to_index[k]],
                             self._data[self._names_to_index[k]])
                             for k in key]
            elif is_iterable_bool_type(key):
                # Remember: length of key must match number of columns
                msg = 'logical indexing not yet implemented'
                raise NotImplementedError(msg)
            else:
                # FIXME: This message needs to be fixed when logical indexing
                # is implemented.
                msg = 'list address must contain all int or all string types'
                raise KeyError(msg)
            return DataFrame.from_items(selection)
        elif isinstance(key, float):
            msg = 'float address is not supported; please cast to int'
            raise KeyError(msg)
        elif isinstance(key, bool):
            msg = 'logical indexing must provide a list of full length'
            raise KeyError(msg)
        elif isinstance(key, tuple):
            # Dual Indexing
            if len(key) != 2:
                msg = 'tuple address must have exactly two elements'
                raise KeyError(msg)
            else:
                row_address = key[0]
                col_address = key[1]
                return self._getitem_using_row_col_address(row_address,
                                                           col_address)
        else:
            # Catchall for all other addresses
            msg = 'address must be int, string, list, slice, or a 2-tuple'
            raise KeyError(msg)

    def _getitem_using_row_col_address(self, row_address, col_address):
        if isinstance(row_address, (int, slice, list)):
            # Note that to allow logical indexing, we need to ensure Column
            # can support logical indexing too.
            row_selection = [(name, value[row_address])
                             for name, value in zip(self._names, self._data)]
            selection = DataFrame.from_items(row_selection)[col_address]

            # Unbox if provided address happens to refer to a scalar
            if isinstance(row_address, int):
                if isinstance(col_address, int) or is_string_type(col_address):
                    selection = selection[0]
            return selection
        else:
            # FIXME: This message needs to be fixed when logical indexing
            # is implemented.
            msg = 'row address must be int, slice, or list (of int)'
            raise KeyError(msg)

    # --------------------------------------------------------------------
    # Set Indexing
    # --------------------------------------------------------------------
    def _create_unnamed_column(self, value):
        if is_scalar(value):
            return Column([value] * self._nrow)
        else:
            return Column(value)

    def _setitem_using_int_key(self, key, value):
        assert isinstance(key, int)
        tmp = self._create_unnamed_column(value)
        if len(tmp) == self._nrow:
            self._data[key] = tmp
        else:
            msg = 'provided value must match number of rows = {}'
            raise ValueError(msg.format(self._nrow))

    def _append_new_column(self, name, value):
        assert is_string_type(name)
        assert name not in self._names
        tmp = self._create_unnamed_column(value)
        if len(tmp) == self._nrow:
            self._names.append(name)
            self._data.append(tmp)
            self._update_nrow_ncol()
            self._update_names_to_index()
        else:
            msg = 'provided value must match number of rows = {}'
            raise ValueError(msg.format(self._nrow))

    def _is_list_of_int_col_address_valid(self, key):
        assert isinstance(key, list)
        assert is_iterable_int_type(key)
        valid_address_pool = range(self._ncol)
        for k in key:
            if k not in valid_address_pool:
                return False
        return True

    def _convert_list_col_address_to_list_of_int(self, key):
        assert isinstance(key, list)
        if is_iterable_int_type(key):
            if is_list_unique(key):
                if self._is_list_of_int_col_address_valid(key):
                    return key
                else:
                    msg = 'column list address must have valid int values'
                    raise KeyError(msg)
            else:
                msg = 'list address must refer to unique columns'
                raise KeyError(msg)
        elif is_iterable_string_type(key):
            if is_list_unique(key):
                return [self._names_to_index[k] for k in key]
            else:
                msg = 'list address must refer to unique columns'
                raise KeyError(msg)
        elif is_iterable_bool_type(key):
            # Remember: length of key must match number of columns
            # No need to check for uniqueness
            msg = 'logical indexing not yet implemented'
            raise NotImplementedError(msg)
        else:
            # FIXME: This message needs to be fixed when logical indexing
            # is implemented.
            msg = 'list address must contain all int or all string types'
            raise KeyError(msg)

    def _setitem_columns_list_of_int_numpy_value(self, key, value):
        assert isinstance(key, list)
        assert is_iterable_int_type(key)
        assert isinstance(value, np.array)
        if len(value.shape) > 2:
            msg = ('provided value and address must has the same number of '
                   'dimensions')
            raise ValueError(msg)
        elif len(value.shape) == 2:
            if len(key) == value.shape[1]:
                if self._nrow == value.shape[0]:
                    for i, k in enumerate(key):
                        self._setitem_using_int_key(k, value[:, i])
                else:
                    msg = ('provided value and address do not have the '
                           'same number of rows')
                    raise ValueError(msg)
            else:
                msg = ('provided value and address do not have the '
                       'same number of columns')
                raise ValueError(msg)
        elif len(value.shape) == 1:
            # FIXME: Should I even support this?
            # This feels hella ambiguous.
            if len(key) == value.shape[0]:
                for k, v in zip(key, value):
                        self._setitem_using_int_key(k, v)
            else:
                msg = ('provided value and address do not have the '
                       'same number of columns')
                raise ValueError(msg)
        elif len(value.shape) == 0:
            msg = 'internal error; numpy array has a zero-length shape tuple'
            raise ValueError(msg)

    def _setitem_columns_list_of_int_pandas_value(self, key, value):
        assert isinstance(key, list)
        assert is_iterable_int_type(key)
        assert isinstance(value, pd.DataFrame)
        if len(value.shape) == 2:
            if len(key) == value.shape[1]:
                if self._nrow == value.shape[0]:
                    for i, k in enumerate(key):
                        self._setitem_using_int_key(k, value.iloc[:, i])
                else:
                    msg = ('provided value and address do not have the '
                           'same number of rows')
                    raise ValueError(msg)
            else:
                msg = ('provided value and address do not have the '
                       'same number of columns')
                raise ValueError(msg)
        else:
            msg = ('assigning a pandas DataFrame value of shape = {} '
                   'is not supported')
            raise NotImplementedError(msg.format(value.shape))

    def _setitem_elements_list_of_int_numpy_value(
            self, row_address, col_address, value):
        assert isinstance(col_address, list)
        assert is_iterable_int_type(col_address)
        assert isinstance(value, np.array)
        if len(value.shape) == 2:
            if len(col_address) == value.shape[1]:
                for i, k in enumerate(col_address):
                    self._data[k][row_address] = value[:, i]
            else:
                msg = ('provided value and address do not have the '
                       'same number of columns')
                raise ValueError(msg)
        else:
            msg = ('provided value and address must has the same number of '
                   'dimensions')
            raise ValueError(msg)

    def _setitem_elements_list_of_int_pandas_value(
            self, row_address, col_address, value):
        assert isinstance(col_address, list)
        assert is_iterable_int_type(col_address)
        assert isinstance(value, pd.DataFrame)
        if len(value.shape) == 2:
            if len(col_address) == value.shape[1]:
                for i, k in enumerate(col_address):
                    self._data[k][row_address] = value.iloc[:, i]
            else:
                msg = ('provided value and address do not have the '
                       'same number of columns')
                raise ValueError(msg)
        else:
            msg = ('provided value and address must has the same number of '
                   'dimensions')
            raise ValueError(msg)

    def _setitem_using_row_col_address(self, row_address, col_address, value):
        if isinstance(col_address, slice):
            col_address = range(*col_address.indices(len(self)))

        if isinstance(col_address, int):
            self._data[col_address][row_address] = value
        elif is_string_type(col_address):
            self._data[self._names_to_index[col_address]][row_address] = value
        elif isinstance(col_address, slice):
            msg = ('slice address found in the wrong conditional, '
                   'which is a bug, please report it')
            raise ValueError(msg)
        elif isinstance(col_address, list):
            col_address = \
                self._convert_list_col_address_to_list_of_int(col_address)
            # FIXME: Make this transactional. Ask around for best practices.
            if is_scalar(value):
                for k in col_address:
                    self._data[k][row_address] = value
            elif isinstance(value, (list, tuple, self.__class__)):
                if len(col_address) == len(value):
                    for k, v in zip(col_address, value):
                        self._data[k][row_address] = v
                else:
                    msg = ('column address and value must have the same '
                           'number of items')
                    raise ValueError(msg)
            elif isinstance(value, np.array):
                self._setitem_elements_list_of_int_numpy_value(
                    row_address, col_address, value)
            elif isinstance(value, pd.Series):
                msg = ('pandas Series is not supported, please use pandas '
                       'DataFrame instead')
                raise ValueError(msg)
            elif isinstance(value, pd.DataFrame):
                self._setitem_elements_list_of_int_pandas_value(
                    row_address, col_address, value)
            elif isinstance(col_address, float):
                msg = ('float column address is not supported; '
                       'please cast to int')
                raise KeyError(msg)
        elif isinstance(col_address, bool):
            msg = 'logical column indexing must provide a list of full length'
            raise KeyError(msg)
        else:
            # FIXME: This message needs to be fixed when logical indexing
            # is implemented.
            msg = 'column address must be int, string, slice, or list of int'
            raise KeyError(msg)

    def __setitem__(self, key, value):
        # Pre-process some address types
        if isinstance(key, slice):
            key = range(*key.indices(len(self)))

        if isinstance(key, int):
            self._setitem_using_int_key(key, value)
        elif is_string_type(key):
            if key in self._names:
                self._setitem_using_int_key(self._names_to_index[key], value)
            else:
                self._append_new_column(key, value)
        elif isinstance(key, slice):
            msg = ('slice address found in the wrong conditional, '
                   'which is a bug, please report it')
            raise ValueError(msg)
        elif isinstance(key, list):
            # Get a valid list-of-int key after running through all checks
            key = self._convert_list_col_address_to_list_of_int(key)
            if is_scalar(value):
                for k in key:
                    self._setitem_using_int_key(k, value)
            elif isinstance(value, (list, tuple, self.__class__)):
                if len(key) == len(value):
                    for k, v in zip(key, value):
                        self._setitem_using_int_key(k, v)
                else:
                    msg = 'address and value must have the same length'
                    raise ValueError(msg)
            elif isinstance(value, np.array):
                self._setitem_columns_list_of_int_numpy_value(key, value)
            elif isinstance(value, pd.Series):
                msg = ('pandas Series is not supported, please use pandas '
                       'DataFrame instead')
                raise ValueError(msg)
            elif isinstance(value, pd.DataFrame):
                self._setitem_columns_list_of_int_pandas_value(key, value)
        elif isinstance(key, float):
            msg = 'float address is not supported; please cast to int'
            raise KeyError(msg)
        elif isinstance(key, bool):
            msg = 'logical indexing must provide a list of full length'
            raise KeyError(msg)
        elif isinstance(key, tuple):
            # Dual Indexing
            if len(key) != 2:
                msg = 'tuple address must have exactly two elements'
                raise KeyError(msg)
            else:
                row_address = key[0]
                col_address = key[1]
                self._setitem_using_row_col_address(row_address,
                                                    col_address, value)
        else:
            # Catchall for all other addresses
            msg = 'address must be int, string, list, slice, or a 2-tuple'
            raise KeyError(msg)

    # --------------------------------------------------------------------
    # Drop indexing
    # --------------------------------------------------------------------
    def __delitem__(self, key):
        # Convert all kinds of inputs to a list of ints
        if isinstance(key, int):
            key = [key]
        elif is_string_type(key):
            key = [self._names_to_index[key]]
        elif isinstance(key, slice):
            key = range(*key.indices(len(self)))
        elif isinstance(key, list):
            if is_iterable_int_type(key):
                pass
            elif is_iterable_string_type(key):
                key = [self._names_to_index[k] for k in key]
            elif is_iterable_bool_type(key):
                # Remember: length of key must match number of columns
                msg = 'logical indexing not yet implemented'
                raise NotImplementedError(msg)
            else:
                # FIXME: This message needs to be fixed when logical indexing
                # is implemented.
                msg = 'list address must contain all int or all string types'
                raise KeyError(msg)
        elif isinstance(key, float):
            msg = 'float address is not supported; please cast to int'
            raise KeyError(msg)
        elif isinstance(key, bool):
            msg = 'logical indexing must provide a list of full length'
            raise KeyError(msg)
        else:
            # Catchall for all other addresses
            msg = 'address must be int, string, list, or a slice'
            raise KeyError(msg)

        # Find the indices to keep
        everything = set(range(len(self)))
        if all([k in everything for k in key]):
            keep = everything - set(key)
            keep = sorted(list(keep))
        else:
            raise ValueError('all int column address must be valid')
        self._data = [self._data[k] for k in keep]
        self._names = [self._names[k] for k in keep]
        self._update_nrow_ncol()
        self._update_names_to_index()

    # --------------------------------------------------------------------
    # Interface
    # --------------------------------------------------------------------
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
        # FIXME: Should we use duck typing instead of checking for dict type?
        assert isinstance(rename_dict, dict)
        new_names = self.names
        for existing_name, new_name in rename_dict.items():
            new_names[self._names_to_index[existing_name]] = \
                new_name
        if is_iterable_string_type(new_names):
            if is_list_unique(new_names):
                self._names = new_names
                self._update_names_to_index()
                if set(self._names_to_index.keys()) != set(self._names):
                    msg = ('renaming failed to maintain name consistency; ',
                           'this is a bug, please report it')
                    raise ValueError(msg)
            else:
                raise ValueError('renaming cannot create duplicate names')
        else:
            raise ValueError('new column names must all be strings')
