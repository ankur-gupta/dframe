from __future__ import absolute_import
from __future__ import print_function
from builtins import super, range

import numpy as np
import pandas as pd
from dateutil import parser

from dframe.compat import Iterable
from dframe.dtypes import (infer_dtype, to_bool, is_string, is_float, is_bool,
                           is_integer)
from dframe.scalar import is_scalar, get_length, is_iterable_integer
from dframe.missing import (__not__, __neg__, __pos__, __abs__,
                            __eq__, __ne__, __ge__, __gt__, __le__,
                            __lt__, __or__, __and__, __xor__,
                            __add__, __sub__, __mul__, __pow__,
                            __mod__, __div__, __truediv__, __floordiv__)


def as_dtype(x, dtypefun):
    assert isinstance(x, Array)
    y = [None] * len(x)
    for index, value in enumerate(x):
        if value is not None:
            y[index] = dtypefun(value)
    return Array(y)


def to_best_dtype(x):
    assert isinstance(x, Array)
    if x.dtype is str:
        try:
            y = as_dtype(x, int)
        except ValueError:
            try:
                y = as_dtype(x, float)
            except ValueError:
                try:
                    y = as_dtype(x, parser.parse)
                except ValueError:
                    try:
                        y = as_dtype(x, to_bool)
                    except ValueError:
                        y = x
    return y


def short_str(x, n_chars=5):
    assert is_string(x)
    assert is_integer(n_chars)
    assert n_chars > 0
    if len(x) < ((2 * n_chars) + 5):
        return x
    else:
        return '{}...{}'.format(x[:(n_chars + 1)], x[-n_chars:])


def nice_str(x):
    if hasattr(x, '__name__'):
        output = x.__name__
    elif is_string(x):
        output = repr(x)
    else:
        output = str(x)
    output = short_str(output)
    return output


class _ArraySlice(object):
    def __init__(self, _data):
        assert isinstance(_data, pd.Series)
        self._data = _data


class Array(object):
    _print_max_n_elements = 10

    def __init__(self, data=[]):
        if isinstance(data, type(self)):
            # Data is not copied a la pd.Series
            self._data = data._data
            self.dtype = data.dtype
        elif isinstance(data, _ArraySlice):
            self._data = data._data
            self.dtype = infer_dtype(self._data)
        else:
            self._data = pd.Series(data, dtype=object)
            # This step is really slow! Avoid this when possible.
            for index, value in enumerate(self._data):
                try:
                    if np.isnan(value):
                        self._data[index] = None
                except (TypeError, ValueError):
                    pass
            self.dtype = infer_dtype(self._data)

    def _is_valid_dtype_element(self, element):
        if self.dtype is type(None):
            return True
        else:
            return type(element) in {self.dtype, type(None)}

    def _is_valid_dtype_iterable(self, iterable):
        return all(self._is_valid_dtype_element(element)
                   for element in iterable)

    def _convert_logical_index_to_int_index(self, key):
        assert infer_dtype(key) is bool
        if get_length(key) == len(self):
            if any([e is None for e in key]):
                msg = 'logical index contains missing values (None)'
                raise IndexError(msg)
            else:
                key = [i for i, k in enumerate(key) if k]
        else:
            msg = 'logical index does not match array length'
            raise IndexError(msg)
        return key

    def __getitem__(self, key):
        if is_float(key):
            msg = 'array index cannot be float; please cast to int'
            raise TypeError(msg)
        elif is_bool(key):
            msg = 'logical indexing must provide an iterable of full length'
            raise TypeError(msg)
        elif is_string(key):
            msg = 'array index cannot be string'
            raise TypeError(msg)
        elif isinstance(key, tuple):
            msg = ('tuple is ambiguous because it can refer to dual-indexing; '
                   'convert index to list')
            raise TypeError(msg)
        elif is_integer(key):
            return self._data.iloc[key]
        elif isinstance(key, Iterable) and infer_dtype(key) is bool:
            key = self._convert_logical_index_to_int_index(key)
            return type(self)(_ArraySlice(self._data.iloc[key]))
        else:
            return type(self)(_ArraySlice(self._data.iloc[key]))

    def _del_by_iterable(self, key):
        assert is_iterable_integer(key)
        valid = range(len(self))
        if not all([k in valid for k in key]):
            msg = 'list index out of range'
            raise IndexError(msg)

        # Instead of deletion, overwrite the data by things we want to keep
        key_inverse = sorted(list(set(valid).difference(set(key))))
        self._data = self._data.iloc[key_inverse]

    # def _del_by_iterable(self, key):
    #     # This implementation deletes by element and can be very slow!
    #     # pd.Series doesn't even support deletion by list or slice.
    #     assert is_iterable_integer(key)
    #     valid = range(len(self))
    #     if not all([k in valid for k in key]):
    #         msg = 'list index out of range'
    #         raise IndexError(msg)

    #     # We can only drop by pd.Series index (not using row numbers).
    #     # So, we ensure that the Series index is the same as row numbers.
    #     self._data.reset_index(drop=True, inplace=True)

    #     # Delete in reverse order so remaining keys remain valid
    #     key = sorted(key, reverse=True)
    #     for k in key:
    #         del self._data[k]

    def __delitem__(self, key):
        if is_float(key):
            msg = 'array index cannot be float; please cast to int'
            raise KeyError(msg)
        elif is_bool(key):
            msg = 'logical indexing must provide an iterable of full length'
            raise KeyError(msg)
        elif is_string(key):
            msg = 'array index cannot be string'
            raise TypeError(msg)
        elif isinstance(key, tuple):
            msg = ('tuple is ambiguous because it can refer to dual-indexing; '
                   'convert index to list')
            raise TypeError(msg)
        elif is_integer(key):
            # We can only drop by pd.Series index (not using row numbers).
            # So, we ensure that the Series index is the same as row numbers.
            self._data.reset_index(drop=True, inplace=True)
            del self._data[key]
            self.dtype = infer_dtype(self._data)
        elif isinstance(key, slice):
            key = range(*key.indices(len(self)))
            self._del_by_iterable(key)
        elif isinstance(key, Iterable):
            if infer_dtype(key) is bool:
                key = self._convert_logical_index_to_int_index(key)
            self._del_by_iterable(key)
            self.dtype = infer_dtype(self._data)
        else:
            msg = 'index can only be int or iterable (int, bool)'
            raise IndexError(msg)

    def __setitem__(self, key, value):
        if is_float(key):
            msg = 'array index cannot be float; please cast to int'
            raise KeyError(msg)
        elif is_bool(key):
            msg = 'logical indexing must provide an iterable of full length'
            raise KeyError(msg)
        elif is_string(key):
            msg = 'array index cannot be string'
            raise TypeError(msg)
        elif isinstance(key, tuple):
            msg = ('tuple is ambiguous because it can refer to dual-indexing; '
                   'convert index to list')
            raise TypeError(msg)
        elif is_integer(key):
                if self._is_valid_dtype_element(value):
                    self._data.iloc[key] = value
                else:
                    msg = 'value type does not match array dtype = {}'
                    raise ValueError(msg.format(self.dtype.__name__))
        else:
            if is_scalar(value):
                if self._is_valid_dtype_element(value):
                    self._data.iloc[key] = value
                else:
                    msg = 'value type does not match array dtype = {}'
                    raise ValueError(msg.format(self.dtype.__name__))
            else:
                if self._is_valid_dtype_iterable(value):
                    self._data.iloc[key] = value
                else:
                    msg = 'value type does not match array dtype = {}'
                    raise ValueError(msg.format(self.dtype.__name__))
        self.dtype = infer_dtype(self._data)

    def extend(self, other):
        assert isinstance(other, type(self))
        if (self.dtype == other.dtype) or (self.dtype is type(None)):
            self._data = self._data.append(other._data, ignore_index=True)
            self.dtype = infer_dtype(self._data)
        else:
            msg = 'cannot extend with a non-{} object'.format(type(self))
            raise TypeError(msg)

    def equals(self, other):
        if self is other:
            return True
        else:
            if type(self) is type(other):
                if self.dtype == other.dtype:
                    if len(self) == len(other):
                        # FIXME: Nesting can be improved.
                        for e1, e2 in zip(self, other):
                            try:
                                if not e1.equals(e2):
                                    return False
                            except AttributeError:
                                try:
                                    if e1 != e2:
                                        return False
                                except Exception:
                                    if e1 is not e2:
                                        return False
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for e in self._data:
            yield e

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if len(self._data) < self._print_max_n_elements:
            output = [nice_str(e) for e in self._data]
            output = '[{}]'.format(', '.join(output))
        else:
            n = int(self._print_max_n_elements / 2)
            start = [nice_str(e) for e in self._data.iloc[:n]]
            end = [nice_str(e) for e in self._data.iloc[-n:]]
            output = '[{}, ..., {}]'.format(', '.join(start), ', '.join(end))
        return output

    def __not__(self):
        return Array([__not__(e) for e in self])

    def __neg__(self):
        return Array([__neg__(e) for e in self])

    def __pos__(self):
        return Array([__pos__(e) for e in self])

    def __abs__(self):
        return Array([__abs__(e) for e in self])

    def __eq__(self, other):
        if is_scalar(other):
            return Array([__eq__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__eq__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __ne__(self, other):
        if is_scalar(other):
            return Array([__ne__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__ne__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __ge__(self, other):
        if is_scalar(other):
            return Array([__ge__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__ge__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __gt__(self, other):
        if is_scalar(other):
            return Array([__gt__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__gt__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __le__(self, other):
        if is_scalar(other):
            return Array([__le__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__le__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __lt__(self, other):
        if is_scalar(other):
            return Array([__lt__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__lt__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __or__(self, other):
        if is_scalar(other):
            return Array([__or__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__or__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __and__(self, other):
        if is_scalar(other):
            return Array([__and__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__and__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __xor__(self, other):
        if is_scalar(other):
            return Array([__xor__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__xor__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __add__(self, other):
        if is_scalar(other):
            return Array([__add__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__add__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __sub__(self, other):
        if is_scalar(other):
            return Array([__sub__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__sub__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __mul__(self, other):
        if is_scalar(other):
            return Array([__mul__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__mul__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __pow__(self, other):
        if is_scalar(other):
            return Array([__pow__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__pow__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __div__(self, other):
        if is_scalar(other):
            return Array([__div__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__div__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __truediv__(self, other):
        if is_scalar(other):
            return Array([__truediv__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__truediv__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __floordiv__(self, other):
        if is_scalar(other):
            return Array([__floordiv__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__floordiv__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))

    def __mod__(self, other):
        if is_scalar(other):
            return Array([__mod__(e, other) for e in self])
        elif isinstance(other, Iterable):
            if len(self) == get_length(other):
                return Array([__mod__(x, y) for x, y in zip(self, other)])
            else:
                msg = 'iterables have different lengths'
                raise ValueError(msg)
        else:
            msg = 'cannot perform this operation with {} object'
            raise ValueError(msg.format(type(object)))
