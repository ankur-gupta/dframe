from __future__ import absolute_import
from __future__ import print_function
from builtins import super, range

import numpy as np
from dateutil import parser

from dframe.compat import Iterable
from dframe.dtypes import infer_dtype, to_bool, is_string
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


class Array(list):
    def _is_valid_dtype_element(self, element):
        if self.dtype is type(None):
            return True
        else:
            return type(element) in {self.dtype, type(None)}

    def _is_valid_dtype_iterable(self, iterable):
        return all(self._is_valid_dtype_element(element)
                   for element in iterable)

    def _set_by_iterable(self, key, value):
        if is_scalar(value):
            if self._is_valid_dtype_element(value):
                for k in key:
                    super().__setitem__(k, value)
            else:
                msg = 'value type does not match array dtype = {}'
                raise ValueError(msg.format(self.dtype.__name__))
        else:
            if self._is_valid_dtype_iterable(value):
                length_value = get_length(value)
                if len(key) == length_value:
                    for k, v in zip(key, value):
                        self.data[k] = v
                else:
                    msg = 'value and key do not have the same lengths'
                    raise ValueError(msg)
            else:
                msg = 'value type(s) does not match array dtype = {}'
                raise ValueError(msg.format(self.dtype.__name__))

    def _del_by_iterable(self, key):
        assert is_iterable_integer(key)
        valid = range(len(self))
        if not all([k in valid for k in key]):
            msg = 'list index out of range'
            raise IndexError(msg)
        key = sorted(key, reverse=True)
        for k in key:
            super().__delitem__(k)

    def __init__(self, *args):
        super().__init__(*args)
        for index, value in enumerate(self):
            try:
                if np.isnan(value):
                    super().__setitem__(index, None)
            except (TypeError, ValueError):
                pass
        self.dtype = infer_dtype(self)

    def __getslice__(self, start, end):
        return Array(super().__getslice__(start, end))

    def __getitem__(self, key):
        # FIXME: Disable tuple because it feels like dual indexing ?
        if is_string(key):
            msg = 'list indices must be integers, not str'
            raise TypeError(msg)
        elif isinstance(key, Iterable):
            if infer_dtype(key) is bool:
                if get_length(key) == len(self):
                    if any([e is None for e in key]):
                        msg = 'logical index contains missing values (None)'
                        raise IndexError(msg)
                    else:
                        return Array([super().__getitem__(index)
                                     for index, logical in enumerate(key)
                                     if logical])
                else:
                    msg = 'logical index does not match array length'
                    raise IndexError(msg)
            else:
                return Array([super().__getitem__(k) for k in key])
        elif isinstance(key, slice):
            return Array(super().__getitem__(key))
        else:
            return super().__getitem__(key)

    def __delitem__(self, key):
        if isinstance(key, Iterable):
            if infer_dtype(key) is bool:
                if get_length(key) == len(self):
                    if any([e is None for e in key]):
                        msg = 'logical index contains missing values (None)'
                        raise IndexError(msg)
                    else:
                        key = [i for i, k in enumerate(key) if k]
                        self._del_by_iterable(key)
                else:
                    msg = 'logical index does not match array length'
                    raise IndexError(msg)
            else:
                self._del_by_iterable(key)
        else:
            super().__delitem__(key)

    def __setslice__(self, start, end, value):
        key = range(*slice(start, end).indices(len(self)))
        self._set_by_iterable(key, value)
        self.dtype = infer_dtype(self)

    def __setitem__(self, key, value):
        if is_string(key):
            msg = 'list indices must be integers, not str'
            raise TypeError(msg)
        elif isinstance(key, (Iterable, slice)):
            if isinstance(key, slice):
                key = range(*key.indices(len(self)))
            if infer_dtype(key) is bool:
                if get_length(key) == len(self):
                    if any([e is None for e in key]):
                        msg = ('logical index contains missing values (None)')
                        raise IndexError(msg)
                    else:
                        key = [index for index, logical in enumerate(key)
                               if logical]
                else:
                    msg = 'logical index does not match array length'
                    raise IndexError(msg)
            self._set_by_iterable(key, value)
        else:
            if self._is_valid_dtype_element(value):
                super().__setitem__(key, value)
            else:
                msg = 'value type does not match array dtype = {}'
                raise ValueError(msg.format(self.dtype.__name__))
        self.dtype = infer_dtype(self)

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
            return super().__eq__(other)

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
            return super().__ne__(other)

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
            return super().__ge__(other)

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
            return super().__gt__(other)

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
            return super().__le__(other)

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
            return super().__lt__(other)

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
            return super().__or__(other)

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
            return super().__and__(other)

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
            return super().__xor__(other)

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
            return super().__add__(other)

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
            return super().__sub__(other)

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
            return super().__mul__(other)

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
            return super().__pow__(other)

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
            return super().__div__(other)

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
            return super().__truediv__(other)

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
            return super().__floordiv__(other)

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
            return super().__mod__(other)
