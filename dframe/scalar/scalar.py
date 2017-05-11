from __future__ import absolute_import

from ..dtypes import is_integer, is_float, is_string
from ..compat import Iterable


def is_scalar(x):
    if x is None:
        return True
    elif is_integer(x) or is_float(x) or is_string(x):
        return True
    elif isinstance(x, Iterable):
        return False
    else:
        return True


def get_length(x):
    assert not is_scalar(x)
    try:
        length = len(x)
    except TypeError:
        try:
            length = 0
            for _ in x:
                length = length + 1
        except TypeError:
            raise
    return length
