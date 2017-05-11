from __future__ import absolute_import
import numpy as np


def is_integer(x):
    if isinstance(x, bool):
        return False
    else:
        int_types = (int, long, np.int, np.intp, np.intc, np.int_,
                     np.int0, np.int8, np.int16, np.int32, np.int64)
        return isinstance(x, int_types)


def is_float(x):
    float_types = (float, np.float, np.float_, np.float16, np.float32,
                   np.float64, np.float128)
    return isinstance(x, float_types)


def is_string(x):
    return isinstance(x, basestring)


def is_bool(x):
    bool_types = (bool, np.bool, np.bool_, np.bool8)
    return isinstance(x, bool_types)


def infer_dtype(data):
    ''' Infer the type of data provided

    Args
    -----
    data (iterable):

    Returns
    --------
    type

    '''
    # Find all types ignoring the NoneType.
    # We allow None element to exist as missing value.
    types = list(set(map(type, data)).difference({type(None)}))

    # If all elements in data are None, then types will be empty.
    # If data is empty, then types will be empty.
    # In both cases, we return dtype as NoneType.
    if len(types) == 0:
        dtype = type(None)
    elif len(types) == 1:
        dtype = types[0]
    else:
        msg = 'Multiple types detected in input: {}'
        raise ValueError(msg.format(map(lambda x: x.__name__, types)))
    return dtype


def to_bool(x):
    if x == 'True':
        return True
    elif x == 'False':
        return False
    else:
        raise ValueError('invalid literal for to_bool(): {}'.format(x))
