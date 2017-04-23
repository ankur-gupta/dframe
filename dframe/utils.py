import six
from dateutil import parser


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
        raise ValueError(msg.format(types))
    return dtype


def to_dtype_iterable(data, dtypefun):
    for i, elem in enumerate(data):
        if elem is not None:
            data[i] = dtypefun(elem)
    return data


def to_best_dtype(data):
    dtype = infer_dtype(data)
    if dtype is str:
        try:
            data = to_dtype_iterable(data, int)
        except ValueError:
            try:
                data = to_dtype_iterable(data, float)
            except ValueError:
                try:
                    data = to_dtype_iterable(data, parser.parse)
                except ValueError:
                    pass
    return data


def is_string_type(x):
    return isinstance(x, six.string_types)


def is_scalar(x):
    if x is None:
        return True
    elif isinstance(x, (int, float)):
        return True
    elif is_string_type(x):
        return True
    elif hasattr(x, '__iter__'):
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


def is_list_unique(x):
    assert isinstance(x, list)
    return len(x) == len(set(x))


def is_iterable_unique(x):
    assert not is_scalar(x)
    return is_list_unique([elem for elem in x])


def is_list_same(x):
    assert isinstance(x, list)
    n_same_elements = len(set(x))
    return n_same_elements == 0 or n_same_elements == 1


def is_iterable_same(x):
    assert not is_scalar(x)
    return is_list_same([elem for elem in x])


def is_iterable_string_type(x):
    # Empty list returns True
    for item in x:
        if not is_string_type(item):
            return False
    return True


def is_iterable_int_type(x):
    # Empty list returns True
    for item in x:
        if not isinstance(item, int):
            return False
    return True


def is_iterable_bool_type(x):
    # Empty list returns True
    for item in x:
        if not isinstance(item, bool):
            return False
    return True
