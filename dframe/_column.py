from utils import infer_dtype, is_scalar, get_length


class Column(object):
    data = None
    dtype = None

    def __init__(self, data=[]):
        if isinstance(data, self.__class__):
            self.data = data.data
            self.dtype = data.dtype
        else:
            if is_scalar(data):
                raise ValueError('provided value must be iterable')
            else:
                self.data = [elem for elem in data]
                self.dtype = infer_dtype(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        msg = '{cls}({data}, dtype={dtype})'
        msgdict = {'cls': self.__class__.__name__,
                   'data': self.data, 'dtype': self.dtype.__name__}
        return msg.format(**msgdict)

    def _is_valid_dtype_element(self, element):
        if self.dtype is type(None):
            return True
        else:
            return type(element) in {self.dtype, type(None)}

    def _is_valid_dtype_iterable(self, iterable):
        return all(self._is_valid_dtype_element(element)
                   for element in iterable)

    def __getitem__(self, key):
        # Specifications:
        # 1. key can only be list, slice, or int
        # 2. Return type is always a Column object
        if isinstance(key, list):
            data = [self.data[k] for k in key]
        elif isinstance(key, slice):
            data = self.data[key]
        else:
            data = [self.data[key]]
        retobj = Column(data)
        retobj.dtype = self.dtype
        return retobj

    def __setitem__(self, key, value):
        # Specifications:
        # 1. key can only be list, slice, or int
        # 2. value can be iterable or element depending upon key
        # 3. value should have dtype in {self.dtype, type(None)} at an
        #    element level
        if isinstance(key, slice):
            key = range(*key.indices(len(self)))

        if isinstance(key, list):
            if is_scalar(value):
                if self._is_valid_dtype_element(value):
                    # Assign scalar value to each element addressed by key
                    for k in key:
                        self.data[k] = value
                else:
                    msg = 'provided value must match existing dtype = {}'
                    raise ValueError(msg.format(self.dtype.__name__))
            else:
                # Provided value is iterable
                if self._is_valid_dtype_iterable(value):
                    length_value = get_length(value)
                    if len(key) == length_value:
                        for k, v in zip(key, value):
                            self.data[k] = v
                    else:
                        msg = ('provided address (len={}) and value (len={}) '
                               'must have the same length')
                        raise ValueError(msg.format(len(key), length_value))
                else:
                    msg = 'provided value must match existing dtype = {}'
                    raise ValueError(msg.format(self.dtype.__name__))
        elif isinstance(key, int):
            # In this case, provided value is always interpreted as a scalar
            if self._is_valid_dtype_element(value):
                self.data[key] = value
            else:
                msg = 'provided value must match existing dtype = {}'
                raise ValueError(msg.format(self.dtype.__name__))
        elif isinstance(key, float):
            msg = 'float address is not supported; please cast to int'
            raise KeyError(msg)
        else:
            msg = 'address must be int, slice, or list'
            raise KeyError(msg)

        # Re-infer dtype
        self.dtype = infer_dtype(self.data)
