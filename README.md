# dframe 

`dframe` is an indexless implementation of the dataframe data structure. 
It focuses on ease-of-use over computational efficiency.

## Installation
[`dframe`](https://pypi.python.org/pypi/dframe) can be installed using `pip`.

```
pip install --upgrade dframe
```


## Try it out

`dframe` should be intuitive to use. Try out these commands yourself to get a feel for it. More features will be added soon. Please file feature requests/bugs as [issues](https://github.com/ankur-gupta/dframe/issues). 

```python
import dframe as df
x = df.DataFrame({'a': [1, 2, 3, 4], 'b': ['a', 'b', 'c', 'd']})

# Try out indexing
print(x[0, 0])  # First row, first column
print(x[1])  # Second column as a list
print(x[[1]])  # Second column as a DataFrame
print(x['a'])  # First column as a list
print(x[['a']])  # First column as a DataFrame
print(x[0, :])  # First row
print(x[0:2, :])  # First two rows
print(x[[1, 3], :])  # Second and fourth rows

print(x[::-1, :])  # Reverse order of rows
print(x[::-1])  # Reverse order of columns

# Set all values in column 'a'
x['a'] = 0

# Create a new column with a missing value in it
x['c'] = [1.0, 2.3, None, 9.0]
```

