from setuptools import setup

setup(name='dframe',
      version='0.1.6',
      description='Easy-to-use, indexless dataframe data structure in Python.',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 2.7'
      ],
      url='http://github.com/ankur-gupta/dframe',
      author='Ankur Gupta',
      author_email='ankur@perfectlyrandom.org',
      keywords='dataframe indexless',
      license='',
      packages=['dframe', 'dframe.array', 'dframe.compat',
                'dframe.dataframe', 'dframe.dtypes', 'dframe.missing',
                'dframe.scalar'],
      install_requires=[
          'future',
          'pandas',
          'numpy',
          'prettytable',
          'python-dateutil'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False)
