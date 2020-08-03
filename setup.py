from setuptools import setup
from Cython.Build import cythonize

setup(name='relpomdp',
      packages=['relpomdp'],
      version='0.1',
      description='Relational POMDP',
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pomdp_py',
          'habitat-sim'
      ])

