from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(name='relpomdp',
      packages=find_packages(),
      version='0.1',
      description='Relational POMDP',
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pomdp_py',
          # 'habitat-sim'
      ])
