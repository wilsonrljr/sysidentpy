from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import itertools
except ImportError:
    print('itertools is required during installation')
    sys.exit(1)

import sys_identfy

setup(name='sys_identify',
      version='0.5',
      description='Open source system identification library in Python',
      long_description=" Take a look into https://github.com/wilsonrljr",
      author='Wilson Luan Samuel Samir',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='wilsonrljr@outlook.com',
      url='https://github.com/wilsonrljr/sys_identify',
      classifiers=[
              'Programming Language :: Python :: 3.6',
              'Development Status :: 2 - Beta',
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: System Identification',
              'Topic :: Software Development :: Libraries'
              ]
      )