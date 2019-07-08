"""SysIdentPy: System Identification Library for Python
is open-source software for mathematics,
science, and engineering regarding the use of NARX models
built on top of numpy and is distributed under the 3-Clause BSD license.

"""

import os
import sys
import subprocess
import textwrap
import warnings
import sysconfig
from distutils.version import LooseVersion
from __future__ import print_function
import sys
from setuptools import setup, find_packages
import builtins
import platform



if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >= 3.5 required.")

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

DISTNAME = 'sys-identpy'
DESCRIPTION = 'Open source system identification library in Python'
LONG_DESCRIPTION = 'Take a look into https://github.com/wilsonrljr'
MAINTAINER = 'Wilson Rocha Lacerda Junior'
MAINTAINER_EMAIL = 'wilsonrljr@outlook.com'
URL = 'http://sys-identpy.org'
DOWNLOAD_URL = 'https://pypi.org/project/sys-identpy/#files'
LICENSE = 'new BSD'
PROJECT_URLS = {
    'Source Code': 'https://github.com/wilsonrljr/sys-identpy'
}

import sys_identfy

VERSION = sys_identfy.__version__

if platform.python_implementation() == 'PyPy':
    ITERTOOLS_MIN_VERSION = '2.6'
    NUMPY_MIN_VERSION = '1.14.0'
else:
    ITERTOOLS_MIN_VERSION = '2.6'
    NUMPY_MIN_VERSION = '1.11.0'


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    project_urls=PROJECT_URLS,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    classifiers=[
                            'Intended Audience :: Science/Research',
                            'Intended Audience :: Developers',
                            'Programming Language :: Python :: 3.5',
                            'Programming Language :: Python :: 3.6',
                            'Programming Language :: Python :: 3.7',
                            'Development Status :: 3 - Alpha',
                            'Intended Audience :: Science/Research',
                            'Topic :: Scientific/Engineering',
                            'Topic :: Software Development'
                            'Operating System :: Microsoft :: Windows',
                            'Operating System :: POSIX',
                            'Operating System :: Unix',
                            'Operating System :: MacOS',
                            ],
                    python_requires=">=3.5",
                    install_requires=[
                                        'numpy>={}'.format(NUMPY_MIN_VERSION),
                                        'itertools>={}'.format(ITERTOOLS_MIN_VERSION)
                                    ],
                    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
