from distutils.util import convert_path
from typing import Dict

from setuptools import setup, find_packages




version_dict = {}  # type: Dict[str, str]
with open(convert_path('framework/version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='framework',
    version=version_dict['__version__'],
    description='',
    long_description='',
    classifiers=['Programming Language :: Python :: 3.6'],
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'gym',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'ase',
        'schnetpack',
        'mpi4py',
    ],
    zip_safe=False,
    test_suite='pytest',
    tests_require=['pytest'],
)
