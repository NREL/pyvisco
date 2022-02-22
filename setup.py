#!/usr/bin/env python

try:
    from setuptools import setup, find_packages
except ImportError:
    raise RuntimeError('setuptools is required')

DESCRIPTION = ('pyvisco is a python library that supports Prony series' + 
               'identification for linear viscoelastic material models.')

LONG_DESCRIPTION = """
pyvisco is a collection of functions to identify Prony series parameters of 
linear viscoelastic materials from measurements in either the time 
(relaxation tests) or frequency domain (DMTA).
Documentation: https://pyvisco.readthedocs.io
Source code: https://github.com/martin-springer/LinViscoFit
"""

DISTNAME = 'pyvisco'
MAINTAINER = "Martin Springer"
MAINTAINER_EMAIL = 'martinspringer.ms@gmail.com'
LICENSE = 'GNU'
URL = 'https://github.com/martin-springer/LinViscoFit'

INSTALL_REQUIRES = [
    'jupyter',
    'matplotlib',
    'numpy',
    'pandas',
    'scipy',
    'ipython',
    'ipywidgets',
    'ipympl',
    'voila',
    'xlrd',
    'markdown',
]

DOCS_REQUIRE = [
    'sphinx == 4.3.1', 'm2r2',
]

EXTRAS_REQUIRE = {
    'doc': DOCS_REQUIRE
}

EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

SETUP_REQUIRES = ['setuptools_scm']

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering'
]

PACKAGES = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])

setup(
    name=DISTNAME,
    use_scm_version=True,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    #tests_require=TESTS_REQUIRE,
    setup_requires=SETUP_REQUIRES,
    ext_modules=[],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    url=URL,
    include_package_data=True,
)