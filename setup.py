#!/usr/bin/env python

try:
    from setuptools import setup, find_packages
except ImportError:
    raise RuntimeError('setuptools is required')

import versioneer

DESCRIPTION = ('Pyvisco is a Python library that supports Prony series ' + 
               'identification for linear viscoelastic material models.')

LONG_DESCRIPTION = """
Pyvisco is a Python library that supports the identification of Prony series 
parameters for linear viscoelastic materials described by a Generalized Maxwell 
model. The necessary material model parameters are identified by fitting a Prony 
series to the experimental measurement data in either the frequency-domain 
(via Dynamic Mechanical Thermal Analysis) or time-domain (via relaxation 
measurements). Pyvisco performs the necessary data processing of the 
experimental measurements, mathematical operations, and curve-fitting routines 
to identify the Prony series parameters. These parameters are used in subsequent 
Finite Element simulations involving linear viscoelastic material models that 
accurately describe the mechanical behavior of polymeric materials such as 
encapsulants and backsheets of PV modules. An optional minimization routine is 
included to reduce the number of Prony elements. This routine is helpful in 
large Finite Element simulations where reducing the computational complexity of 
the linear viscoelastic material models can shorten the simulation time.

Documentation: https://pyvisco.readthedocs.io  
Source code: https://github.com/NREL/pyvisco
"""

DISTNAME = 'pyvisco'
AUTHOR = "Martin Springer"
AUTHOR_EMAIL = 'martin.springer@nrel.gov'
MAINTAINER = "Martin Springer"
MAINTAINER_EMAIL = 'martin.springer@nrel.gov'
LICENSE = 'BSD-3'
URL = 'https://github.com/NREL/pyvisco'

INSTALL_REQUIRES = [
    'jupyter>=1.0.0',
    'matplotlib>=3.5.1',
    'numpy>=1.22.2',
    'pandas>=1.4.1',
    'scipy>=1.8.0',
    'ipython>=8.0.1',
    'ipywidgets>=7.6.5',
    'ipympl>=0.8.8',
    'voila>=0.3.1',
    'xlrd>=2.0.1',
    'Markdown>=3.3.6',
    'jinja2==2.11.3',
    'markupsafe==2.0.1',
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

KEYWORDS = [
    'curve-fitting',
    'material-modelling',
    'viscoelasticity',
    'prony'
]

PROJECT_URLS = {
    "Documentation": "https://pyvisco.readthedocs.io/",
    "Source Code": "https://github.com/NREL/pyvisco/",
    "Web application" : "https://pyvisco.herokuapp.com/"
}

PACKAGES = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])

setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    use_scm_version=True,
    packages=PACKAGES,
    keywords=KEYWORDS,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    #tests_require=TESTS_REQUIRE,
    setup_requires=SETUP_REQUIRES,
    ext_modules=[],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    url=URL,
    project_urls=PROJECT_URLS,
    include_package_data=True,
)
