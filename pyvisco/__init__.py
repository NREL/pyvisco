#!/usr/bin/env python3

"""Collection of submodules to identify Prony series parameters of linear 
viscoelastic materials from measurements in either the time (relaxation tests) 
or frequency domain (DMTA).
"""

__version__ = '0.9.0'

from . import load
from . import shift
from . import master
from . import prony
from . import opt
from . import verify
from . import out
from . import _version
__version__ = _version.get_versions()['version']
