#!/usr/bin/env python3

"""Collection of functions to identify Prony series parameters of linear 
viscoelastic materials from measurements in either the time (relaxation tests) 
or frequency domain (DMTA).
"""

__version__ = '0.0.1'

from . import load
from . import shift
from . import master
from . import prony
from . import opt
from . import verify