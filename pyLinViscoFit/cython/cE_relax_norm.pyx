cimport cython
cimport numpy as np
import numpy as np

from cython.parallel import prange
from libc.math cimport exp 

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def func(np.ndarray[double, ndim=1] time, 
                  np.ndarray[double, ndim=1] alpha_i,
                  np.ndarray[double, ndim=1] tau_i):
   
    cdef int i, j
    cdef np.ndarray [np.float64_t, ndim=1] out = np.empty_like(time, dtype=float)
    cdef double temp_sum  

    for i in prange(time.shape[0], nogil=True):
    #for i in range(time.shape[0], nogil=True):
        temp_sum = 0
        for j in range(alpha_i.shape[0]):
            temp_sum = temp_sum + alpha_i[j]*(1-exp(-time[i]/tau_i[j]))
        out[i] = 1 - temp_sum
    return out