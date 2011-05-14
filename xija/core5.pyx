"""Core model integration using TMAL code
"""

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int
ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t
ctypedef np.int32_t INT32_t
ctypedef np.float32_t FLOAT32_t

@cython.boundscheck(False)
def calc_model(np.ndarray[INT_t, ndim=1] indexes,
               py_dt, py_n_preds,
               np.ndarray[FLOAT_t, ndim=2] mvals,
               np.ndarray[INT32_t, ndim=2] tmal_ints,
               np.ndarray[FLOAT32_t, ndim=2] tmal_floats):

    cdef np.ndarray[FLOAT_t] deriv = np.zeros(py_n_preds)
    cdef np.ndarray[FLOAT_t] y = np.zeros(py_n_preds)
    cdef np.ndarray[FLOAT_t] y1 = np.zeros(py_n_preds)

    cdef int n_preds = py_n_preds
    cdef int n_tmals = tmal_ints.shape[0]
    cdef float dt = py_dt
    cdef float k2

    cdef int i    
    cdef int j    
    cdef int j0
    cdef int i1
    cdef int i2
    cdef int rki
    cdef int opcode

    for j0 in indexes:
        # print '========== j=', j0, '==============='
        for rki in range(2):
            # 2nd order Runge-Kutta (do 4th order later as needed)
            if rki == 0:
                j = j0
            else:
                j = j0 + 1

            for i in range(n_preds):
                y[i] = mvals[i, j] if rki == 0 else y[i] + dt * deriv[i] / 2.0
                deriv[i] = 0.0

            for i in range(n_tmals):
                opcode = tmal_ints[i, 0]
                i1 = tmal_ints[i, 1]
                i2 = tmal_ints[i, 2]

                if opcode == 0:  # coupling
                    if i2 < n_preds and i1 < n_preds:
                        deriv[i1] = deriv[i1] + (y[i2] - y[i1]) / tmal_floats[i, 0]
                    else:
                        deriv[i1] = deriv[i1] + (mvals[i2, j] - y[i1]) / tmal_floats[i, 0]

                elif opcode == 1:  # heat sink
                    if i1 < n_preds:
                        deriv[i1] = deriv[i1] + (tmal_floats[i, 0] - y[i1]) / tmal_floats[i, 1]

                elif opcode == 2:  # precomputed heat
                    if i1 < n_preds:
                        deriv[i1] = deriv[i1] + mvals[i2, j]

        for i in range(n_preds):
            k2 = dt * deriv[i]
            mvals[i, j0+1] = y[i] + k2 / 2.0
            mvals[i, j0+2] = y[i] + k2
