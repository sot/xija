import numpy as np
cimport numpy as np
cimport cython

print "Here in core.pyx3!"

DTYPE = np.int
ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t
ctypedef np.float32_t FLOAT32_t

@cython.boundscheck(False)
def dT_dt(int j,
          np.ndarray[FLOAT_t, ndim=1] y,
          np.ndarray[FLOAT_t, ndim=1] deriv,
          float dt,
          int n_preds,
          np.ndarray[FLOAT_t, ndim=2] mvals,
          np.ndarray[FLOAT_t, ndim=1] parvals,
          np.ndarray[INT_t, ndim=2] mults,
          np.ndarray[INT_t, ndim=2] heats,
          np.ndarray[INT_t, ndim=2] heatsinks          ):

    cdef int i    
    cdef int n_mults = mults.shape[0]
    cdef int n_heats = heats.shape[0]
    cdef int n_heatsinks = heatsinks.shape[0]
    cdef int i1
    cdef int i2

    for i in range(n_preds):
        deriv[i] = 0.0

    # Couplings with other nodes
    for i in range(n_mults):
        i1 = mults[i, 0] 
        i2 = mults[i, 1]
        if i2 < n_preds and i1 < n_preds:
            deriv[i1] = deriv[i1] + (y[i2] - y[i1]) / parvals[mults[i, 2]]
        else:
            deriv[i1] = deriv[i1] + (mvals[i2, j] - y[i1]) / parvals[mults[i, 2]]

    # Direct heat inputs (e.g. Solar, Earth)
    for i in range(n_heats):
        i1 = heats[i, 0]
        if i1 < n_preds:
            i2 = heats[i, 1]
            deriv[i1] = deriv[i1] + mvals[i2, j]

    # Couplings to heat sinks
    for i in range(n_heatsinks):
        i1 = heatsinks[i, 0]
        if i1 < n_preds:
            deriv[i1] = deriv[i1] + (parvals[heatsinks[i, 1]] - y[i1]) / parvals[heatsinks[i, 2]]

    return deriv

@cython.boundscheck(False)
def calc_model(np.ndarray[INT_t, ndim=1] indexes,
               py_dt, py_n_preds,
               np.ndarray[FLOAT_t, ndim=2] mvals,
               np.ndarray[FLOAT_t, ndim=1] parvals,
               np.ndarray[INT_t, ndim=2] mults,
               np.ndarray[INT_t, ndim=2] heats,
               np.ndarray[INT_t, ndim=2] heatsinks):

    deriv = np.zeros(py_n_preds)
    y = np.zeros(py_n_preds)
    cdef int n_preds = py_n_preds
    cdef float dt = py_dt
    cdef int j    

    for j in indexes:
        # 2nd order Runge-Kutta (do 4th order later as needed)
        for i in range(n_preds):
            y[i] = mvals[i, j]

        k1 = dt * dT_dt(j, y, deriv, dt, n_preds,
                        mvals, parvals, mults, heats, heatsinks)
        k2 = dt * dT_dt(j+1, y + k1 / 2.0, deriv, dt, n_preds, 
                        mvals, parvals, mults, heats, heatsinks)

        for i in range(n_preds):
            mvals[i, j+1] = y[i] + k2[i] / 2.0
            mvals[i, j+2] = y[i] + k2[i]
