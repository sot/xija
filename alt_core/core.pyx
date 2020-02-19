import numpy as np
cimport numpy as np

print "Here in core.pyx2!"

DTYPE = np.int
ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t
ctypedef np.float32_t FLOAT32_t

def calc_model(np.ndarray[INT_t, ndim=1] indexes,
               py_dt, py_n_preds,
               np.ndarray[FLOAT_t, ndim=2] mvals,
               np.ndarray[FLOAT_t, ndim=1] parvals,
               np.ndarray[INT_t, ndim=2] mults,
               np.ndarray[INT_t, ndim=2] heats,
               np.ndarray[INT_t, ndim=2] heatsinks):

    cdef float dt = py_dt
    cdef int n_preds = py_n_preds
    cdef int n_mults = mults.shape[0]
    cdef int n_heats = heats.shape[0]
    cdef int n_headsinks = heatsinks.shape[0]
    cdef int i    

    deriv = np.zeros(n_preds)
    def dT_dt(j, y, np.ndarray[INT_t, ndim=2] mults2):
        for i in range(n_preds):
            deriv[i] = 0.0

        # Couplings with other nodes
        for i in range(n_mults):
            i1 = mults2[i, 0] 
            i2 = mults[i, 1]
            tau = parvals[mults[i, 2]]
            if i2 < n_preds and i1 < n_preds:
                deriv[i1] += (y[i2] - y[i1]) / tau
            else:
                deriv[i1] += (mvals[i2, j] - y[i1]) / tau

        # Direct heat inputs (e.g. Solar, Earth)
        for i in range(n_heats):
            i1 = heats[i, 0]
            if i1 < n_preds:
                i2 = heats[i, 1]
                deriv[i1] += mvals[i2, j]

        # Couplings to heat sinks
        for i in range(n_heatsinks):
            i1 = heatsinks[i, 0]
            if i1 < n_preds:
                T = parvals[heatsinks[i, 1]]
                tau = parvals[heatsinks[i, 2]]
                deriv[i1] += (T - y[i1]) / tau

        return deriv

    for j in indexes:
        # 2nd order Runge-Kutta (do 4th order later as needed)
        y = mvals[:n_preds, j]
        k1 = dt * dT_dt(j, y, mults)
        k2 = dt * dT_dt(j+1, y + k1 / 2.0)
        mvals[:n_preds, j+1] = y + k2 / 2.0
        mvals[:n_preds, j+2] = y + k2
