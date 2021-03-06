import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int
ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t
ctypedef np.float32_t FLOAT32_t

@cython.boundscheck(False)
def calc_model(np.ndarray[INT_t, ndim=1] indexes,
               py_dt, py_n_preds,
               np.ndarray[FLOAT_t, ndim=2] mvals,
               np.ndarray[FLOAT_t, ndim=1] parvals,
               np.ndarray[INT_t, ndim=2] mults,
               np.ndarray[INT_t, ndim=2] heats,
               np.ndarray[INT_t, ndim=2] heatsinks):

    cdef np.ndarray[FLOAT_t] deriv = np.zeros(py_n_preds)
    cdef np.ndarray[FLOAT_t] y = np.zeros(py_n_preds)
    cdef np.ndarray[FLOAT_t] y1 = np.zeros(py_n_preds)

    cdef int n_preds = py_n_preds
    cdef float dt = py_dt
    cdef float k2

    cdef int i    
    cdef int j    
    cdef int j0
    cdef int n_mults = mults.shape[0]
    cdef int n_heats = heats.shape[0]
    cdef int n_heatsinks = heatsinks.shape[0]
    cdef int i1
    cdef int i2
    cdef int rki

    for j0 in indexes:
        # print '========== j=', j0, '==============='
        for rki in range(2):
            # 2nd order Runge-Kutta (do 4th order later as needed)
            if rki == 0:
                j = j0
            else:
                j = j0 + 1

            for i in range(n_preds):
                if rki == 0:
                    y[i] = mvals[i, j]
                else:
                    y[i] = y[i] + dt * deriv[i] / 2.0
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
                    # print "Deriv[{0}] += {1:.3f} (heat P)".format(i, mvals[i2, j])
                    deriv[i1] = deriv[i1] + mvals[i2, j]

            # Couplings to heat sinks
            for i in range(n_heatsinks):
                i1 = heatsinks[i, 0]
                if i1 < n_preds:
                    # print "Deriv[{0}] += {1} {2} {3} (heat Ue*Te)".format(
                    #   i, parvals[heatsinks[i,1]] / parvals[heatsinks[i,2]],
                    #   parvals[heatsinks[i,1]] , parvals[heatsinks[i,2]] )
                    # print "Deriv[{0}] += {1} (Ue*Ti)".format(i, -y[i1] / parvals[heatsinks[i,2]])
                    deriv[i1] = deriv[i1] + (parvals[heatsinks[i, 1]] - y[i1]) / parvals[heatsinks[i, 2]]

        for i in range(n_preds):
            k2 = dt * deriv[i]
            mvals[i, j0+1] = y[i] + k2 / 2.0
            mvals[i, j0+2] = y[i] + k2
