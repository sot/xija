def calc_model(indexes, dt, n_preds, mvals, parvals, mults, heats, heatsinks):
    def dT_dt(j, y):
        deriv[:] = 0.0

        # Couplings with other nodes
        for i in xrange(len(mults)):
            i1 = mults[i, 0] 
            i2 = mults[i, 1]
            tau = parvals[mults[i, 2]]
            if i2 < n_preds and i1 < n_preds:
                deriv[i1] += (y[i2] - y[i1]) / tau
            else:
                deriv[i1] += (mvals[i2, j] - y[i1]) / tau

        # Direct heat inputs (e.g. Solar, Earth)
        for i in xrange(len(heats)):
            i1 = heats[i, 0]
            if i1 < n_preds:
                i2 = heats[i, 1]
                deriv[i1] += mvals[i2, j]

        # Couplings to heat sinks
        for i in xrange(len(heatsinks)):
            i1 = heatsinks[i, 0]
            if i1 < n_preds:
                T = parvals[heatsinks[i, 1]]
                tau = parvals[heatsinks[i, 2]]
                deriv[i1] += (T - y[i1]) / tau

        return deriv

    deriv = np.zeros(n_preds)
    for j in indexes:
        # 2nd order Runge-Kutta (do 4th order later as needed)
        y = mvals[:n_preds, j]
        k1 = dt * dT_dt(j, y)
        k2 = dt * dT_dt(j+1, y + k1 / 2.0)
        mvals[:n_preds, j+1] = y + k2 / 2.0
        mvals[:n_preds, j+2] = y + k2
