/* Core model integration using TMAL code */

#include "Python.h"

#if PY_MAJOR_VERSION < 3
void initcore(void)
#else
void PyInit_core(void)
#endif
{
    /* stub initialization function needed by distutils on Windows */
}

int calc_model(int n_times, int n_preds, int n_tmals, double dt, 
               double **mvals, int **tmal_ints, double **tmal_floats)
{
    double deriv[100];  /* n_preds -- learn to malloc! */
    double y[100];
    double k2, dt2;
    int i, j, j0, i1, i2, i3, rki, opcode;    

    for (j0 = 0; j0 < n_times-2; j0 += 2) {
        for (rki = 0; rki < 2; rki++) {
            j = (rki == 0) ? j0 : j0 + 1;

            for (i = 0; i < n_preds; i++) {
                y[i] = (rki == 0) ? mvals[i][j] : y[i] + dt * deriv[i] / 2.0;
                deriv[i] = 0.0;
            }

            for (i = 0; i < n_tmals; i++) {
                opcode = tmal_ints[i][0];
                i1 = tmal_ints[i][1];
                i2 = tmal_ints[i][2];

                switch (opcode) {
                    case 0:  /* Node to node coupling */
                        if (i2 < n_preds && i1 < n_preds) {
                            deriv[i1] = deriv[i1] + (y[i2] - y[i1]) / tmal_floats[i][0];
                        }
                        else {
                            deriv[i1] = deriv[i1] + (mvals[i2][j] - y[i1]) / tmal_floats[i][0];
                        }
                        break;
                    case 1:  /* heat sink (coupling to fixed temperature) */
                        if (i1 < n_preds) {
                            deriv[i1] = deriv[i1] + (tmal_floats[i][0] - y[i1]) / tmal_floats[i][1];
                        }
                        break;
                    case 2:  /* precomputed heat */
                        if (i1 < n_preds) {
                            deriv[i1] = deriv[i1] + mvals[i2][j];
                        }
                        break;
                    case 3: /* active proportional heater */
                        dt2 =  tmal_floats[i][0] - ((i2 < n_preds) ? y[i2] : mvals[i2][j]);
                        i3 = tmal_ints[i][3];
                        if (dt2 > 0) {
                            mvals[i3][j] = tmal_floats[i][1] * dt2;
                            if (i1 < n_preds) {
                                deriv[i1] = deriv[i1] + mvals[i3][j];
                            }
                        } else {
                            mvals[i3][j] = 0.0;
                        }
                        break;
                    case 4: /* active thermostatic heater */
                        dt2 =  tmal_floats[i][0] - ((i2 < n_preds) ? y[i2] : mvals[i2][j]);
                        i3 = tmal_ints[i][3];
                        if (dt2 > 0) {
                            mvals[i3][j] = tmal_floats[i][1];
                            if (i1 < n_preds) {
                                deriv[i1] = deriv[i1] + mvals[i3][j];
                            }
                        } else {
                            mvals[i3][j] = 0.0;
                        }
                        break;
                    }
            }
        }

        for (i = 0; i < n_preds; i++) {
            k2 = dt * deriv[i];
            /* Note that the use of y[i] here is not 
               strictly correct for RK2, as it was 
               updated above in line 27 and thus 
               represents the value at mvals[i][j0+1]
               already--changing it now will break
               all models, however. See discussion at
               https://github.com/sot/xija/issues/72.
            */
            mvals[i][j0 + 1] = y[i] + k2 / 2.0;
            mvals[i][j0 + 2] = y[i] + k2;
        }
    }

    return 0;
}
