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

void dTdt(int j, int n_preds, int n_tmals, int **tmal_ints, double **tmal_floats,
          double **mvals, double *deriv, double *y)
{
    int i, i1, i2, i3, opcode;
    double dt2;
    
    for (i = 0; i < n_preds; i++) {
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
                dt2 = tmal_floats[i][0] - ((i2 < n_preds) ? y[i2] : mvals[i2][j]);
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

int calc_model(int n_times, int n_preds, int n_tmals, double dt, 
               double **mvals, int **tmal_ints, double **tmal_floats)
{
    double *y, *yh, *deriv;
    double k1, k2;
    int i, j;    

    deriv = (double *)malloc(n_preds*sizeof(double));
    y = (double *)malloc(n_preds*sizeof(double));
    yh = (double *)malloc(n_preds*sizeof(double));

    for (j = 0; j < n_times-2; j += 2) {
        
        for (i = 0; i < n_preds; i++) {
            y[i] = mvals[i][j];
        }

        dTdt(j, n_preds, n_tmals, tmal_ints, tmal_floats, mvals, deriv, y);
        
        for (i = 0; i < n_preds; i++) {
            k1 = dt * deriv[i];
            yh[i] = y[i] + 0.5*k1;
        }

        dTdt(j+1, n_preds, n_tmals, tmal_ints, tmal_floats, mvals, deriv, yh);

        for (i = 0; i < n_preds; i++) {
            k2 = dt * deriv[i];
            mvals[i][j + 1] = y[i] + 0.5*k2;
            mvals[i][j + 2] = y[i] + k2;
        }
    }

    free(y);
    free(yh);
    free(deriv);
   
    return 0;
}
