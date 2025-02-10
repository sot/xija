# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from xija import tmal
from xija.component.base import ModelComponent


class PrecomputedHeatPower(ModelComponent):
    """Component that provides static (precomputed) direct heat power input"""

    def update(self):
        self.mvals = self.dvals
        self.tmal_ints = (
            tmal.OPCODES["precomputed_heat"],
            self.node.mvals_i,  # dy1/dt index
            self.mvals_i,
        )
        self.tmal_floats = ()

    @staticmethod
    def linear(days, k_inv):
        return days / k_inv

    @staticmethod
    def exp(days, tau):
        return 1 - np.exp(-days / tau)


class ActiveHeatPower(ModelComponent):
    """Component that provides active heat power input which depends on
    current or past computed model values

    Parameters
    ----------

    Returns
    -------

    """
