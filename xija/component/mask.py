# Licensed under a 3-clause BSD style license - see LICENSE.rst
import operator

import numpy as np
try:
    from Ska.Matplotlib import plot_cxctime
except ImportError:
    pass

from .base import ModelComponent


class Mask(ModelComponent):
    """Create object with a ``mask`` attribute corresponding to
      node.dvals op val
    where op is a binary operator in operator module that returns a np mask
      "ge": >=
      "gt": >
      "le": <=
      "lt": <
      "eq": ==
      "ne" !=
    """
    def __init__(self, model, node, op, val, min_=-1e38, max_=1e38):
        ModelComponent.__init__(self, model)
        # Usually do self.node = model.get_comp(node) right away.  But here
        # allow for a forward reference to a not-yet-existent node and check
        # only when self.mask is actually used. This allows for masking in a
        # node based on data for that same node.
        self.node = node
        self.op = op
        self.model = model
        self.add_par('val', val, min=min_, max=max_, frozen=True)
        self.cache_key = None

    def compute_cache_key(self):
        return self.val

    def compute_mask(self):
        mask = getattr(operator, self.op)(self.node.dvals, self.val)
        return mask

    @property
    def mask(self):
        if not isinstance(self.node, ModelComponent):
            self.node = self.model.get_comp(self.node)
        # cache latest version of mask
        cache_key = self.compute_cache_key()
        if cache_key != self.cache_key:
            self.cache_key = cache_key
            self._mask = self.compute_mask()
        return self._mask

    def __str__(self):
        return "mask__{}_{}".format(self.node, self.op)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        y = np.where(self.mask, 1, 0)
        if lines:
            lines[0].set_data(self.model_plotdate, y)
        else:
            plot_cxctime(self.model.times, y, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_ylim(-0.1, 1.1)
            ax.set_title('{}: data'.format(self.name))


class MaskBox(Mask):
    """Create object with a ``mask`` attribute corresponding to
      val0 < node.dvals < val1
    """
    def __init__(self, model, node, val0, val1, min_=-1000, max_=1000):
        ModelComponent.__init__(self, model)
        # Usually do self.node = model.get_comp(node) right away.  But here
        # allow for a forward reference to a not-yet-existent node and check
        # only when self.mask is actually used. This allows for masking in a
        # node based on data for that same node.
        self.node = node
        self.model = model
        self.add_par('val0', val0, min=min_, max=max_, frozen=True)
        self.add_par('val1', val1, min=min_, max=max_, frozen=True)
        self.cache_key = None

    def compute_cache_key(self):
        return (self.val0, self.val1)

    def compute_mask(self):
        dvals = self.node.dvals
        mask = (self.val0 < dvals) & (dvals < self.val1)
        return mask

    def __str__(self):
        return "maskbox__{}".format(self.node)


