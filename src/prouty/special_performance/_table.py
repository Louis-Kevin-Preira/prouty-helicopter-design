"""
Akima table on a fixed speed grid whose values are model outputs (G6).

The tables of the return-to-target maneuver (n_turn, V_dot, acc on V_grid)
are computed by the rotor groups, so the interpolant needs derivatives with
respect to its values as well as to the query speed. InterpND provides both;
d/dvalues is switched on the way MetaModelStructuredComp does it.

delta_x smooths the absolute values in Akima's slope weights. Without it the
interpolant is not differentiable with respect to the values where two
consecutive slopes are equal (flat or straight parts of a table, e.g. n_turn
= 2.0 above 60 kt): finite differences then jump by orders of magnitude.
"""

import numpy as np
from openmdao.components.interp_util.interp import InterpND


class SpeedTable:
    """value(V), d/dV and d/dvalues of an Akima table on V_grid."""

    def __init__(self, V_grid, values, delta_x=1e-3):
        self._interp = InterpND(method='akima', points=(V_grid,), values=values,
                                extrapolate=True, delta_x=delta_x)
        self._interp._compute_d_dvalues = True
        self._interp.table._compute_d_dvalues = True

    def __call__(self, V):
        val, dV = self._interp.interpolate(np.array([[V]]), compute_derivative=True)
        return val[0], dV[0, 0], np.array(self._interp._d_dvalues).reshape(-1)
