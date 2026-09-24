"""
AutorotationLimitComp -- G6, autorotative limit of the banked turn.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Return-to-Target Maneuver" p. 371 ("if the speed drops below the
autorotative limit ... a powered, steady turn at this speed should be used").

The limit is the speed at which the zero-torque rotor at maximum thrust can
no longer hold the weight in the turn: n_auto(V_sw) = 1. The table n_auto on
V_grid (TurnDecelerationGroup) is interpolated as in ReturnToTargetComp and
the root found by Newton in complex-safe arithmetic (partials by complex step).

    n_tab (M,) --> V_sw
"""

import numpy as np
import openmdao.api as om


class AutorotationLimitComp(om.ExplicitComponent):
    """Speed where the autorotative turn load factor falls to 1, p. 371."""

    def initialize(self):
        self.options.declare('V_grid', types=np.ndarray, desc='table speeds, ft/s, increasing')

    def setup(self):
        M = len(self.options['V_grid'])
        self.add_input('n_tab', val=np.linspace(0.8, 2.0, M))
        self.add_output('V_sw', val=50.0, units='ft/s')
        self.declare_partials('V_sw', 'n_tab', method='cs')

    def compute(self, inputs, outputs):
        xg, n = self.options['V_grid'], inputs['n_tab']
        m = np.gradient(n, xg)

        def f(v):
            i = int(np.clip(np.searchsorted(xg, v.real) - 1, 0, len(xg) - 2))
            h = xg[i + 1] - xg[i]
            s = (v - xg[i]) / h
            val = ((2 * s ** 3 - 3 * s ** 2 + 1) * n[i] + (s ** 3 - 2 * s ** 2 + s) * h * m[i]
                   + (-2 * s ** 3 + 3 * s ** 2) * n[i + 1] + (s ** 3 - s ** 2) * h * m[i + 1])
            der = ((6 * s ** 2 - 6 * s) * n[i] + (3 * s ** 2 - 4 * s + 1) * h * m[i]
                   + (-6 * s ** 2 + 6 * s) * n[i + 1] + (3 * s ** 2 - 2 * s) * h * m[i + 1]) / h
            return val, der

        k = int(np.argmax(n.real >= 1.0))            # first grid speed above the limit
        v = xg[max(k - 1, 0)] + 0.5 * (xg[k] - xg[max(k - 1, 0)])
        for _ in range(50):
            val, der = f(v)
            v = v - (val - 1.0) / der
        outputs['V_sw'] = v
