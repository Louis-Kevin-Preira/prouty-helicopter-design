"""
AutorotationLimitComp -- G6, autorotative limit of the banked turn.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Return-to-Target Maneuver" p. 371 ("if the speed drops below the
autorotative limit ... a powered, steady turn at this speed should be used").

The limit is the speed at which the zero-torque rotor at maximum thrust can
no longer hold the weight in the turn: n_auto(V_sw) = 1. The table n_auto on
V_grid (TurnDecelerationGroup) is an Akima SpeedTable; the root is found by
Newton and differentiated by the implicit function theorem:
    dV_sw/dn_tab = -(dn/dn_tab) / (dn/dV)

    n_tab (M,) --> V_sw
"""

import numpy as np
import openmdao.api as om

from prouty.special_performance._table import SpeedTable


class AutorotationLimitComp(om.ExplicitComponent):
    """Speed where the autorotative turn load factor falls to 1, p. 371."""

    def initialize(self):
        self.options.declare('V_grid', types=np.ndarray, desc='table speeds, ft/s, increasing')

    def setup(self):
        M = len(self.options['V_grid'])
        self.add_input('n_tab', val=np.linspace(0.8, 2.0, M))
        self.add_output('V_sw', val=50.0, units='ft/s')
        self.declare_partials('V_sw', 'n_tab')

    def _solve(self, n_tab):
        Vg = self.options['V_grid']
        table = SpeedTable(Vg, n_tab)
        k = max(int(np.argmax(n_tab >= 1.0)), 1)      # first grid speed above the limit
        V = 0.5 * (Vg[k - 1] + Vg[k])
        for _ in range(50):
            n, dn, _ = table(V)
            step = (n - 1.0) / dn
            V -= step
            if abs(step) < 1e-12 * V:
                break
        return V, table(V)

    def compute(self, inputs, outputs):
        outputs['V_sw'] = self._solve(inputs['n_tab'])[0]

    def compute_partials(self, inputs, J):
        _, (_, dn, dn_tab) = self._solve(inputs['n_tab'])
        J['V_sw', 'n_tab'] = -dn_tab / dn
