"""
LowSpeedPowerComp -- G5, level flight power between hover and the lowest
forward flight trim.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Optimum Takeoff Procedure at High Gross Weights" p. 368
(h.p. level at the rotation speed); Chapter 4 hover (pp. 308-312) and
forward flight (pp. 317-319) power.

The Chapter 3 trim is a forward flight model (fuselage downwash singular in
hover), so between 0 and V_b the power is a cubic Hermite joining

    P(0)   = P_hover,       dP/dV(0)   = 0      (P even in V)
    P(V_b) = P_b,           dP/dV(V_b) = dP_b   (forward flight chain)

This segment is an interpolation, not a flight mechanics model.

    V (nn,), V_b, P_hover, P_b, dP_b --> P_level (nn,)       0 <= V <= V_b
"""

import numpy as np
import openmdao.api as om


class LowSpeedPowerComp(om.ExplicitComponent):
    """Hermite join of hover and forward flight power, p. 368."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('V', val=30.0 * np.ones(nn), units='ft/s')
        self.add_input('V_b', val=40.0 * 1.6878, units='ft/s')
        self.add_input('P_hover', val=2300.0, units='hp')
        self.add_input('P_b', val=1300.0, units='hp')
        self.add_input('dP_b', val=-10.0, units='hp*s/ft')
        self.add_output('P_level', val=np.ones(nn), units='hp')
        self.declare_partials('P_level', 'V', rows=ar, cols=ar)
        self.declare_partials('P_level', ['V_b', 'P_hover', 'P_b', 'dP_b'], rows=ar, cols=zero)

    @staticmethod
    def _basis(s):
        return (2 * s ** 3 - 3 * s ** 2 + 1, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2,
                6 * s ** 2 - 6 * s, -6 * s ** 2 + 6 * s, 3 * s ** 2 - 2 * s)

    def compute(self, inputs, outputs):
        Vb = inputs['V_b']
        h00, h01, h11, _, _, _ = self._basis(inputs['V'] / Vb)
        outputs['P_level'] = (inputs['P_hover'] * h00 + inputs['P_b'] * h01
                              + Vb * inputs['dP_b'] * h11)

    def compute_partials(self, inputs, J):
        V, Vb, P0, Pb, dPb = (inputs[k] for k in ('V', 'V_b', 'P_hover', 'P_b', 'dP_b'))
        s = V / Vb
        h00, h01, h11, d00, d01, d11 = self._basis(s)
        dP_ds = P0 * d00 + Pb * d01 + Vb * dPb * d11
        J['P_level', 'V'] = dP_ds / Vb
        J['P_level', 'V_b'] = -dP_ds * s / Vb + dPb * h11
        J['P_level', 'P_hover'] = h00
        J['P_level', 'P_b'] = h01
        J['P_level', 'dP_b'] = Vb * h11
