"""
HoverAccelerationComp -- G3, maximum acceleration from hover.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Maximum Acceleration" p. 364.

Maximum rotor thrust tilted until its vertical component equals G.W.:

    acc = g sqrt((T_max/G.W.)^2 - 1)

T_max: maximum hover gross weight (hover ceiling plot, Figure 4.35).
Valid forward, rearward and sideward from hover. Requires T_max > G.W.

    T_max, GW --> acc_hover
"""

import numpy as np
import openmdao.api as om

G = 32.2       # ft/s^2, as printed in Chapter 5


class HoverAccelerationComp(om.ExplicitComponent):
    """Acceleration capability at hover, p. 364."""

    def setup(self):
        self.add_input('T_max', val=27800.0, units='lbf')
        self.add_input('GW', val=20000.0, units='lbf')
        self.add_output('acc_hover', val=30.0, units='ft/s**2')
        self.declare_partials('acc_hover', ['T_max', 'GW'])

    def compute(self, inputs, outputs):
        r = inputs['T_max'] / inputs['GW']
        outputs['acc_hover'] = G * np.sqrt(r ** 2 - 1.0)

    def compute_partials(self, inputs, J):
        T, W = inputs['T_max'], inputs['GW']
        r = T / W
        d = G * r / np.sqrt(r ** 2 - 1.0)
        J['acc_hover', 'T_max'] = d / W
        J['acc_hover', 'GW'] = -d * T / W ** 2
