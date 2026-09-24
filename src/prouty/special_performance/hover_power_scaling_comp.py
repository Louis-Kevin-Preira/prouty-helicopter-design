"""
HoverPowerScalingComp -- G6, hover power at the effective weight n G.W.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1 momentum theory (induced power ~ T^(3/2)); Chapter 4 hover chain.

The Chapter 4 hover chain cannot trim its tail rotor much above the example's
hover capability (about 32,000 lb IGE), while the powered turn of p. 371
needs the hover anchor of the low-speed power join at n_p G.W. (up to about
1.8 x 20,000 lb). The anchor is scaled from the hover power at the actual
weight, as for an induced-dominated rotor:

    P_hover(n G.W.) = P_hover(G.W.) n^(3/2)

It weighs about 20 % of P_level at the autorotative limit (Hermite weight of
the V = 0 end at 29 kt, V_b = 40 kt).

    P_hover, n --> P_hover_eff
"""

import numpy as np
import openmdao.api as om


class HoverPowerScalingComp(om.ExplicitComponent):
    """Momentum scaling of hover power with load factor."""

    def setup(self):
        self.add_input('P_hover', val=2300.0, units='hp')
        self.add_input('n', val=1.0)
        self.add_output('P_hover_eff', val=2300.0, units='hp')
        self.declare_partials('P_hover_eff', ['P_hover', 'n'])

    def compute(self, inputs, outputs):
        outputs['P_hover_eff'] = inputs['P_hover'] * inputs['n'] ** 1.5

    def compute_partials(self, inputs, J):
        J['P_hover_eff', 'P_hover'] = inputs['n'] ** 1.5
        J['P_hover_eff', 'n'] = 1.5 * inputs['P_hover'] * inputs['n'] ** 0.5
