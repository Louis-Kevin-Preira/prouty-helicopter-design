"""
AutorotationCollectiveComp -- collective pitch in vertical autorotation.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rate of Descent in Vertical Autorotation" pp. 111, 114.

Tip pitch of the ideal twist (p. 111): theta_T = (4/a) C_T/sigma + phi_T, with
phi_T = -(V_D - v_1)/(Omega R) = -(v_1hov/Omega R)(V_D_bar - v1_bar); the linear
twist of the same thrust (p. 114):

    theta_0 = (3/2) theta_T - (3/4) theta_1
            = (3/2) [(4/a) C_T/sigma - (v_1hov/Omega R)(V_D_bar - v1_bar)] - (3/4) theta_1

in radians (the book multiplies the bracket by 57.3). theta_0 is the root pitch,
theta_75 = theta_0 + 0.75 theta_1 = (3/2) theta_T.

    CT_sigma, v_hov, VD_minus_v1_bar (nn,), a, V_tip, theta_1 --> theta_0_auto (nn,)
"""

import numpy as np
import openmdao.api as om


class AutorotationCollectiveComp(om.ExplicitComponent):
    """Collective pitch of vertical autorotation, p. 114."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('CT_sigma', val=np.full(nn, 0.085), desc='thrust coefficient over solidity')
        self.add_input('v_hov', val=np.ones(nn), units='ft/s', desc='hover induced velocity')
        self.add_input('VD_minus_v1_bar', val=np.full(nn, 0.2), desc='V_D_bar - v1_bar')
        self.add_input('a', val=6.0, units='1/rad', desc='lift curve slope')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed Omega R')
        self.add_input('theta_1', val=-0.17453, units='rad', desc='blade twist')
        self.add_output('theta_0_auto', val=np.full(nn, 0.2), units='rad',
                        desc='root collective pitch in vertical autorotation')
        self.declare_partials('theta_0_auto', ['CT_sigma', 'v_hov', 'VD_minus_v1_bar'],
                              rows=ar, cols=ar)
        self.declare_partials('theta_0_auto', ['a', 'V_tip', 'theta_1'])

    def compute(self, inputs, outputs):
        a, V = inputs['a'][0], inputs['V_tip'][0]
        theta_T = 4.0 / a * inputs['CT_sigma'] - inputs['v_hov'] / V * inputs['VD_minus_v1_bar']
        outputs['theta_0_auto'] = 1.5 * theta_T - 0.75 * inputs['theta_1'][0]

    def compute_partials(self, inputs, partials):
        a, V = inputs['a'][0], inputs['V_tip'][0]
        x, vh, y = inputs['CT_sigma'], inputs['v_hov'], inputs['VD_minus_v1_bar']
        partials['theta_0_auto', 'CT_sigma'] = 6.0 / a + 0.0 * x
        partials['theta_0_auto', 'v_hov'] = -1.5 * y / V
        partials['theta_0_auto', 'VD_minus_v1_bar'] = -1.5 * vh / V
        partials['theta_0_auto', 'a'] = (-6.0 * x / a ** 2).reshape(-1, 1)
        partials['theta_0_auto', 'V_tip'] = (1.5 * vh * y / V ** 2).reshape(-1, 1)
        partials['theta_0_auto', 'theta_1'] = np.full((x.size, 1), -0.75)
