"""
AutorotationParameterComp -- the autorotation parameter V_D_bar - v1_bar.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rate of Descent in Vertical Autorotation" pp. 109-112, 115.

Zero torque of the ideally twisted rotor (p. 109) with momentum velocities
normalized by v_1hov = Omega R sqrt(C_T/2) (p. 112):

    V_D_bar - v1_bar = (3/2) sqrt(3) / (sqrt(sigma) cl_bar^(3/2) / c_d)      p. 112

When the rotor must still supply dh.p. (tail rotor, accessories), p. 115:

    V_D_bar - v1_bar = (3/2) sqrt(3) / (sqrt(sigma) cl_bar^(3/2) / c_d)
                       x [1 + 4,400 dh.p. / (rho A_b (Omega R)^3 c_d)]

written here as (3/2) sqrt(3) (c_d + 8 dP / (rho A_b (Omega R)^3)) / (sqrt(sigma) cl_bar^(3/2)),
dP in ft lb/s (4,400 = 8 x 550), A_b = sigma A.

    c_d, cl_bar (nn,), sigma, dP_auto (nn,), rho, A, V_tip --> VD_minus_v1_bar (nn,)
"""

import numpy as np
import openmdao.api as om

K_AUTO = 1.5 * np.sqrt(3.0)
HP_TO_FT_LBF_PER_S = 550.0


class AutorotationParameterComp(om.ExplicitComponent):
    """V_D_bar - v1_bar of vertical autorotation, pp. 112, 115."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('cd_bar', val=np.full(nn, 0.01), desc='section drag at cl_bar')
        self.add_input('cl_bar', val=np.full(nn, 0.5), desc='mean lift coefficient')
        self.add_input('dP_auto', val=np.zeros(nn), units='hp',
                       desc='power the rotor supplies in autorotation, p. 115')
        self.add_input('sigma', val=0.085, desc='rotor solidity')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A', val=1.0, units='ft**2', desc='disc area')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed Omega R')
        self.add_output('VD_minus_v1_bar', val=np.full(nn, 0.2), desc='V_D_bar - v1_bar')
        self.declare_partials('VD_minus_v1_bar', ['cd_bar', 'cl_bar', 'dP_auto'], rows=ar, cols=ar)
        self.declare_partials('VD_minus_v1_bar', ['sigma', 'rho', 'A', 'V_tip'])

    def _terms(self, inputs):
        sig, rho, A, V = (inputs[n][0] for n in ('sigma', 'rho', 'A', 'V_tip'))
        extra = 8.0 * HP_TO_FT_LBF_PER_S * inputs['dP_auto'] / (rho * sig * A * V ** 3)
        c = K_AUTO / (np.sqrt(sig) * inputs['cl_bar'] ** 1.5)
        return c, extra

    def compute(self, inputs, outputs):
        c, extra = self._terms(inputs)
        outputs['VD_minus_v1_bar'] = c * (inputs['cd_bar'] + extra)

    def compute_partials(self, inputs, partials):
        c, extra = self._terms(inputs)
        y = c * (inputs['cd_bar'] + extra)
        partials['VD_minus_v1_bar', 'cd_bar'] = c
        partials['VD_minus_v1_bar', 'cl_bar'] = -1.5 * y / inputs['cl_bar']
        partials['VD_minus_v1_bar', 'dP_auto'] = c * 8.0 * HP_TO_FT_LBF_PER_S / (
            inputs['rho'][0] * inputs['sigma'][0] * inputs['A'][0] * inputs['V_tip'][0] ** 3)
        for name, d in (('sigma', -0.5 * y - c * extra), ('rho', -c * extra),
                        ('A', -c * extra), ('V_tip', -3.0 * c * extra)):
            partials['VD_minus_v1_bar', name] = (d / inputs[name][0]).reshape(-1, 1)
