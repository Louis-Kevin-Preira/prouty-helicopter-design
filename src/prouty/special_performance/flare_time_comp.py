"""
FlareTimeComp -- G2e, time available for the final nose-down rotation and
collective flare.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Minimum Touchdown Speed" p. 361.

    Delta_t = J Omega_0^2 [1 - (C_W/sigma)/(C_T/sigma)_max] / (1,100 hp_OGE)

(C_T/sigma)_max at the end of the maneuver from hover charts (Chapter 1).
Example: 1.25 s (p. 362). Same form as t_equiv of p. 363 without the 0.8.

    J, Omega_0, CW_sigma (nn,), CT_sigma_max (nn,), P_OGE (nn,) --> dt (nn,)
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class FlareTimeComp(om.ExplicitComponent):
    """Time the rotor energy supplies hover power down to (C_T/sigma)_max, p. 361."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('J', val=11735.0, units='slug*ft**2')
        self.add_input('Omega_0', val=21.67, units='rad/s')
        self.add_input('CW_sigma', val=0.083 * np.ones(nn))
        self.add_input('CT_sigma_max', val=0.14 * np.ones(nn))
        self.add_input('P_OGE', val=1650.0 * np.ones(nn), units='hp')
        self.add_output('dt', val=np.ones(nn), units='s')
        self.declare_partials('dt', ['CW_sigma', 'CT_sigma_max', 'P_OGE'], rows=ar, cols=ar)
        self.declare_partials('dt', ['J', 'Omega_0'], rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        g = 1.0 - inputs['CW_sigma'] / inputs['CT_sigma_max']
        outputs['dt'] = (inputs['J'] * inputs['Omega_0'] ** 2 * g
                         / (2.0 * HP_TO_FT_LBF_PER_S * inputs['P_OGE']))

    def compute_partials(self, inputs, J):
        Jr, w, P = inputs['J'], inputs['Omega_0'], inputs['P_OGE']
        cw, cm = inputs['CW_sigma'], inputs['CT_sigma_max']
        e = Jr * w ** 2 / (2.0 * HP_TO_FT_LBF_PER_S * P)
        g = 1.0 - cw / cm
        J['dt', 'J'] = e * g / Jr
        J['dt', 'Omega_0'] = 2.0 * e * g / w
        J['dt', 'P_OGE'] = -e * g / P
        J['dt', 'CW_sigma'] = -e / cm
        J['dt', 'CT_sigma_max'] = e * cw / cm ** 2
