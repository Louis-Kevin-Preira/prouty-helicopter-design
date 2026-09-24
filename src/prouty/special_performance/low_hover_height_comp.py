"""
LowHoverHeightComp -- G2d, low hover height h_lo of the height-velocity diagram.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Generating the Deadman's Curve" p. 354 (single engine) and
p. 357 (multiengine).

Vertical descent at the landing gear sink speed V_LG while the rotor kinetic
energy supplies hover power IGE, down to C_T/sigma = 0.2 (thrust ~ Omega^2):

    single:  h_lo = V_LG J Omega_0^2 [1 - g] / (1,100 hp_IGE)
    multi:   h_lo = V_LG J Omega_0^2 [1 - g] / (1,100 (hp_IGE - hp_avail))

    g = (C_W/sigma)/0.2          (energy balance, both cases)
    g = sqrt((C_W/sigma)/0.2)    multi-engine as printed, book=True (C5-3)

hp_IGE excludes the tail rotor induced power (p. 354).
Multi-engine: requires hp_IGE > hp_avail (else no single-engine envelope).

    V_LG, J, Omega_0, CW_sigma, P_IGE [, P_avail] --> h_lo
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0
CT_SIGMA_LIMIT = 0.2


class LowHoverHeightComp(om.ExplicitComponent):
    """Low hover height, pp. 354 and 357."""

    def initialize(self):
        self.options.declare('engines', default='single', values=('single', 'multi'))
        self.options.declare('book', types=bool, default=False,
                             desc='multi-engine only: printed square root (C5-3)')

    def setup(self):
        self.add_input('V_LG', val=8.0, units='ft/s')
        self.add_input('J', val=11735.0, units='slug*ft**2')
        self.add_input('Omega_0', val=21.67, units='rad/s')
        self.add_input('CW_sigma', val=0.083)
        self.add_input('P_IGE', val=1500.0, units='hp')
        if self.options['engines'] == 'multi':
            self.add_input('P_avail', val=0.0, units='hp')
        self.add_output('h_lo', val=10.0, units='ft')
        self.declare_partials('h_lo', '*')

    def _terms(self, inputs):
        u = inputs['CW_sigma'] / CT_SIGMA_LIMIT
        use_sqrt = self.options['engines'] == 'multi' and self.options['book']
        g, dg_dcw = ((np.sqrt(u), 0.5 / np.sqrt(u) / CT_SIGMA_LIMIT) if use_sqrt
                     else (u, 1.0 / CT_SIGMA_LIMIT))
        P = inputs['P_IGE'] - (inputs['P_avail'] if 'P_avail' in inputs else 0.0)
        return g, dg_dcw, P

    def compute(self, inputs, outputs):
        g, _, P = self._terms(inputs)
        outputs['h_lo'] = (inputs['V_LG'] * inputs['J'] * inputs['Omega_0'] ** 2 * (1.0 - g)
                           / (2.0 * HP_TO_FT_LBF_PER_S * P))

    def compute_partials(self, inputs, J):
        g, dg_dcw, P = self._terms(inputs)
        V, Jr, w = inputs['V_LG'], inputs['J'], inputs['Omega_0']
        e = V * Jr * w ** 2 / (2.0 * HP_TO_FT_LBF_PER_S * P)
        h = e * (1.0 - g)
        J['h_lo', 'V_LG'] = h / V
        J['h_lo', 'J'] = h / Jr
        J['h_lo', 'Omega_0'] = 2.0 * h / w
        J['h_lo', 'CW_sigma'] = -e * dg_dcw
        J['h_lo', 'P_IGE'] = -h / P
        if 'P_avail' in inputs:
            J['h_lo', 'P_avail'] = h / P
