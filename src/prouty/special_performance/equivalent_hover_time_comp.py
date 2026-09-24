"""
EquivalentHoverTimeComp -- G2f, equivalent hover time t/k.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Autorotative Indices" pp. 363-364, Figure 5.13.

    t_equiv = J Omega^2 (1 - (C_W/sigma) / (0.8 (C_T/sigma)_max)) / (1,100 hp_OGE)

Design goal for single-engine helicopters: at least 1.5 s (Figure 5.13).
(C_T/sigma)_max from the isolated rotor hover charts (Chapter 1).

    J, Omega, CW_sigma, CT_sigma_max, P_OGE --> t_equiv
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class EquivalentHoverTimeComp(om.ExplicitComponent):
    """Time the stored rotor energy can supply hover power before stall, p. 363."""

    def setup(self):
        self.add_input('J', val=11735.0, units='slug*ft**2')
        self.add_input('Omega', val=21.67, units='rad/s')
        self.add_input('CW_sigma', val=0.083)
        self.add_input('CT_sigma_max', val=0.155)
        self.add_input('P_OGE', val=2000.0, units='hp')
        self.add_output('t_equiv', val=1.0, units='s')
        self.declare_partials('t_equiv', '*')

    def compute(self, inputs, outputs):
        g = 1.0 - inputs['CW_sigma'] / (0.8 * inputs['CT_sigma_max'])
        outputs['t_equiv'] = (inputs['J'] * inputs['Omega'] ** 2 * g
                              / (2.0 * HP_TO_FT_LBF_PER_S * inputs['P_OGE']))

    def compute_partials(self, inputs, J):
        Jr, w, P = inputs['J'], inputs['Omega'], inputs['P_OGE']
        cw, cm = inputs['CW_sigma'], inputs['CT_sigma_max']
        g = 1.0 - cw / (0.8 * cm)
        e = Jr * w ** 2 / (2.0 * HP_TO_FT_LBF_PER_S * P)
        J['t_equiv', 'J'] = e * g / Jr
        J['t_equiv', 'Omega'] = 2.0 * e * g / w
        J['t_equiv', 'P_OGE'] = -e * g / P
        J['t_equiv', 'CW_sigma'] = -e / (0.8 * cm)
        J['t_equiv', 'CT_sigma_max'] = e * cw / (0.8 * cm ** 2)
