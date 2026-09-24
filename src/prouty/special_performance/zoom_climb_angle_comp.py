"""
ZoomClimbAngleComp -- G2c, climb angle during the zoom maneuver.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Glide Distance" p. 352.

    gamma_c = arccos( (C_W/sigma) / (C_T/sigma)_max )

(C_T/sigma)_max = 0.12 is the conservative value advised on p. 352.

    CW_sigma, CT_sigma_max_zoom --> gamma_c
"""

import numpy as np
import openmdao.api as om


class ZoomClimbAngleComp(om.ExplicitComponent):
    """Zoom climb angle, p. 352. Requires C_W/sigma < (C_T/sigma)_max."""

    def setup(self):
        self.add_input('CW_sigma', val=0.083)
        self.add_input('CT_sigma_max_zoom', val=0.12)
        self.add_output('gamma_c', val=0.8, units='rad')
        self.declare_partials('gamma_c', '*')

    def compute(self, inputs, outputs):
        outputs['gamma_c'] = np.arccos(inputs['CW_sigma'] / inputs['CT_sigma_max_zoom'])

    def compute_partials(self, inputs, J):
        cw, cm = inputs['CW_sigma'], inputs['CT_sigma_max_zoom']
        u = cw / cm
        d = -1.0 / np.sqrt(1.0 - u ** 2)
        J['gamma_c', 'CW_sigma'] = d / cm
        J['gamma_c', 'CT_sigma_max_zoom'] = -d * cw / cm ** 2
