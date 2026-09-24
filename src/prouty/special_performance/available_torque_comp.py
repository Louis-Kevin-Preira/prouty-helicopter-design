"""
AvailableTorqueComp -- G3, maximum torque coefficient available to the main rotor.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Maximum Acceleration" p. 365 ("C_Q/sigma max. avail. to main rotor").

    C_Q/sigma = 550 hp_MR / (rho A_b (Omega R)^3)

hp_MR: power available at the main rotor (engine rating less tail rotor and
drive losses, Chapter 4).

    P_MR_avail, rho, A_b, V_tip --> CQ_sigma_avail
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class AvailableTorqueComp(om.ExplicitComponent):
    """Main rotor torque coefficient at the available power, p. 365."""

    def setup(self):
        self.add_input('P_MR_avail', val=3600.0, units='hp')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A_b', val=240.0, units='ft**2')
        self.add_input('V_tip', val=650.0, units='ft/s')
        self.add_output('CQ_sigma_avail', val=0.01)
        self.declare_partials('CQ_sigma_avail', '*')

    def compute(self, inputs, outputs):
        i = inputs
        outputs['CQ_sigma_avail'] = (HP_TO_FT_LBF_PER_S * i['P_MR_avail']
                                     / (i['rho'] * i['A_b'] * i['V_tip'] ** 3))

    def compute_partials(self, inputs, J):
        i = inputs
        cq = HP_TO_FT_LBF_PER_S * i['P_MR_avail'] / (i['rho'] * i['A_b'] * i['V_tip'] ** 3)
        J['CQ_sigma_avail', 'P_MR_avail'] = cq / i['P_MR_avail']
        J['CQ_sigma_avail', 'rho'] = -cq / i['rho']
        J['CQ_sigma_avail', 'A_b'] = -cq / i['A_b']
        J['CQ_sigma_avail', 'V_tip'] = -3.0 * cq / i['V_tip']
