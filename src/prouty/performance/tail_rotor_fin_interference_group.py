"""
TailRotorFinInterferenceGroup -- G3, tail rotor with its fin in hover.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Tail Rotor-Fin Interference in Hover", pp. 283-287; tail rotor
power in the hover example pp. 309-310; Chapter 1 hover rotor, pp. 69-72.

    fin          FinInterferenceRatioComp    F_T        Figure 4.9     S_A, x_R, installation
    gross        TailRotorGrossThrustComp    T_gross    p. 283         T_req from yaw balance (G5)
    tail_rotor   HoverRotorGroup, trim       P_TR_iso   Chapter 1      trimmed to T_gross
    power        TailRotorFinPowerComp       P_TR       p. 286

No loop: T_req is an input. The tail rotor keeps its own Newton trim.

Names. The tail rotor inputs are prefixed tr_ (tr_R, tr_b, tr_theta_1, ...) so
this group can sit next to the main rotor of VerticalDragGroup; altitude and dT
are shared. Its isolated power is promoted as P_TR_iso, its collective and
blade loading as tr_theta_0 and tr_CT_sigma; the rest stays under tail_rotor.
P_TR is the input of PowerLossesGroup (G1).

    T_req, S_A, x_R, tr_* geometry, altitude, dT
        --> F_T, T_gross, P_TR_iso, P_TR, tr_theta_0, tr_CT_sigma
"""

import openmdao.api as om

from prouty.hover import HoverRotorGroup
from prouty.performance.fin_interference_ratio_comp import INSTALLATIONS, FinInterferenceRatioComp
from prouty.performance.tail_rotor_fin_power_comp import TailRotorFinPowerComp
from prouty.performance.tail_rotor_gross_thrust_comp import TailRotorGrossThrustComp

TAIL_ROTOR_INPUTS = ['R', 'c_root', 'c_tip', 'r_1', 'r_cutout', 'V_tip', 'b', 'theta_1']


class TailRotorFinInterferenceGroup(om.Group):
    """Fin interference ratio, gross thrust, tail rotor trim and installed power."""

    def initialize(self):
        self.options.declare('installation', default='pusher', values=INSTALLATIONS)
        self.options.declare('num_elements', types=int, default=10,
                             desc='blade elements of the Chapter 1 tail rotor')
        self.options.declare('theta_0_bounds', types=tuple, default=(0.0, 25.0),
                             desc='tail rotor collective bounds in trim, deg')

    def setup(self):
        self.add_subsystem('fin', FinInterferenceRatioComp(
            installation=self.options['installation']), promotes=['*'])
        self.add_subsystem('gross', TailRotorGrossThrustComp(), promotes=['*'])
        self.add_subsystem('tail_rotor', HoverRotorGroup(
            num_elements=self.options['num_elements'], mode='trim',
            theta_0_bounds=self.options['theta_0_bounds']),
            promotes_inputs=[(name, 'tr_' + name) for name in TAIL_ROTOR_INPUTS]
            + ['altitude', 'dT', ('T_target', 'T_gross')],
            promotes_outputs=[('power_hp', 'P_TR_iso'), ('theta_0', 'tr_theta_0'),
                              ('CT_sigma', 'tr_CT_sigma')])
        self.add_subsystem('power', TailRotorFinPowerComp(), promotes=['*'])
