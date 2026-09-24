"""
ClimbPowerGroup -- G2, power required in a vertical climb.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Power Required in a Vertical Climb" pp. 97-101, Figures 2.3-2.4.

The full equation of p. 98 is the one Chapter 4 repeats on p. 314, so the
Chapter 4 components are reused as they are:

    vi        ClimbInducedVelocityComp (Ch. 4)   v_hov, v_sum = v_1c + V_c     p. 98
    power     VerticalClimbPowerComp (Ch. 4)     dP (full equation), k_T       p. 98
    approx    ClimbPowerApproxComp               dP_mom, dP_low                pp. 99-100

tail_rotor selects the tail rotor hover induced velocity v_hov_T, which p. 98
holds at its hover value:
    'input'        v_hov_T is a free input (default)
    'hover_power'  from the hover main rotor power, as p. 98 does:
        tr_thrust  TailRotorThrustComp (Ch. 4)   T_T = 550 P_M R_M / [(Omega R)_M l_T]
        tail_disc  A_T = pi tr_R^2
        vi_tail    ClimbInducedVelocityComp      v_hov_T

inflow = 'external' drops vi when G0 already provides v_hov and v_sum.

Dv_GW, dAz_CD and v_hov_T keep the Chapter 4 names, so they link to the
Chapter 4 vertical drag and tail rotor groups by promotion.

C2-5: the braces of p. 98 put the tail rotor factor on the whole climb-minus-
hover difference, which confirms the reading taken for C4-3 on p. 314.

    GW, T, V_c (nn,), rho, A, Dv_GW (nn,), dAz_CD, R, V_tip, l_T,
    [v_hov_T] or [P_MR, tr_R] --> v_hov, v_sum, dP, k_T, dP_mom, dP_low
"""

import openmdao.api as om

from prouty.performance.climb_induced_velocity_comp import ClimbInducedVelocityComp
from prouty.performance.tail_rotor_thrust_comp import TailRotorThrustComp
from prouty.performance.vertical_climb_power_comp import VerticalClimbPowerComp
from prouty.vertical.climb_power_approx_comp import ClimbPowerApproxComp


class ClimbPowerGroup(om.Group):
    """Excess power of a vertical climb over hover, pp. 97-101."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('tail_rotor', default='input', values=('input', 'hover_power'))
        self.options.declare('inflow', default='internal', values=('internal', 'external'),
                             desc="'external': v_hov and v_sum come from G0 (FlowStatesGroup)")

    def setup(self):
        nn = self.options['num_nodes']

        if self.options['inflow'] == 'internal':
            self.add_subsystem('vi', ClimbInducedVelocityComp(num_nodes=nn),
                               promotes_inputs=['T', 'rho', 'A', 'V_c'],
                               promotes_outputs=['v_hov', 'v_sum'])

        if self.options['tail_rotor'] == 'hover_power':
            self.add_subsystem('tr_thrust', TailRotorThrustComp(),
                               promotes_inputs=['P_MR', ('R_M', 'R'), ('V_tip_M', 'V_tip'), 'l_T'],
                               promotes_outputs=[('T_req', 'T_T')])
            self.add_subsystem('tail_disc', om.ExecComp(
                'A_T = pi * tr_R ** 2', A_T={'units': 'ft**2'}, tr_R={'units': 'ft'}),
                promotes=['*'])
            self.add_subsystem('vi_tail', ClimbInducedVelocityComp(),
                               promotes_inputs=[('T', 'T_T'), 'rho', ('A', 'A_T')],
                               promotes_outputs=[('v_hov', 'v_hov_T')])
            self.set_input_defaults('tr_R', 1.0, units='ft')

        self.add_subsystem('power', VerticalClimbPowerComp(num_nodes=nn),
                           promotes_inputs=['GW', 'v_hov', 'v_sum', 'V_c', 'Dv_GW', 'rho',
                                            ('A_M', 'A'), 'dAz_CD', 'v_hov_T',
                                            ('R_M', 'R'), ('V_tip_M', 'V_tip'), 'l_T'],
                           promotes_outputs=['dP', 'k_T'])
        self.add_subsystem('approx', ClimbPowerApproxComp(num_nodes=nn), promotes=['*'])

        # the Chapter 4 components default the tip speed to 1 ft/s; the chapter uses 650
        self.set_input_defaults('V_tip', 650.0, units='ft/s')
