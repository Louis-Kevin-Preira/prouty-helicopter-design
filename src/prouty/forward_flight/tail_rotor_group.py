"""
TailRotorGroup -- the tail rotor solved as a rotor, not as a constant.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 187-191 (tail rotor equations) and p. 234, 236 (Table 3.5,
case 3).

Given the thrust the anti-torque balance demands, this group returns the
tail rotor's own trim and the two quantities the helicopter trim loop needs
back from it:

    H_T     in-plane force, which adds to the fuselage drag in the alpha_TPP
            balance of p. 192
    hp_T    power, which adds to the main rotor power in the total

The chain is the main rotor chain with tail rotor constants, plus the
delta-three coupling:

    T_T                -> C_T/sigma_T                 ThrustCoefComp
    C_T/sigma_T, mu_T  -> v1/Omega R                  InducedVelocityComp
    C_T/sigma_T        -> a_0                         ConingComp
    ... , delta_3      -> theta_0, a_1s, b_1s         TailRotorTrimComp
    ...                -> C_H/sigma_T                 HForceCoefComp
    ...                -> C_Q/sigma_T, hp_T           power identity, p. 189

Two things make the tail rotor different from the main rotor, and both are
handled by TailRotorTrimComp: there is no cyclic pitch, so the disc tilts
freely and the shaft angle rather than the tip path plane angle is known; and
delta_three couples flapping to feathering.

Reusing the main rotor H-force. Prouty's tail rotor H-force equation (p. 189)
is the main rotor equation of p. 176 with theta_0 replaced by
theta_0 + a_0 tan(delta_3) and lambda' replaced by lambda + mu a_1s.
TailRotorTrimComp emits exactly those two as theta_0_eff and lambda_eff, so
HForceCoefComp is reused unchanged rather than duplicated.

Careful with the two inflows. The torque identity of p. 189 uses lambda, the
SHAFT inflow mu alpha_s - v1/Omega R, not the lambda_eff that feeds the
H-force. They differ by mu a_1s, which is 0.009 at mu = 0.3 against a lambda
of -0.009 -- the same order as lambda itself.

Appendix A, p. 670: R_T = 6.5 ft, c_T = 1 ft, b_T = 3, sigma_T = 0.146,
A_bT = 19.4 ft^2, Omega R_T = 650 ft/s, theta_1T = -5 deg, gamma_T = 4,
l_T = 37 ft, delta_3 = -30 deg. Note that Appendix A specifies -30 deg while
Table 3.5 case 3 works with delta_3 = 0; the default here follows the Appendix.

What is still missing: the fin blockage, which Appendix A gives as 31.5 ft^2
of blocked area, and the fact that the tail rotor works partly in the main
rotor wake. Both raise the thrust actually required. Chapter 4 handles them.

    T_T, mu, V_tip --> TailRotorGroup --> H_T, hp_T, CT_sigma_T, theta_0_T
"""

import numpy as np
import openmdao.api as om

from .coning_comp import ConingComp
from .h_force_coef_comp import HForceCoefComp
from .induced_velocity_comp import InducedVelocityComp
from .tail_rotor_load_comp import RotorPowerComp
from .tail_rotor_trim_comp import TailRotorTrimComp
from .thrust_coef_comp import ThrustCoefComp


class TailRotorGroup(om.Group):
    """Tail rotor trim, H-force and power, p. 187-191."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        zeros = np.zeros(nn)

        # tail rotor tip speed ratio, from the main rotor's
        self.add_subsystem(
            'advance_ratio',
            om.ExecComp('mu_T = mu * V_tip / V_tip_T',
                        # non-zero start: the first residual evaluation runs
                        # every component once, and mu_T = 0 is singular in
                        # InducedVelocityComp
                        mu_T=0.3 * np.ones(nn), mu=0.3 * np.ones(nn),
                        V_tip={'val': 650.0, 'units': 'ft/s'},
                        V_tip_T={'val': 650.0, 'units': 'ft/s'}),
            promotes=['*'])

        self.add_subsystem('thrust_coef', ThrustCoefComp(num_nodes=nn),
                           promotes_inputs=[('T', 'T_T'), 'rho',
                                            ('A_b', 'A_b_T'),
                                            ('V_tip', 'V_tip_T')],
                           promotes_outputs=[('CT_sigma', 'CT_sigma_T')])

        self.add_subsystem('induced_velocity', InducedVelocityComp(num_nodes=nn),
                           promotes_inputs=[('CT_sigma', 'CT_sigma_T'),
                                            ('mu', 'mu_T'),
                                            ('sigma', 'sigma_T')],
                           promotes_outputs=[('vi_OR', 'vi_OR_T')])

        self.add_subsystem('coning', ConingComp(num_nodes=nn, form='ct'),
                           promotes_inputs=[('gamma', 'gamma_T'),
                                            ('CT_sigma', 'CT_sigma_T'), 'a',
                                            ('R', 'R_T'), ('V_tip', 'V_tip_T'),
                                            'g'],
                           promotes_outputs=[('a0', 'a0_T')])

        self.add_subsystem('trim', TailRotorTrimComp(num_nodes=nn),
                           promotes_inputs=[('mu', 'mu_T'),
                                            ('CT_sigma', 'CT_sigma_T'),
                                            ('a0', 'a0_T'),
                                            ('vi_OR', 'vi_OR_T'),
                                            ('alpha_s', 'alpha_s_T'),
                                            ('theta_1', 'theta_1_T'), 'a',
                                            'delta_3'],
                           promotes_outputs=[('theta_0', 'theta_0_T'),
                                             ('a1s', 'a1s_T'),
                                             ('b1s', 'b1s_T'),
                                             'theta_0_eff', 'lambda_eff'])

        self.add_subsystem('h_force', HForceCoefComp(num_nodes=nn, form='direct'),
                           promotes_inputs=[('cd_bar', 'cd_bar_T'),
                                            ('mu', 'mu_T'),
                                            ('lambda_p', 'lambda_eff'),
                                            ('CT_sigma', 'CT_sigma_T'),
                                            ('a1s', 'a1s_T'),
                                            ('theta_0', 'theta_0_eff'),
                                            ('a0', 'a0_T'),
                                            ('vi_OR', 'vi_OR_T'),
                                            ('theta_1', 'theta_1_T'), 'a',
                                            ('B', 'B_T'), ('x_0', 'x_0_T')],
                           promotes_outputs=[('CH_sigma', 'CH_sigma_T')])

        # p. 189: C_Q/sigma_T uses the SHAFT inflow, not lambda_eff
        self.add_subsystem(
            'torque',
            om.ExecComp('CQ_sigma_T = cd_bar_T / 8 * (1 + 3 * mu_T**2)'
                        ' - (mu_T * alpha_s_T - vi_OR_T) * CT_sigma_T'
                        ' - mu_T * CH_sigma_T',
                        mu_T=zeros.copy(),
                        alpha_s_T={'val': zeros.copy(), 'units': 'rad'},
                        vi_OR_T=zeros.copy(), CQ_sigma_T=zeros.copy(),
                        cd_bar_T=zeros.copy(), CT_sigma_T=zeros.copy(),
                        CH_sigma_T=zeros.copy()),
            promotes=['*'])

        self.add_subsystem('power', RotorPowerComp(num_nodes=nn),
                           promotes_inputs=[('CQ_sigma', 'CQ_sigma_T'), 'rho',
                                            ('A_b', 'A_b_T'),
                                            ('V_tip', 'V_tip_T')],
                           promotes_outputs=[('hp', 'hp_T')])

        self.add_subsystem(
            'h_force_dim',
            om.ExecComp('H_T = CH_sigma_T * rho * A_b_T * V_tip_T**2',
                        H_T={'val': zeros.copy(), 'units': 'lbf'},
                        CH_sigma_T=zeros.copy(),
                        rho={'val': 0.002377, 'units': 'slug/ft**3'},
                        A_b_T={'val': 19.4, 'units': 'ft**2'},
                        V_tip_T={'val': 650.0, 'units': 'ft/s'}),
            promotes=['*'])

        # Appendix A p. 670
        self.set_input_defaults('A_b_T', val=19.4, units='ft**2')
        self.set_input_defaults('V_tip_T', val=650.0, units='ft/s')
        self.set_input_defaults('sigma_T', val=0.146)
        self.set_input_defaults('R_T', val=6.5, units='ft')
        self.set_input_defaults('gamma_T', val=4.0 * np.ones(nn))
        self.set_input_defaults('theta_1_T', val=np.deg2rad(-5.0), units='rad')
        self.set_input_defaults('delta_3', val=np.deg2rad(-30.0), units='rad')
        self.set_input_defaults('alpha_s_T', val=zeros.copy(), units='rad')
        # backed out of Table 3.1: the tail rotor runs at C_T/sigma = 0.039,
        # so alpha_ref = 2.2 deg against the main rotor's 4.9 deg
        self.set_input_defaults('cd_bar_T', val=0.0084 * np.ones(nn))
        self.set_input_defaults('mu_T', val=0.3 * np.ones(nn))
        self.set_input_defaults('CT_sigma_T', val=0.04 * np.ones(nn))
        self.set_input_defaults('CH_sigma_T', val=zeros.copy())
        self.set_input_defaults('vi_OR_T', val=zeros.copy())
