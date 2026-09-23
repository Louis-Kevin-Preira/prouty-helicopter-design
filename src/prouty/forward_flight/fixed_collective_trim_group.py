"""
FixedCollectiveTrimGroup -- trim with the collective given, p. 242-246.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, Table 3.5 case 7 (dive at constant collective pitch, p. 242-244)
and case 8 (helicopter with auxiliary propulsion, p. 245-246).

Why this is not another mode of TrimConditionsGroup. In level flight, climb
and autorotation the thrust comes out of the in-plane force balance, with
alpha_TPP, H_M and H_T all coupled, and the collective is an output. These two
cases invert that: the collective is GIVEN, and p. 243 step d takes the thrust
from a bare vertical balance,

    C_T/sigma = (G.W. - L_F) / [ (rho/rho_0) 241,100 ]

with no in-plane term at all. The flight path angle then falls out at the end,
from the horizontal balance of p. 244. One state instead of two, a different
residual, and the rotor running in the opposite direction -- theta_0 to
C_T/sigma rather than C_T/sigma to theta_0. That is a different model, not a
branch.

The loop, and it is a short one:

    alpha_F  (state)
        -> lambda' = mu (alpha_F + i_s + a_1s)
        -> L_F, f            from Figure A.2
        -> C_T/sigma         from the rotor, given theta_0 and lambda'
        -> T
    residual: T - (G.W. - L_F)

Everything after that is explicit: power, tail rotor, the rotor's own flat
plate area f_M, and finally the propulsive balance. Prouty iterates the same
loop by hand over steps b through i.

The two modes differ only in who supplies the propulsive force. Case 7 gives it
to gravity and asks for the dive angle; case 8 gives it to a propeller and asks
for its thrust. Both come out of PropulsiveBalanceComp, and on the same flight
condition both give 2,764 lb -- Prouty notes the coincidence himself at the end
of case 8.

Chart twist correction, absent by design. p. 243 step e converts the collective
to the chart's twist, theta_0 + 0.75(theta_1 + 5), because the charts are drawn
at theta_1 = -5 deg. Nothing here reads a chart: the rotor group takes theta_1
directly, so the actual collective is used and the correction has no place.

    theta_0, mu, GW, ... --> alpha_F, CT_sigma, hp_M, f_M
                             + gamma_D, R_D   ('dive')
                             + T_aux          ('auxiliary')
"""

import numpy as np
import openmdao.api as om

from .closed_form_rotor_group import ClosedFormRotorGroup
from .disc_airfoil_group import DiscAirfoilGroup
from .numerical_rotor_group import NumericalRotorGroup
from .descent_angle_comp import DescentAngleComp
from .dynamic_pressure_comp import DynamicPressureComp
from .fuselage_aero_comp import FuselageAeroComp
from .fuselage_angle_comp import FuselageAngleComp
from .propulsive_balance_comp import PropulsiveBalanceComp
from .rotor_drag_area_comp import RotorDragAreaComp
from .tail_rotor_group import TailRotorGroup
from .tail_rotor_load_comp import RotorPowerComp, TailRotorLoadComp

LIFT_SCALE = 1.0e-3          # residual is a force in lbf, the state an angle


class FixedCollectiveTrimGroup(om.Group):
    """Table 3.5 cases 7 and 8: dive, and auxiliary propulsion."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', values=('dive', 'auxiliary'),
                             default='dive')
        self.options.declare('tail_rotor', types=bool, default=True)
        self.options.declare('rotor', values=('closed_form', 'numerical'),
                             default='closed_form',
                             desc="'numerical' swaps G1 for G2. It matters "
                                  'most in the dive, where the flow is upward '
                                  'through the disc and the induced torque is '
                                  'negative, so C_Q is a small difference of '
                                  'cancelling terms and a mean c_d has no '
                                  'signal left to give.')
        self.options.declare('grid', types=tuple, default=(12, 15),
                             desc='(num_azimuth, num_radial) for G2')
        self.options.declare('rotor_options', types=dict,
                             default={'compressibility': False})
        self.options.declare('cross_check', types=bool, default=False,
                             desc='also compute the descent angle from rotor '
                                  'coefficients, p. 239, as an independent '
                                  'route to the same answer')

    def setup(self):
        nn = self.options['num_nodes']
        ones = np.ones(nn)

        # FuselageAngleComp and the rotor group both take vi_OR, with
        # different defaults; declare it once so the promotion is unambiguous
        self.set_input_defaults('mu', val=0.3 * ones)
        self.set_input_defaults('alpha_F', val=-0.08 * ones, units='rad')
        self.set_input_defaults('vi_OR', val=0.012 * ones)
        self.set_input_defaults('CT_sigma', val=0.085 * ones)
        if self.options['rotor'] == 'closed_form':
            self.set_input_defaults('a0', val=0.075 * ones, units='rad')
        else:
            self.set_input_defaults('V_son', val=1116.0, units='ft/s')
            self.set_input_defaults('c_R', val=2.0 / 30.0)
            # RotorDragAreaComp and G2 default sigma differently, 0.084883
            # against 0.085; declare it once rather than let OpenMDAO pick
            self.set_input_defaults('sigma', val=0.084883)

        balance = om.BalanceComp()
        balance.add_balance('alpha_F', val=-0.08 * ones, units='rad',
                            lhs_name='res_lift', rhs_val=np.zeros(nn),
                            lower=-0.45, upper=0.45)
        self.add_subsystem('balance', balance, promotes=['*'])

        self.add_subsystem('dynamic_pressure', DynamicPressureComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('inflow',
                           FuselageAngleComp(num_nodes=nn, mode='to_inflow'),
                           promotes_inputs=['alpha_F', 'mu', 'i_s', 'a1s',
                                            'vi_OR'],
                           promotes_outputs=['lambda_p'])
        self.add_subsystem('fuselage_aero', FuselageAeroComp(num_nodes=nn),
                           promotes=['*'])

        if self.options['rotor'] == 'closed_form':
            self.add_subsystem(
                'rotor', ClosedFormRotorGroup(num_nodes=nn, mode='thrust',
                                              **self.options['rotor_options']),
                promotes=['*'])
        else:
            n_psi, n_r = self.options['grid']
            grid = dict(num_nodes=nn, num_azimuth=n_psi, num_radial=n_r)
            # inflow='momentum' lets G2 own v_1 and a_0, which it derives from
            # its own C_T/sigma. That is acyclic here because FuselageAngleComp
            # in 'to_inflow' mode builds lambda' from alpha_F alone -- v_1 only
            # feeds its alpha_DW diagnostic.
            self.add_subsystem(
                'rotor', NumericalRotorGroup(airfoil=DiscAirfoilGroup(**grid),
                                             trim=True, inflow='momentum',
                                             **grid),
                promotes_inputs=['mu', 'lambda_p', 'theta_0', 'theta_1',
                                 'V_tip', 'V_son', 'c_R', 'B', 'x_0', 'gamma',
                                 'a', 'R', 'sigma'],
                promotes_outputs=['CT_sigma', 'CQ_sigma', 'CH_sigma', 'vi_OR',
                                  'a0', 'alpha_1270', 'CQ0_sigma_max',
                                  ('A_1', 'A1_b1s'), ('B_1', 'B1_a1s')])

        self.add_subsystem(
            'thrust',
            om.ExecComp('T = CT_sigma * rho * A_b * V_tip**2',
                        T={'val': np.zeros(nn), 'units': 'lbf'},
                        CT_sigma=np.zeros(nn),
                        rho={'val': 0.002377, 'units': 'slug/ft**3'},
                        A_b={'val': 240.0, 'units': 'ft**2'},
                        V_tip={'val': 650.0, 'units': 'ft/s'}),
            promotes=['*'])
        self.add_subsystem(
            'residual',
            om.ExecComp('res_lift = SCALE * (T - (GW - L_F))',
                        res_lift=np.zeros(nn), T={'val': np.zeros(nn),
                                                  'units': 'lbf'},
                        GW={'val': 20000.0, 'units': 'lbf'},
                        L_F={'val': np.zeros(nn), 'units': 'lbf'},
                        SCALE=LIFT_SCALE),
            promotes=['*'])

        power = ('CQ_sigma_total'
                 if self.options['rotor_options'].get('compressibility', False)
                 else 'CQ_sigma')
        self.add_subsystem('power', RotorPowerComp(num_nodes=nn),
                           promotes_inputs=['rho', 'A_b', 'V_tip'],
                           promotes_outputs=[('hp', 'hp_M')])
        self.connect(power, 'power.CQ_sigma')
        self.add_subsystem('tail_load', TailRotorLoadComp(num_nodes=nn),
                           promotes=['*'])
        if self.options['tail_rotor']:
            self.add_subsystem('tail_rotor', TailRotorGroup(num_nodes=nn),
                               promotes=['*'])
            self.add_subsystem(
                'total_power',
                om.ExecComp('hp_total = hp_M + hp_T',
                            hp_total={'val': np.zeros(nn), 'units': 'hp'},
                            hp_M={'val': np.zeros(nn), 'units': 'hp'},
                            hp_T={'val': np.zeros(nn), 'units': 'hp'}),
                promotes=['*'])

        self.add_subsystem('drag_area',
                           RotorDragAreaComp(num_nodes=nn, source='inflow'),
                           promotes=['*'])
        self.add_subsystem('propulsive',
                           PropulsiveBalanceComp(num_nodes=nn,
                                                 mode=self.options['mode']),
                           promotes=['*'])

        if self.options['cross_check']:
            self.add_subsystem(
                'coefficient_route', DescentAngleComp(num_nodes=nn),
                promotes_inputs=['lambda_p', 'CT_sigma', 'CH_sigma', 'mu',
                                 'sigma', 'f', 'A_b', 'V_tip'],
                promotes_outputs=[('gamma_D', 'gamma_D_coef'),
                                  ('R_D', 'R_D_coef'), 'CD_F_sigma',
                                  ('alpha_TPP', 'alpha_TPP_coef'),
                                  ('sin_gamma_D', 'sin_gamma_D_coef'),
                                  ('gamma_fp', 'gamma_fp_coef')])

        newton = self.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True, max_sub_solves=100)
        for key, value in dict(maxiter=60, atol=1e-12, rtol=1e-14,
                               iprint=0).items():
            newton.options[key] = value
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar',
                                                 maxiter=8, iprint=0)
        self.linear_solver = om.DirectSolver()
