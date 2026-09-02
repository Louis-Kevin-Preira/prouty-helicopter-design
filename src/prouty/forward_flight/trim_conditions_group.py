"""
TrimConditionsGroup (G1b) -- helicopter trim in level flight, climb and
autorotation.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 192-199.

Why this has to be a loop. The chain is circular twice over:

    alpha_F -> (Figure A.2) -> L_F, D_F -> alpha_TPP, T -> C_T/sigma, lambda'
            -> alpha_F

    C_T/sigma, lambda' -> theta_0, a_0 -> C_H/sigma -> alpha_TPP, T
                       -> C_T/sigma

The fuselage needs its own angle of attack to give its lift; that lift changes
the thrust the rotor must produce; the thrust changes the inflow; the inflow
gives back the fuselage angle. And the rotor H-force, 401 lb against 904 lb of
fuselage drag in level flight, is a third of the numerator of alpha_TPP, so it
cannot be dropped -- but getting it needs the whole closed-form rotor solution,
which needs C_T/sigma. Neither loop can be eliminated analytically, because
L/q and f are tabulated curves.

Two states close the two loops, and they are the two quantities Prouty carries
from one hand iteration to the next (p. 193): alpha_F and C_T/sigma. Declaring
them as balances makes the rest of the graph acyclic:

    alpha_F   -> lambda'          FuselageAngleComp, 'to_inflow'
    alpha_F   -> L_F, D_F         FuselageAeroComp
    C_T/sigma -> v1/Omega R       InducedVelocityComp
    C_T/sigma, lambda' -> theta_0, a_0, C_Q/sigma, C_H/sigma   G1
    ... -> H_M, hp_M, T_T
    L_F, D_F, H_M, H_T, gamma -> alpha_TPP, T                  TppAngleComp
    T         -> C_T/sigma computed                            residual 2
    alpha_TPP, v1/Omega R -> lambda' computed -> alpha_F computed
                                                               residual 1

Modes

    'level'         gamma = 0.
    'climb'         gamma from the prescribed rate of climb, p. 194.
    'autorotation'  C_Q/sigma prescribed by the losses, p. 197, and gamma
                    becomes a third state. This is the direct method; the
                    sweep of p. 196 is the same answer reached by running
                    'climb' at several descent rates.

The starting point is Prouty's own: alpha_TPP from parasite drag alone
(p. 167), which is the first line of his Table 3.2 iteration.

Two things this group does not yet do. H_T is an input rather than a tail
rotor solution -- it is 36 lb against 904 lb of fuselage drag in level flight,
4 % of the numerator, and wiring TailRotorTrimComp and HForceCoefComp with the
tail rotor constants is the obvious next step. And c_d is an input, backed out
of Table 3.3 as 0.0100; connecting a Chapter 6 airfoil through G1's `airfoil`
option removes that.

Compressibility is off by default. Feeding the trim loop with the Figure 3.43
penalty puts hp_M 10 % above Table 3.4 at mu = 0.45 (2,421 hp against 2,193),
whereas leaving it out reproduces the book to 0.7 % when evaluated at Prouty's
own tabulated point. The correction belongs after the trim, as an increment
on C_Q/sigma -- which is exactly how Table 3.5 case 2 applies it.

Known disagreements with the book, decided in advance so they are not chased:
climb thrust runs about 1 % high because T carries the G.W. sin(gamma) term
that Table 3.3 omits, and the autorotation column of Table 3.3 is not a fixed
point of its own equations -- its H_M and its power cannot both be right.
"""

import numpy as np
import openmdao.api as om

from .closed_form_rotor_group import ClosedFormRotorGroup
from .coning_comp import ConingComp
from .disc_airfoil_group import DiscAirfoilGroup
from .numerical_rotor_group import NumericalRotorGroup
from .dynamic_pressure_comp import DynamicPressureComp
from .flight_path_comp import FlightPathComp
from .fuselage_aero_comp import FuselageAeroComp
from .fuselage_angle_comp import FuselageAngleComp
from .induced_velocity_comp import InducedVelocityComp
from .losses_torque_comp import LossesTorqueComp
from .tail_rotor_group import TailRotorGroup
from .tail_rotor_load_comp import RotorPowerComp, TailRotorLoadComp
from .thrust_coef_comp import ThrustCoefComp
from .tpp_angle_comp import TppAngleComp


class TrimConditionsGroup(om.Group):
    """G1b -- trim loop of p. 192-199."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', default='level',
                             values=('level', 'climb', 'autorotation'))
        self.options.declare(
            'rotor_options', types=dict,
            default={'compressibility': False},
            desc="passed through to ClosedFormRotorGroup. Compressibility is "
                 "OFF by default: Table 3.4 was computed without it, and "
                 "adding it puts hp_M 10 % high at mu = 0.45 where the "
                 "closed-form trim otherwise reproduces the book to 0.7 %. "
                 "The Figure 3.43 penalty is applied after the trim, the way "
                 "Table 3.5 case 2 does it, not inside the loop.")
        self.options.declare('rotor', values=('closed_form', 'numerical'),
                             default='closed_form',
                             desc="'numerical' swaps G1 for G2. It costs an "
                                  'extra state: G1 runs backwards, taking '
                                  'C_T/sigma and returning theta_0, while G2 '
                                  'only runs forwards, so theta_0 becomes an '
                                  'unknown with the thrust match as its '
                                  'residual.')
        self.options.declare('grid', types=tuple, default=(12, 15),
                             desc='(num_azimuth, num_radial) for G2')
        self.options.declare('tail_rotor', types=bool, default=True,
                             desc='solve the tail rotor for H_T and hp_T, '
                                  'instead of taking H_T as an input')

    def setup(self):
        nn = self.options['num_nodes']
        mode = self.options['mode']
        ones = np.ones(nn)

        # ExecComps default their inputs to zero, the physical components to
        # sensible values; declare the shared ones once to remove the ambiguity
        self.set_input_defaults('mu', val=0.3 * ones)
        self.set_input_defaults('rho', val=0.002377, units='slug/ft**3')
        self.set_input_defaults('A_b', val=240.0, units='ft**2')
        self.set_input_defaults('V_tip', val=650.0, units='ft/s')
        if self.options['rotor'] == 'closed_form':
            self.set_input_defaults('CH_sigma', val=np.zeros(nn))
        else:
            self.set_input_defaults('V_son', val=1116.0, units='ft/s')
            self.set_input_defaults('c_R', val=2.0 / 30.0)
            self.set_input_defaults('sigma', val=0.084883)
        self.set_input_defaults('vi_OR', val=np.zeros(nn))
        self.set_input_defaults('alpha_TPP', val=np.zeros(nn), units='rad')
        self.set_input_defaults('CT_sigma', val=0.085 * ones)
        self.set_input_defaults('alpha_F', val=-0.1 * ones, units='rad')

        # ---------------------------------------------------------- states
        balance = om.BalanceComp()
        # Bounds are not decoration. Wiring the tail rotor made the system
        # stiff enough that an unbounded Newton walks into alpha_TPP = 62 deg
        # and C_H/sigma_T = 2.4, where the compressibility term switches on and
        # the step never recovers. These limits bracket any flight condition
        # this chapter describes.
        balance.add_balance('alpha_F', val=-0.1 * ones, units='rad',
                            lhs_name='res_alpha_F', rhs_val=np.zeros(nn),
                            eq_units='rad', lower=-0.45, upper=0.45)
        balance.add_balance('CT_sigma', val=0.085 * ones,
                            lhs_name='res_CT_sigma', rhs_val=np.zeros(nn),
                            lower=0.005, upper=0.20)
        if mode == 'autorotation':
            # sin(gamma) is periodic, so the residual has infinitely many
            # roots; bound the state to a physical flight path angle or the
            # Newton will happily settle on gamma - 2 k pi
            balance.add_balance('gamma_fp', val=-0.1 * ones, units='rad',
                                lhs_name='res_CQ_sigma', rhs_val=np.zeros(nn),
                                lower=-1.4, upper=1.4)
        if self.options['rotor'] == 'numerical':
            balance.add_balance('theta_0', val=0.26 * ones, units='rad',
                                lhs_name='res_theta', rhs_val=np.zeros(nn),
                                lower=-0.1, upper=0.6)
        self.add_subsystem('balance', balance, promotes=['*'])

        # ------------------------------------------------- flight condition
        self.add_subsystem('dynamic_pressure', DynamicPressureComp(num_nodes=nn),
                           promotes=['*'])
        if mode == 'climb':
            self.add_subsystem('flight_path',
                               FlightPathComp(num_nodes=nn, mode='from_rate'),
                               promotes=['*'])
        elif mode == 'autorotation':
            self.add_subsystem('flight_path',
                               FlightPathComp(num_nodes=nn, mode='from_angle'),
                               promotes=['*'])
            self.add_subsystem('losses', LossesTorqueComp(num_nodes=nn),
                               promotes=['*'])

        # ------------------------------------------------------- fuselage
        self.add_subsystem('inflow_from_angle',
                           FuselageAngleComp(num_nodes=nn, mode='to_inflow'),
                           promotes_inputs=['alpha_F', 'mu', 'i_s', 'a1s',
                                            'vi_OR'],
                           promotes_outputs=['lambda_p'])
        self.add_subsystem('fuselage_aero', FuselageAeroComp(num_nodes=nn),
                           promotes=['*'])

        # ---------------------------------------------------------- rotor
        self.add_subsystem('induced_velocity',
                           InducedVelocityComp(num_nodes=nn), promotes=['*'])
        if self.options['rotor'] == 'closed_form':
            self.add_subsystem(
                'rotor', ClosedFormRotorGroup(num_nodes=nn, mode='collective',
                                              **self.options['rotor_options']),
                promotes=['*'])
        else:
            n_psi, n_r = self.options['grid']
            grid = dict(num_nodes=nn, num_azimuth=n_psi, num_radial=n_r)
            # G2 takes a_0 as an input where ClosedFormRotorGroup makes its
            # own. Leaving it unconnected defaults it to 1 RADIAN and nothing
            # complains -- see validation_notes section 5.
            self.add_subsystem(
                'coning', ConingComp(num_nodes=nn, form='ct'),
                promotes_inputs=['CT_sigma', 'gamma', 'a', 'R', 'V_tip', 'g'],
                promotes_outputs=['a0'])
            self.add_subsystem(
                'rotor', NumericalRotorGroup(airfoil=DiscAirfoilGroup(**grid),
                                             trim=True, inflow='imposed',
                                             **grid),
                promotes_inputs=['mu', 'lambda_p', 'theta_0', 'theta_1',
                                 'V_tip', 'V_son', 'c_R', 'B', 'x_0', 'gamma',
                                 'a', 'vi_OR', 'a0'],
                promotes_outputs=[('CT_sigma', 'CT_sigma_rotor'), 'CQ_sigma',
                                  'CH_sigma', 'alpha_1270', 'CQ0_sigma_max',
                                  ('A_1', 'A1_b1s'), ('B_1', 'B1_a1s')])
            self.add_subsystem(
                'thrust_match',
                om.ExecComp('res_theta = SCALE * (CT_sigma_rotor - CT_sigma)',
                            res_theta=np.zeros(nn),
                            CT_sigma_rotor=np.zeros(nn),
                            CT_sigma=np.zeros(nn), SCALE=1.0e2),
                promotes=['*'])

        self.add_subsystem(
            'h_force_dim',
            om.ExecComp('H_M = CH_sigma * rho * A_b * V_tip**2',
                        H_M={'val': np.zeros(nn), 'units': 'lbf'},
                        CH_sigma=np.zeros(nn),
                        rho={'val': 0.002377, 'units': 'slug/ft**3'},
                        A_b={'val': 240.0, 'units': 'ft**2'},
                        V_tip={'val': 650.0, 'units': 'ft/s'}),
            promotes=['*'])
        power = ('CQ_sigma_total'
                 if self.options['rotor_options'].get('compressibility', False)
                 and self.options['rotor'] == 'closed_form'
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

        # ------------------------------------------------ force equilibrium
        self.add_subsystem('tpp_angle',
                           TppAngleComp(num_nodes=nn, form='forces'),
                           promotes=['*'])
        self.add_subsystem('thrust_coef', ThrustCoefComp(num_nodes=nn),
                           promotes_inputs=['T', 'rho', 'A_b', 'V_tip'],
                           promotes_outputs=[('CT_sigma', 'CT_sigma_computed')])
        self.add_subsystem(
            'inflow_computed',
            om.ExecComp('lambda_p_computed = mu * alpha_TPP - vi_OR',
                        lambda_p_computed=np.zeros(nn), mu=np.zeros(nn),
                        alpha_TPP={'val': np.zeros(nn), 'units': 'rad'},
                        vi_OR=np.zeros(nn)),
            promotes=['*'])
        self.add_subsystem('inflow_out',
                           FuselageAngleComp(num_nodes=nn, mode='from_inflow'),
                           promotes_inputs=['mu', 'i_s', 'a1s', 'vi_OR'],
                           promotes_outputs=[('alpha_F', 'alpha_F_computed'),
                                             'alpha_DW'])
        self.connect('lambda_p_computed', 'inflow_out.lambda_p')

        # ------------------------------------------------------- residuals
        residuals = ['res_alpha_F = alpha_F_computed - alpha_F',
                     'res_CT_sigma = CT_sigma_computed - CT_sigma']
        kwargs = dict(
            res_alpha_F={'val': np.zeros(nn), 'units': 'rad'},
            alpha_F_computed={'val': np.zeros(nn), 'units': 'rad'},
            alpha_F={'val': np.zeros(nn), 'units': 'rad'},
            res_CT_sigma=np.zeros(nn), CT_sigma_computed=np.zeros(nn),
            CT_sigma=np.zeros(nn))
        if mode == 'autorotation':
            # C_Q/sigma is O(1e-3) while the other two residuals are O(1e-1),
            # so Newton would take enormous steps in gamma; scale it up to
            # roughly horsepower, which puts all three residuals on a par
            residuals.append('res_CQ_sigma = CQ_SCALE * '
                             '(CQ_sigma_rotor - CQ_sigma_target)')
            kwargs.update(res_CQ_sigma=np.zeros(nn),
                          CQ_sigma_rotor=np.zeros(nn),
                          CQ_sigma_target=np.zeros(nn),
                          CQ_SCALE=1.0e4)
        promotes = ['*'] if mode != 'autorotation' else [
            n for n in ('res_alpha_F', 'alpha_F_computed', 'alpha_F',
                        'res_CT_sigma', 'CT_sigma_computed', 'CT_sigma',
                        'res_CQ_sigma', 'CQ_sigma_target')]
        self.add_subsystem('residuals', om.ExecComp(residuals, **kwargs),
                           promotes=promotes)
        if mode == 'autorotation':
            self.connect(power, 'residuals.CQ_sigma_rotor')

        # --------------------------------------------------------- solvers
        # solve_subsystems must be ON when the rotor is G2: G2 carries its own
        # Newton for A_1 and B_1, and with subsystem solves off the outer
        # iteration never runs it, so every residual is evaluated on an
        # untrimmed disc. The symptom is spectacular rather than subtle --
        # lambda' ran to -7.9 and theta_0 to 220 deg -- but the cause is the
        # same one-word setting that bit G2 itself, see validation_notes
        # section 5.
        numerical = self.options['rotor'] == 'numerical'
        newton = self.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=numerical, max_sub_solves=100)
        for key, value in dict(maxiter=60, atol=1e-10, rtol=1e-12,
                               iprint=0, err_on_non_converge=True).items():
            newton.options[key] = value
        # 'scalar' clips each state at its own bound; 'vector' rescales the
        # whole step, which stalls here because gamma is the only bounded
        # state and it would drag alpha_F and C_T/sigma back with it
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar',
                                                 maxiter=10, iprint=0)
        self.linear_solver = om.DirectSolver()
