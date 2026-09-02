"""
WindTunnelRotorGroup -- an isolated rotor at a prescribed shaft angle, with
wind tunnel wall corrections.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 247-249 (Table 3.5, case 9), validated against test 276 run 3 of
reference 3.27, an H-34 rotor in the 40 by 80 foot tunnel.

This is the only case in the chapter checked against measurement rather than
against the book, which makes it worth more than the other eight put together.

What is prescribed and what is solved. A tunnel run fixes the shaft angle and
the collective; thrust is whatever the rotor makes. So alpha_TPP is an input
here -- there is no weight to balance -- and C_T/sigma is the unknown, closing
through

    lambda' = mu alpha_TPP - sigma (C_T/sigma) / (2 mu)
    C_T/sigma = rotor(theta_0, lambda')

Prouty solves it graphically, plotting the two lambda' curves against
C_T/sigma and reading the intersection (the inset on p. 247). One balance does
the same thing, and the wall correction folds into the same loop: the corrected
disc angle depends on C_T/sigma, which depends on the corrected angle.

Collective. The tunnel reports theta_.75, so theta_0 = theta_.75 - 0.75
theta_1: 9 deg at the 75 % station with -8 deg of twist is 15 deg of
collective. p. 247 step b instead converts to the chart's -5 deg twist, which
is a chart artefact and has no place here -- the rotor group takes theta_1
directly. See validation_notes on why that conversion is not exact anyway.

Flapping. The charts give the combinations, so the book recovers
A_1 = (A_1 - b_1s) + b_1s and B_1 = (B_1 + a_1s) - a_1s using the measured
flapping. Same here: a_1s and b_1s are inputs, from the run.

Measured against computed, p. 249:

    C_T/sigma    0.110 test   0.106 book
    C_P/sigma    0.0040       0.0037
    C_XR/sigma   -0.0058      -0.0065
    A_1          -3.3 deg     -2.8 deg
    B_1          10 deg       9.9 deg

The thrust is 4 % low and the power 8 % low, which for a 1980s hand method
against a full scale rotor is respectable. Do not expect this group to do
better than the book against the measurement: it is the same physics, and
where it differs from the book -- notably in needing an assumed c_d where the
book reads C_Q/sigma off a chart -- it can only be worse.

    theta_.75, alpha_s, a_1s, b_1s, mu, sigma, ... --> CT_sigma, CQ_sigma,
                                                       CXR_sigma, A_1, B_1
"""

import numpy as np
import openmdao.api as om

from .closed_form_rotor_group import ClosedFormRotorGroup
from .disc_airfoil_group import DiscAirfoilGroup
from .numerical_rotor_group import NumericalRotorGroup
from .coning_comp import ConingComp
from .induced_velocity_comp import InducedVelocityComp
from .inflow_comp import InflowComp
from .rotor_side_force_comp import RotorSideForceComp
from .tunnel_wall_correction_comp import TunnelWallCorrectionComp

THRUST_SCALE = 1.0e2


class WindTunnelRotorGroup(om.Group):
    """Isolated rotor at fixed shaft angle with wall corrections, p. 247-249."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('wall_correction', types=bool, default=True,
                             desc='apply the reference 3.47 correction; off '
                                  'gives the uncorrected tunnel result')
        self.options.declare('rotor', values=('closed_form', 'numerical'),
                             default='closed_form',
                             desc="'numerical' swaps G1 for G2, which computes "
                                  'the drag element by element instead of '
                                  'assuming a mean c_d. Case 9 is where that '
                                  'matters: at positive shaft angle the '
                                  'induced torque is NEGATIVE and C_Q is a '
                                  'small difference of cancelling terms, so '
                                  'all the information sits in the profile '
                                  'part.')
        self.options.declare('rotor_options', types=dict,
                             default={'compressibility': False})
        self.options.declare('grid', types=tuple, default=(12, 25),
                             desc='(num_azimuth, num_radial) when '
                                  "rotor='numerical'")

    def setup(self):
        nn = self.options['num_nodes']
        ones = np.ones(nn)

        self.set_input_defaults('mu', val=0.3 * ones)
        self.set_input_defaults('CT_sigma', val=0.10 * ones)
        self.set_input_defaults('sigma', val=0.062)
        self.set_input_defaults('theta_1', val=-0.13963, units='rad')
        if self.options['rotor'] == 'numerical':
            self.set_input_defaults('V_son', val=1116.0, units='ft/s')

        balance = om.BalanceComp()
        balance.add_balance('CT_sigma_ref', val=0.10 * ones,
                            lhs_name='res_CT', rhs_val=np.zeros(nn),
                            lower=0.005, upper=0.25)
        self.add_subsystem('balance', balance, promotes=['*'])

        self.add_subsystem(
            'disc_angle',
            om.ExecComp('alpha_TPP_uncorr = alpha_s + a1s',
                        alpha_TPP_uncorr={'val': np.zeros(nn), 'units': 'rad'},
                        alpha_s={'val': np.zeros(nn), 'units': 'rad'},
                        a1s={'val': np.zeros(nn), 'units': 'rad'}),
            promotes=['*'])
        self.add_subsystem(
            'collective',
            om.ExecComp('theta_0 = theta_75 - 0.75 * theta_1',
                        theta_0={'val': np.zeros(nn), 'units': 'rad'},
                        theta_75={'val': np.zeros(nn), 'units': 'rad'},
                        theta_1={'val': -0.13963, 'units': 'rad'}),
            promotes=['*'])

        if self.options['wall_correction']:
            self.add_subsystem(
                'walls', TunnelWallCorrectionComp(num_nodes=nn),
                promotes_inputs=['alpha_TPP_uncorr', ('CT_sigma',
                                                      'CT_sigma_ref'),
                                 'mu', 'sigma', 'area_ratio', 'delta_WL'],
                promotes_outputs=['chi', 'inflow_ratio', 'delta_alpha_TPP',
                                  ('alpha_TPP_corr', 'alpha_TPP')])
        else:
            self.add_subsystem(
                'no_walls',
                om.ExecComp('alpha_TPP = alpha_TPP_uncorr',
                            alpha_TPP={'val': np.zeros(nn), 'units': 'rad'},
                            alpha_TPP_uncorr={'val': np.zeros(nn),
                                              'units': 'rad'}),
                promotes=['*'])

        self.add_subsystem('induced_velocity', InducedVelocityComp(num_nodes=nn),
                           promotes_inputs=[('CT_sigma', 'CT_sigma_ref'),
                                            'mu', 'sigma'],
                           promotes_outputs=['vi_OR'])
        self.add_subsystem('inflow', InflowComp(num_nodes=nn), promotes=['*'])

        # ClosedFormRotorGroup carries its own ConingComp; G2 does not, and
        # takes a0 as an input. Leaving it unconnected is silent and ruinous:
        # the default is 1 rad, so the rotor runs with 57.3 deg of coning and
        # A_1 comes out at -20.9 deg instead of -2.8. Nothing raises, nothing
        # fails to converge, and the residuals read 1e-13.
        if self.options['rotor'] == 'numerical':
            self.add_subsystem(
                'coning', ConingComp(num_nodes=nn, form='ct'),
                promotes_inputs=[('CT_sigma', 'CT_sigma_ref'), 'gamma', 'a',
                                 'R', 'V_tip', 'g'],
                promotes_outputs=['a0'])

        if self.options['rotor'] == 'closed_form':
            self.add_subsystem(
                'rotor', ClosedFormRotorGroup(num_nodes=nn, mode='thrust',
                                              **self.options['rotor_options']),
                promotes=['*'])
        else:
            n_psi, n_r = self.options['grid']
            grid = dict(num_nodes=nn, num_azimuth=n_psi, num_radial=n_r)
            # inflow='imposed': the C_T loop belongs to this group, which also
            # owns the wall correction, so G2 only trims A_1 and B_1. Its
            # cyclic IS the chart's A_1 - b_1s and B_1 + a_1s, by the
            # flapping-feathering equivalence of p. 211.
            self.add_subsystem(
                'rotor', NumericalRotorGroup(airfoil=DiscAirfoilGroup(**grid),
                                             trim=True, inflow='imposed',
                                             **grid),
                promotes_inputs=['mu', 'lambda_p', 'theta_0', 'theta_1',
                                 'V_tip', 'V_son', 'c_R', 'B', 'x_0', 'gamma',
                                 'a', 'vi_OR', 'a0'],
                promotes_outputs=['CT_sigma', 'CQ_sigma', 'CH_sigma',
                                  ('A_1', 'A1_b1s'), ('B_1', 'B1_a1s')])

        self.add_subsystem(
            'residual',
            om.ExecComp('res_CT = SCALE * (CT_sigma - CT_sigma_ref)',
                        res_CT=np.zeros(nn), CT_sigma=np.zeros(nn),
                        CT_sigma_ref=np.zeros(nn), SCALE=THRUST_SCALE),
            promotes=['*'])

        self.add_subsystem('side_force', RotorSideForceComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem(
            'cyclic',
            om.ExecComp(['A_1 = A1_b1s + b1s', 'B_1 = B1_a1s - a1s'],
                        A_1={'val': np.zeros(nn), 'units': 'rad'},
                        B_1={'val': np.zeros(nn), 'units': 'rad'},
                        A1_b1s={'val': np.zeros(nn), 'units': 'rad'},
                        B1_a1s={'val': np.zeros(nn), 'units': 'rad'},
                        a1s={'val': np.zeros(nn), 'units': 'rad'},
                        b1s={'val': np.zeros(nn), 'units': 'rad'}),
            promotes=['*'])

        newton = self.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True, max_sub_solves=100)
        for key, value in dict(maxiter=50, atol=1e-12, rtol=1e-14,
                               iprint=0).items():
            newton.options[key] = value
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar',
                                                 maxiter=8, iprint=0)
        self.linear_solver = om.DirectSolver()
