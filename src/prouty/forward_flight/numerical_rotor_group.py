"""
NumericalRotorGroup (G2) -- blade element numerical integration.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 208-228.

Where G1 integrates the rotor analytically under a list of assumptions, this
group integrates it numerically under none of them: reverse flow, stall, sweep,
compressibility and the dynamic stall delay all fall out of evaluating the
blade element equations at every point of a (psi, r/R) grid.

Execution is one pass through the disc, then a two-state trim:

    grid, pitch                                  p. 209-213
    U_T, U_P, U_R, U_B, U_TR                     p. 209-213
    M, alpha with the quadrant rule              p. 214
    sweep, dynamic stall delay                   p. 218-221
    airfoil model, then the p. 221 bounds on c_l
    c_N and c_c                                  p. 209-212
    the five loadings, integrated                p. 209-212
    gyroscopic moments, trim on A_1 and B_1      p. 214
    stall indicators for the chart limit lines   p. 228

Inflow. p. 213 closes the loop by iteration: "the value of C_T/sigma is
initially either known or can be estimated from the closed-form equations. The
value will be updated after the first full cycle of calculations." Three
quantities depend on it -- the induced velocity through the momentum relation
v_1/Omega R = (C_T/sigma) sigma / 2 mu, the coning angle through
a_0 = (2/3) gamma (C_T/sigma)/a - (3/2) g R/(Omega R)^2, and hence U_P -- so
running the disc with a C_T/sigma taken from elsewhere means running it with
the inflow of a different rotor.

lambda' stays an INPUT, which is what makes this the analysis direction rather
than a trim: p. 225 lists the inputs of the program that produced the isolated
rotor charts as "lambda', theta_0, theta_1, mu and M_1,90", and lambda' is an
axis of the charts themselves (p. 255). C_T/sigma is therefore the third state,
with residual C_T/sigma - C_T/sigma_computed, and the chain stays acyclic:
the state feeds v_1/Omega R and a_0, those feed U_P, and the disc integral
returns the value to compare. InducedVelocityComp and ConingComp are the same
components G1 uses, not copies.

Setting inflow='imposed' leaves v_1/Omega R and a_0 as inputs, which is how
this group behaved before the loop existed and is useful for isolating the
effect of the inflow from everything else.

Trim. p. 211: "the equivalence of flapping and feathering allows performance
calculations to be based on a rigid rotor whose tip path plane is
perpendicular to the shaft and whose pitching and rolling moments are trimmed
out with cyclic pitch". So the disc does not flap here; A_1 and B_1 are the
two unknowns, and the residuals are the moment balances of p. 214,

    C_M/sigma + C_M/sigma_gyro = 0
    C_R/sigma + C_R/sigma_gyro = 0

which reduce to zero aerodynamic moment in steady flight. Setting trim=False
leaves A_1 and B_1 as inputs, which is what the isolated rotor charts need
when they are swept in collective rather than in moment.

The airfoil model is injected. It must accept the field-shaped inputs alpha
(rad) and M and return cl_raw and cd on the same shape; sec_Lambda and
d_alpha_stall are promoted alongside so a Chapter 6 model can raise its stall
angle by sweep and pitch rate, and a model that ignores them still works.

Residual scaling. The moment coefficients are 1e-4 to 1e-3 while the cyclic
angles are 1e-2, so the raw residuals are three orders below the states.
MOMENT_SCALE puts them on a par; without it Newton's first step overshoots by
enough to leave the disc.

Known departures from the printed text, both measured and documented in the
components: the induced velocity distribution takes cos(psi) rather than the
sin(psi) of p. 209, and the spanwise skin friction term of the H-force uses
U_TR rather than U_B^2/U_T, which changes C_H/sigma by a factor of 43 and its
sign.
"""

import numpy as np
import openmdao.api as om

from .alpha_comp import AlphaComp
from .chord_force_comp import ChordForceComp
from .disc_integral_comp import DiscIntegralComp
from .coning_comp import ConingComp
from .gyro_moment_comp import GyroMomentComp
from .induced_velocity_comp import InducedVelocityComp
from .lift_coef_bounds_comp import LiftCoefBoundsComp
from .loadings_comp import LoadingsComp
from .local_mach_comp import LocalMachComp
from .normal_force_comp import NormalForceComp
from .pitch_dist_comp import PitchDistComp
from .resultant_vel_comp import ResultantVelComp
from .rotor_disc_grid_comp import RotorDiscGridComp
from .stall_delay_comp import StallDelayComp
from .stall_indicators_comp import StallIndicatorsComp
from .sweep_angle_comp import SweepAngleComp
from .unsteady_lift_comp import UnsteadyLiftComp
from .unsteady_lift_comp import UnsteadyLiftComp
from .unsteady_lift_comp import UnsteadyLiftComp
from .velocity_comps import PerpVelComp, RadialVelComp, TangentialVelComp

MOMENT_SCALE = 1.0e3
THRUST_SCALE = 1.0e1

# C_T/sigma is allowed to go negative. A rotor at low collective with -10 deg
# of twist genuinely produces downward thrust -- the disc returns -0.10 at
# theta_0 = 4 deg -- and a lower bound at zero makes the residual infeasible
# there rather than merely hard to solve.
CT_BOUNDS = (-0.15, 0.25)

FIELD_CHAIN = (
    ('grid', RotorDiscGridComp),
    ('pitch', PitchDistComp),
    ('tangential_vel', TangentialVelComp),
    ('perp_vel', PerpVelComp),
    ('radial_vel', RadialVelComp),
    ('resultant_vel', ResultantVelComp),
    ('mach', LocalMachComp),
    ('alpha', AlphaComp),
    ('sweep', SweepAngleComp),
    ('stall_delay', StallDelayComp),
)

FORCE_CHAIN = (
    ('normal_force', NormalForceComp),
    ('chord_force', ChordForceComp),
    ('loadings', LoadingsComp),
    ('integral', DiscIntegralComp),
)


class NumericalRotorGroup(om.Group):
    """G2 -- numerical integration over the rotor disc."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('trim', types=bool, default=True,
                             desc='solve A_1 and B_1 for zero hub moment')
        self.options.declare('unsteady', types=bool, default=False,
                             desc='add the unsteady lift increment of '
                                  'p. 224-225 to the steady coefficient '
                                  'before the p. 221 bounds are applied')
        self.options.declare('unsteady', types=bool, default=False,
                             desc='add the shed vorticity correction of '
                                  'p. 224-225 to the steady lift coefficient')
        self.options.declare('unsteady', types=bool, default=False,
                             desc='add the shed vorticity correction of '
                                  'p. 224-225 to the steady lift coefficient')
        self.options.declare('inflow', values=('momentum', 'imposed'),
                             default='momentum',
                             desc="'momentum' solves C_T/sigma so that "
                                  'v_1/Omega R and a_0 are consistent with it '
                                  "(p. 213); 'imposed' takes both as inputs")
        self.options.declare('airfoil', recordable=False,
                             desc='alpha, M -> cl_raw, cd on the field shape')
        self.options.declare('component_options', types=dict, default={},
                             desc='per-component overrides, keyed by name')

    def setup(self):
        nn = self.options['num_nodes']
        grid = dict(num_nodes=nn, num_azimuth=self.options['num_azimuth'],
                    num_radial=self.options['num_radial'])
        field = (nn, self.options['num_azimuth'], self.options['num_radial'])
        shape_field = (nn, self.options['num_azimuth'],
                       self.options['num_radial'])
        extra = self.options['component_options']
        grid_shape = (nn, self.options['num_azimuth'],
                      self.options['num_radial'])

        momentum = self.options['inflow'] == 'momentum'
        balance = om.BalanceComp()
        if self.options['trim']:
            for state, residual in (('A_1', 'res_CM'), ('B_1', 'res_CR')):
                balance.add_balance(state, val=np.zeros(nn), units='rad',
                                    lhs_name=residual, rhs_val=np.zeros(nn),
                                    lower=-0.5, upper=0.5)
        if momentum:
            balance.add_balance('CT_sigma_ref', val=0.08 * np.ones(nn),
                                lhs_name='res_CT', rhs_val=np.zeros(nn),
                                lower=CT_BOUNDS[0], upper=CT_BOUNDS[1])
        if self.options['trim'] or momentum:
            self.add_subsystem('balance', balance, promotes=['*'])

        if momentum:
            self.add_subsystem(
                'induced_velocity', InducedVelocityComp(num_nodes=nn,
                                                        **extra.get(
                                                            'induced_velocity',
                                                            {})),
                promotes_inputs=[('CT_sigma', 'CT_sigma_ref'), 'mu', 'sigma'],
                promotes_outputs=['vi_OR'])
            self.add_subsystem(
                'coning', ConingComp(num_nodes=nn, form='ct'),
                promotes_inputs=[('CT_sigma', 'CT_sigma_ref'), 'gamma', 'a',
                                 'R', 'V_tip', 'g'],
                promotes_outputs=['a0'])

        for name, comp in FIELD_CHAIN:
            self.add_subsystem(name, comp(**grid, **extra.get(name, {})),
                               promotes=['*'])

        self.add_subsystem('airfoil', self.options['airfoil'],
                           promotes_inputs=['alpha', 'M', 'sec_Lambda',
                                            'd_alpha_stall'],
                           promotes_outputs=['cl_raw', 'cd'])
        bounded_input = 'cl_raw'
        if self.options['unsteady']:
            self.add_subsystem('unsteady',
                               UnsteadyLiftComp(**grid,
                                                **extra.get('unsteady', {})),
                               promotes=['*'])
            self.add_subsystem(
                'cl_total',
                om.ExecComp('cl_steady_plus = cl_raw + d_cl_unsteady',
                            cl_steady_plus=np.zeros(shape_field),
                            cl_raw=np.zeros(shape_field),
                            d_cl_unsteady=np.zeros(shape_field)),
                promotes=['*'])
            bounded_input = 'cl_steady_plus'

        self.add_subsystem('bounds', LiftCoefBoundsComp(**grid),
                           promotes_inputs=['alpha', ('cl', bounded_input)],
                           promotes_outputs=[('cl_bounded', 'cl'),
                                             'cl_max', 'cl_min'])

        for name, comp in FORCE_CHAIN:
            self.add_subsystem(name, comp(**grid, **extra.get(name, {})),
                               promotes=['*'])

        self.add_subsystem('gyro', GyroMomentComp(num_nodes=nn), promotes=['*'])

        # p. 228: the two chart limit lines. They cost nothing to carry and
        # G3 needs them at every point it plots, so they are part of the
        # group rather than something the caller assembles afterwards.
        self.add_subsystem('stall_indicators',
                           StallIndicatorsComp(num_nodes=nn,
                                               num_azimuth=self.options[
                                                   'num_azimuth'],
                                               **extra.get('stall_indicators',
                                                           {})),
                           promotes=['*'])

        expressions, kwargs = [], {}
        if self.options['trim']:
            expressions += ['res_CM = MSCALE * (CM_sigma + CM_sigma_gyro)',
                            'res_CR = MSCALE * (CR_sigma + CR_sigma_gyro)']
            kwargs.update(res_CM=np.zeros(nn), res_CR=np.zeros(nn),
                          CM_sigma=np.zeros(nn), CR_sigma=np.zeros(nn),
                          CM_sigma_gyro=np.zeros(nn),
                          CR_sigma_gyro=np.zeros(nn), MSCALE=MOMENT_SCALE)
        if momentum:
            expressions.append('res_CT = TSCALE * (CT_sigma - CT_sigma_ref)')
            kwargs.update(res_CT=np.zeros(nn), CT_sigma=np.zeros(nn),
                          CT_sigma_ref=np.zeros(nn), TSCALE=THRUST_SCALE)

        if expressions:
            self.add_subsystem('residuals', om.ExecComp(expressions, **kwargs),
                               promotes=['*'])

            # solve_subsystems is not optional here. With it off, Newton
            # treats every intermediate field as a state and its first step
            # drives U_B negative, which puts a NaN in the stall delay's square
            # root. With it on, the explicit chain is re-run in sequence before
            # each residual evaluation and the system reduces to the two
            # cyclic angles, which is what it actually is.
            newton = self.nonlinear_solver = om.NewtonSolver(
                solve_subsystems=True, max_sub_solves=100)
            # 40 was not enough: at theta_0 = 24 deg the residuals reach zero
            # on the fortieth iteration exactly, so the solver reported failure
            # on a converged answer. The system is stiff wherever a large part
            # of the disc is stalled.
            for key, value in dict(maxiter=150, atol=1e-12, rtol=1e-14,
                                   iprint=0).items():
                newton.options[key] = value
            newton.linesearch = om.ArmijoGoldsteinLS(
                bound_enforcement='scalar', maxiter=8, iprint=0)
            self.linear_solver = om.DirectSolver()
