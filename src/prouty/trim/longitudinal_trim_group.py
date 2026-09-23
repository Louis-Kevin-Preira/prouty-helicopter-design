"""Longitudinal trim in forward flight, solved.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", pp. 516-522. Equations in Table 8.4
pp. 518-521, elements in Table 8.5 pp. 523-525, solution on p. 522.
"""

import numpy as np
import openmdao.api as om

from .control_positions_comp import ControlPositionsComp
from .long_cyclic_pitch_comp import LongCyclicPitchComp
from .long_m_equilibrium_comp import LongMEquilibriumComp
from .long_x_equilibrium_comp import LongXEquilibriumComp
from .long_z_equilibrium_comp import LongZEquilibriumComp
from .trim_elements_group import TrimElementsGroup


class LongitudinalTrimGroup(om.Group):
    """X, Z and M closed on the three longitudinal unknowns, pp. 516-522.

    ``TrimElementsGroup`` computes what every component contributes, the
    three equilibrium components sum them, and a ``BalanceComp`` drives the
    three residuals to zero::

        T_M    <-- res_Z
        Theta  <-- res_X
        a1s_M  <-- res_M

    Why that pairing
    ----------------
    Newton solves the coupled system and does not need the states matched to
    residuals, but the Jacobian is far better conditioned when they are, and
    this pairing is the one p. 517 describes. The Z equation is
    ``-T_M cos(a1s_M + i_M)`` plus 425 lb of everything else, so it fixes the
    thrust almost alone. The X equation is fuselage drag against rotor
    H-force, and the weight tilt ``-G.W. sin(Theta)`` is what balances the
    remainder, so it fixes the attitude. The M equation is where the hub
    moment lives, so it fixes the flapping.

    p. 516 offers two methods: write the three as linear functions of the
    three unknowns and solve simultaneously, or iterate. This is the second,
    which p. 516 calls well suited to computers and not constrained to linear
    functions. Setting ``linearized=True`` recovers the first, since the
    residuals then are the linear functions of Table 8.4 — the same answer by
    the route Prouty took by hand.

    Options
    -------
    num_nodes : int
    linearized : bool
        Passed down to the elements and the equilibrium equations.
        ``True`` reproduces Table 8.4 term for term, omissions included.
    thrust : {'antitorque', 'given'}
        Default ``'antitorque'``: the tail rotor thrust is not one of the
        three longitudinal unknowns, and p. 487 supplies it.
    blade_closest, downwash_model, tail_rotor_feedback, charts
        Passed to ``TrimElementsGroup``. ``charts=True`` reads the digitised
        abaques instead of taking Prouty's readings of them.

    Scaling is not decoration
    -------------------------
    The three states span six orders of magnitude, 20,000 lb against
    0.02 rad, and the three residuals four, 20,000 lb-ft against a pound.
    An unscaled Newton takes a first step that moves ``T_M`` by a fraction of
    a pound and ``Theta`` past a radian. The ``ref`` and ``res_ref`` below
    put every state and every residual near unity.

    Bounds are also not decoration. ``Theta`` is bounded to a quarter turn
    because the weight tilt ``-G.W. sin(Theta)`` is periodic: without a
    bracket the residual has roots every ``2 pi`` and Newton is content to
    find one.

    Anchor
    ------
    p. 522, example helicopter at 115 knots:

    ======= ============ ==================
    T_M      20,586 lb
    Theta    -0.0165 rad  -0.9 deg
    a1s_M    -0.019 rad   -1.1 deg
    B_1                    8.9 deg
    pct_B1                 38 % from full forward
    ======= ============ ==================

    ``B_1`` needs ``B1_a1s`` from Chapter 3, 7.8 deg here, and the stick
    position follows from it through the Appendix A rigging of Figure A.5.
    Both sit outside the Newton loop: they follow from the trim and nothing
    feeds back, so solving for them would only slow the loop down.

    The stick position is where the answer becomes checkable against a
    flying qualities specification, since p. 530 writes its control shift
    criterion in inches and not in degrees of cyclic. ``in_B1`` carries
    that.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)
        self.options.declare('thrust', values=('given', 'antitorque'),
                             default='antitorque')
        self.options.declare('blade_closest', values=('up', 'down'),
                             default='up')
        self.options.declare('downwash_model',
                             values=('input', 'vortex_axis'), default='input')
        self.options.declare('tail_rotor_feedback', types=bool, default=False)
        self.options.declare('charts', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        lin = self.options['linearized']
        ones = np.ones(nn)

        # l_T and l_V reach both the antitorque relation and the moment
        # equation; same lever arms, different declared defaults
        for name in ('l_T', 'l_V'):
            self.set_input_defaults(name, val=1.0, units='ft')

        # -------------------------------------------------------- states
        balance = om.BalanceComp()
        balance.add_balance(
            'T_M', val=20000.0 * ones, units='lbf', lhs_name='res_Z',
            rhs_val=np.zeros(nn), eq_units='lbf', normalize=False,
            ref=20000.0, res_ref=1000.0, lower=1000.0, upper=100000.0)
        balance.add_balance(
            'Theta', val=-0.02 * ones, units='rad', lhs_name='res_X',
            rhs_val=np.zeros(nn), eq_units='lbf', normalize=False,
            ref=0.1, res_ref=100.0, lower=-0.8, upper=0.8)
        balance.add_balance(
            'a1s_M', val=-0.02 * ones, units='rad', lhs_name='res_M',
            rhs_val=np.zeros(nn), eq_units='lbf*ft', normalize=False,
            ref=0.1, res_ref=1000.0, lower=-0.5, upper=0.5)
        self.add_subsystem('balance', balance, promotes=['*'])

        # ------------------------------------------------------ elements
        self.add_subsystem(
            'elements',
            TrimElementsGroup(
                num_nodes=nn, linearized=lin,
                thrust=self.options['thrust'],
                blade_closest=self.options['blade_closest'],
                downwash_model=self.options['downwash_model'],
                tail_rotor_feedback=self.options['tail_rotor_feedback'],
                charts=self.options['charts']),
            promotes=['*'])

        # --------------------------------------------------- equilibrium
        self.add_subsystem('x_equilibrium',
                           LongXEquilibriumComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])
        self.add_subsystem('z_equilibrium',
                           LongZEquilibriumComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])
        self.add_subsystem('m_equilibrium',
                           LongMEquilibriumComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])

        # ------------------------------------------------------ controls
        # outside the loop: B_1 follows from the solution, nothing feeds back
        self.add_subsystem('cyclic', LongCyclicPitchComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('controls',
                           ControlPositionsComp(num_nodes=nn,
                                                channels=('long',)),
                           promotes=['*'])

        self.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True, maxiter=40, atol=1e-10, rtol=1e-12,
            iprint=0, err_on_non_converge=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
            bound_enforcement='vector', maxiter=6, iprint=0)
        self.linear_solver = om.DirectSolver()
