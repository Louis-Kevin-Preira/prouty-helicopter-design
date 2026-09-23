"""Lateral-directional trim in forward flight, solved.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", pp. 531-538. Equations in Table 8.11
pp. 536-537, results in Figure 8.31 p. 538.
"""

import numpy as np
import openmdao.api as om

from .control_positions_comp import ControlPositionsComp
from .lat_cyclic_pitch_comp import LatCyclicPitchComp
from .lat_n_equilibrium_comp import LatNEquilibriumComp
from .lat_r_equilibrium_comp import LatREquilibriumComp
from .lat_y_equilibrium_comp import LatYEquilibriumComp
from .trim_elements_group import TrimElementsGroup


class LateralDirectionalTrimGroup(om.Group):
    """Y, R and N closed on three lateral unknowns, pp. 531-538.

    Same shape as ``LongitudinalTrimGroup``: the element group computes the
    contributions, three components sum them, a ``BalanceComp`` drives the
    residuals to zero. What differs is that there are four candidate
    unknowns for three equations, and which one is held fixed is a choice
    about how the helicopter is being flown.

    The three modes, p. 535
    -----------------------
    ============= ============ ==========================
    ``mode``       held fixed   solved for
    ============= ============ ==========================
    given_sideslip ``beta``     b1s_M, Phi, T_T
    zero_sideslip  ``beta = 0`` b1s_M, Phi, T_T
    zero_bank      ``Phi = 0``  b1s_M, beta, T_T
    ============= ============ ==========================

    p. 535 explains why both of the last two matter. Few helicopters have a
    sideslip indicator, so pilots fly at zero bank where they are
    comfortable, and find whatever sideslip makes the fuselage sideforce
    balance the tail rotor. Instrumented aircraft on test are asked to fly
    at zero sideslip instead, to minimise drag. Real flight is neither.

    ``zero_sideslip`` is ``given_sideslip`` with ``beta`` pinned at zero, and
    shares its implementation; it exists as a separate name because that is
    how the flight condition is described, not because the arithmetic
    differs.

    Residual pairing
    ----------------
    ``R`` fixes ``b1s_M``, since the roll stiffness ``dR_M/db1s + T_M h_M``
    is 355,485 and nothing else in that equation is close. ``N`` fixes
    ``T_T``, being the torque balance. ``Y`` then fixes whichever of ``Phi``
    and ``beta`` is free: the weight tilt is 20,000 per radian and the
    sideslip term 13,068, and both dominate what is left.

    The tail rotor is an unknown here
    ---------------------------------
    ``thrust='given'`` is forced on the element group. In the longitudinal
    problem the tail rotor thrust comes from the antitorque relation of
    p. 487; here it is what the N equation is solved for, and using both
    would impose that equation twice.

    Anchors
    -------
    Figure 8.31 p. 538, example helicopter at 115 knots, annotates two
    points:

    ================= ========== ========== ==========
    condition          b1s_M      T_T        other
    ================= ========== ========== ==========
    ``beta = 0``       -0.78 deg  661 lb     Phi -1.9 deg
    ``Phi = 0``        -0.89 deg  789 lb     beta 3.2 deg
    ================= ========== ========== ==========

    The sign of the weight term in the Y equation follows p. 531 and not
    Table 8.11; see C8-10 in ``docs/validation_trim.md``.

    Control positions
    -----------------
    ``LatCyclicPitchComp`` turns ``b1s_M`` into ``A_1`` and the Appendix A
    rigging of Figure A.5 turns that into a lateral stick position, which is
    what Figure 8.31 actually plots. At zero sideslip it gives 42.3 % against
    a printed 44, and that two point agreement is what settles C8-11: the
    other reading of p. 535 puts the stick at 51.4 %.

    ``A_1`` needs ``A1_b1s`` and ``B_1`` from elsewhere -- Chapter 3 for the
    first, the longitudinal solution for the second. The pedal needs
    ``theta_75_T``, which Chapter 8 never computes: p. 535 sends it to the
    Chapter 3 tail rotor method, given the ``alpha_TPP_T`` of
    ``TailRotorTppAngleComp``. It is a free input here.

    Frozen from the longitudinal solution
    -------------------------------------
    ``T_M``, ``Theta`` and ``a1s_M`` are inputs, not unknowns. p. 516 treats
    the two sets as independent, and Table 8.11 bakes that in: its Y row
    carries ``-(H + T_M a1s_M + T_M i_M) beta`` built on the *converged*
    longitudinal flapping. Feed them from ``LongitudinalTrimGroup``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)
        self.options.declare(
            'mode', default='given_sideslip',
            values=('given_sideslip', 'zero_sideslip', 'zero_bank'))
        self.options.declare('blade_closest', values=('up', 'down'),
                             default='up')
        self.options.declare('downwash_model',
                             values=('input', 'vortex_axis'), default='input')
        self.options.declare('tail_rotor_feedback', types=bool, default=False)
        self.options.declare('charts', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        lin = self.options['linearized']
        mode = self.options['mode']
        ones = np.ones(nn)

        # l_T and l_V reach both the fin sidewash chain and the N equation
        for name in ('l_T', 'l_V'):
            self.set_input_defaults(name, val=1.0, units='ft')

        # -------------------------------------------------------- states
        balance = om.BalanceComp()
        balance.add_balance(
            'b1s_M', val=-0.02 * ones, units='rad', lhs_name='res_R',
            rhs_val=np.zeros(nn), eq_units='lbf*ft', normalize=False,
            ref=0.1, res_ref=1000.0, lower=-0.5, upper=0.5)
        balance.add_balance(
            'T_T', val=700.0 * ones, units='lbf', lhs_name='res_N',
            rhs_val=np.zeros(nn), eq_units='lbf*ft', normalize=False,
            ref=1000.0, res_ref=10000.0, lower=-5000.0, upper=10000.0)

        if mode == 'zero_bank':
            # Phi is pinned, the sideslip it takes to hold it is the unknown
            balance.add_balance(
                'beta', val=0.05 * ones, units='rad', lhs_name='res_Y',
                rhs_val=np.zeros(nn), eq_units='lbf', normalize=False,
                ref=0.1, res_ref=100.0, lower=-0.6, upper=0.6)
            self.set_input_defaults('Phi', val=np.zeros(nn), units='rad')
        else:
            balance.add_balance(
                'Phi', val=-0.03 * ones, units='rad', lhs_name='res_Y',
                rhs_val=np.zeros(nn), eq_units='lbf', normalize=False,
                ref=0.1, res_ref=100.0, lower=-0.8, upper=0.8)
            self.set_input_defaults('beta', val=np.zeros(nn), units='rad')

        self.add_subsystem('balance', balance, promotes=['*'])

        # ------------------------------------------------------ elements
        self.add_subsystem(
            'elements',
            TrimElementsGroup(
                num_nodes=nn, linearized=lin, thrust='given',
                blade_closest=self.options['blade_closest'],
                downwash_model=self.options['downwash_model'],
                tail_rotor_feedback=self.options['tail_rotor_feedback'],
                charts=self.options['charts']),
            promotes=['*'])

        # --------------------------------------------------- equilibrium
        self.add_subsystem('y_equilibrium',
                           LatYEquilibriumComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])
        self.add_subsystem('r_equilibrium',
                           LatREquilibriumComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])
        self.add_subsystem('n_equilibrium',
                           LatNEquilibriumComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])

        # ------------------------------------------------------ controls
        # outside the loop: they follow from the solution, nothing feeds back
        self.add_subsystem('cyclic', LatCyclicPitchComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('controls',
                           ControlPositionsComp(num_nodes=nn,
                                                channels=('lat', 'pedal')),
                           promotes=['*'])

        self.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True, maxiter=40, atol=1e-10, rtol=1e-12,
            iprint=0, err_on_non_converge=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
            bound_enforcement='vector', maxiter=6, iprint=0)
        self.linear_solver = om.DirectSolver()
