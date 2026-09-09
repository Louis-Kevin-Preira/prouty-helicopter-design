"""Forces and moments of every airframe component, given the trim unknowns.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", pp. 485-515. Anchors in Table 8.4
pp. 518-521, Table 8.5 pp. 523-525 and Table 8.11 pp. 536-537.
"""

import numpy as np
import openmdao.api as om

from .dyn_pressure_increase_comp import DynPressureIncreaseComp
from .end_plate_factor_comp import EndPlateFactorComp
from .fuselage_alpha_comp import FuselageAlphaComp
from .fuselage_derivatives_comp import FuselageDerivativesComp
from .fuselage_downwash_horiz_stab_comp import FuselageDownwashAtHorizStabComp
from .fuselage_forces_comp import FuselageForcesComp
from .fuselage_sidewash_comp import FuselageSidewashComp
from .horiz_stab_alpha_comp import HorizStabAlphaComp
from .interference_factor_comp import InterferenceFactorComp
from .lift_curve_slope_comp import LiftCurveSlopeComp
from .span_efficiency_comp import SpanEfficiencyComp
from .horiz_stab_forces_comp import HorizStabForcesComp
from .horiz_stab_lift_drag_comp import HorizStabLiftDragComp
from .main_rotor_forces_comp import MainRotorForcesComp
from .rotor_downwash_comp import RotorDownwashComp
from .tail_rotor_forces_comp import TailRotorForcesComp
from .tail_rotor_sidewash_comp import TailRotorSidewashComp
from .vert_stab_effective_ar_comp import VertStabEffectiveARComp
from .vert_stab_forces_comp import VertStabForcesComp
from .vert_stab_interference_drag_comp import VertStabInterferenceDragComp
from .vert_stab_lift_drag_comp import VertStabLiftDragComp


class TrimElementsGroup(om.Group):
    """The whole of pp. 485-515, wired.

    Takes the trim unknowns and returns what each component contributes to
    the six equations of equilibrium::

        T_M, a1s_M, b1s_M, Theta, beta, T_T
            --> X, Y, Z, R, M, N  for  M, T, H, V, F

    It solves nothing. The two solution sections put a ``BalanceComp`` around
    it, one on the longitudinal three and one on the lateral three, and this
    group is what their residuals are assembled from.

    The chain, in execution order
    -----------------------------
    Rotor downwash reaches three stations, and each one needs a different
    ``v/v1``: 1.5 at both stabilisers and 1.0 at the fuselage, per Table 8.5
    and Table 8.4. The fuselage angle of attack follows from its own
    downwash, and then feeds the fuselage-induced downwash at both
    stabilisers, so it has to come early::

        eps_MH, eps_MF, eps_MV      three RotorDownwashComp
        alpha_F                     from eps_MF
        eps_FH, eps_FV              from alpha_F
        X_M .. N_M                  main rotor
        T_T, X_T .. M_T             tail rotor
        eta_TV, eta_FV              sidewash at the fin
        alpha_H, L_H, D_H, X_H, Z_H horizontal stabiliser
        dD_int, L_V, D_V, X_V .. Z_V  vertical stabiliser
        L_F .. R_F, X_F, Y_F, Z_F   fuselage

    With ``tail_rotor_feedback=False`` that order is acyclic and the group
    runs once through.

    The loop, and why it is open by default
    ---------------------------------------
    Two of the relations close a circle. p. 487 makes the tail rotor thrust
    depend on the fin side force, ``T_T = Q_M/l_T - Y_V l_V/l_T``; p. 509
    makes the fin's interference drag depend on both; and the fin side force
    depends on the tail rotor through its sidewash and on that drag through
    its own force resolution. p. 510 breaks it by taking ``Y_V`` at the value
    it has with no interference.

    ``tail_rotor_feedback=False``, the default, does exactly that: the two
    consumers take a separate free input ``Y_V_bar`` and the group stays
    acyclic. Setting it ``True`` connects the real ``Y_V`` and adds a
    Gauss-Seidel iteration. The difference is not large here — the
    interference drag changes ``D_V`` by 42 lb, which reaches ``Y_V`` only
    through ``sin(psi_V)`` at 0.006 rad — but it is not zero, and on a
    configuration with more sidewash it would not be small.

    Options
    -------
    num_nodes : int
    linearized : bool
        Passed to the three components that have small-angle forms:
        ``MainRotorForcesComp``, ``HorizStabForcesComp``,
        ``VertStabForcesComp`` and ``FuselageForcesComp``. ``True``
        reproduces Tables 8.4 and 8.11 term for term.
    thrust : {'given', 'antitorque'}
        Where ``T_T`` comes from. ``'given'`` for the lateral-directional
        problem, where it is an unknown; ``'antitorque'`` for the
        longitudinal one, where p. 487 supplies it.
    blade_closest : {'up', 'down'}
    downwash_model : {'input', 'vortex_axis'}
        Applied to all three rotor downwash stations at once.
    tail_rotor_feedback : bool

    ``charts``: read the abaques or take Prouty's readings
    ------------------------------------------------------
    ``False``, the default, leaves every chart quantity a free input, which
    is how the anchors of this package were established: Prouty read his
    charts and printed the numbers, and reproducing his answers means using
    his readings.

    ``True`` replaces seven of them with the digitised charts, and the group
    then takes the geometry the charts are drawn against instead:

    ======================= =========================== ===================
    now computed             from                        replaces
    ======================= =========================== ===================
    ``A_R_H``                Figure 8.8, ``h_bH``        free input
    ``A_R_V``                Figure 8.19, fin geometry   free input
    ``a_H``, ``a_V``         Figure 8.6, sweep           free inputs
    ``delta_H``, ``delta_V`` Figure 8.17, taper          free inputs
    ``K_int``                Figure 8.22, ``y_V``        free input
    ``qH_q``                 Figures 8.9 and 8.10        free input
    ======================= =========================== ===================

    The two that stay free either way are ``eta_MV`` from Figure 8.21 p. 508,
    which is measured scatter rather than a curve, and the three ``v/v1``
    from Figure 8.11 pp. 494-495; see C8-3.

    Expect the answers to move. The digitised values differ from Prouty's
    readings by 2 to 4 % each -- ``A_R_V`` becomes 3.34 against his 3.2 and
    ``delta_V`` 0.021 against his 0.01 -- and the differences do not cancel.
    That is the cost of reading the charts rather than being told what they
    say, and it is the honest error bar on any *other* helicopter, where
    there is no printed reading to fall back on.

    ``V/v1_hover`` is not an input. It is
    ``2 sqrt(q / D.L.)``, in which the density cancels, so Figure 8.10 is
    driven by the flight condition the group already has. At 115 knots that
    is 4.97, beyond the 3.7 where the figure returns to zero, so wiring it
    changes nothing there -- it bites in transition, where it nearly doubles
    the local dynamic pressure.

    Defaults are the example helicopter at 115 knots, Tables 8.2, 8.3 and
    8.5, so that the group runs and reproduces the book out of the box.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)
        self.options.declare('thrust', values=('given', 'antitorque'),
                             default='given')
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
        dw = dict(num_nodes=nn, downwash_model=self.options['downwash_model'])

        # shared inputs reach several components with different declared
        # defaults; fix them once, at the example helicopter's values
        self.set_input_defaults('T_M', val=20606.0 * ones, units='lbf')
        self.set_input_defaults('q', val=45.0 * ones, units='lbf/ft**2')
        self.set_input_defaults('A_M', val=2827.0, units='ft**2')
        self.set_input_defaults('beta', val=np.zeros(nn), units='rad')
        self.set_input_defaults('Theta', val=np.zeros(nn), units='rad')
        self.set_input_defaults('gamma_c', val=np.zeros(nn), units='rad')
        if self.options['thrust'] == 'given':
            # T_T reaches the tail rotor, its sidewash and the interference
            self.set_input_defaults('T_T', val=661.0 * ones, units='lbf')
        else:
            # Q_M reaches the main rotor moments and the antitorque relation
            self.set_input_defaults('Q_M', val=34726.0 * ones, units='lbf*ft')

        if self.options['charts']:
            self._add_charts(nn, ones)

        # ------------------------------------------------- rotor downwash
        for station, ratio in (('H', 1.5), ('F', 1.0), ('V', 1.5)):
            names = [('eps_M', f'eps_M{station}')]
            ins = ['T_M', 'q', 'A_M']
            if self.options['downwash_model'] == 'input':
                ins.append(('v_ratio', f'v{station}_v1'))
                self.set_input_defaults(f'v{station}_v1', val=ratio * ones)
            else:
                ins += [('X_over_R', f'X_over_R_{station}'),
                        ('Z_over_R', f'Z_over_R_{station}')]
                names.append(('v_ratio', f'v{station}_v1'))
            self.add_subsystem(f'downwash_{station}', RotorDownwashComp(**dw),
                               promotes_inputs=ins, promotes_outputs=names)

        # ----------------------------------------------- fuselage geometry
        self.add_subsystem('fuselage_alpha', FuselageAlphaComp(num_nodes=nn),
                           promotes=['*'])

        # the same relation at both stabilisers, with its own two parameters
        for station in ('H', 'V'):
            self.add_subsystem(
                f'fuselage_downwash_{station}',
                FuselageDownwashAtHorizStabComp(num_nodes=nn),
                promotes_inputs=['alpha_F',
                                 ('eps_F0', f'eps_F0_{station}'),
                                 ('depsF_dalphaF', f'depsF_dalphaF_{station}')],
                promotes_outputs=[('eps_FH', f'eps_F{station}')])

        # ---------------------------------------------------- main rotor
        self.add_subsystem('main_rotor',
                           MainRotorForcesComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])

        # ---------------------------------------------------- tail rotor
        feedback = self.options['tail_rotor_feedback']
        fin_force = 'Y_V' if feedback else 'Y_V_bar'
        tr_inputs = ['H_T', 'Q_T', 'b1s_T']
        if self.options['thrust'] == 'antitorque':
            tr_inputs += ['Q_M', 'l_T', 'l_V', ('Y_V', fin_force)]
        else:
            tr_inputs.append('T_T')
        self.add_subsystem(
            'tail_rotor',
            TailRotorForcesComp(num_nodes=nn, thrust=self.options['thrust'],
                                blade_closest=self.options['blade_closest']),
            promotes_inputs=tr_inputs, promotes_outputs=['*'])

        self.add_subsystem('tail_rotor_sidewash',
                           TailRotorSidewashComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('fuselage_sidewash',
                           FuselageSidewashComp(num_nodes=nn), promotes=['*'])

        # ------------------------------------------ horizontal stabiliser
        self.add_subsystem('horiz_stab_alpha', HorizStabAlphaComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem(
            'horiz_stab_lift_drag', HorizStabLiftDragComp(num_nodes=nn),
            promotes_inputs=['alpha_H', 'q', 'qH_q', 'A_H', 'a_H',
                             'alpha_LO_H', ('A_R', 'A_R_H'),
                             ('delta', 'delta_H'), ('C_D0', 'C_D0_H')],
            promotes_outputs=['C_LH', 'L_H', 'D_H'])
        self.add_subsystem('horiz_stab_forces',
                           HorizStabForcesComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])

        # -------------------------------------------- vertical stabiliser
        self.add_subsystem(
            'vert_stab_interference',
            VertStabInterferenceDragComp(num_nodes=nn),
            promotes_inputs=['T_T', 'q', 'R_T', 'b_V', 'K_int',
                             ('Y_V', fin_force)],
            promotes_outputs=['dD_int'])
        self.add_subsystem(
            'vert_stab_lift_drag', VertStabLiftDragComp(num_nodes=nn),
            promotes_inputs=['beta', 'eta_MV', 'eta_TV', 'eta_FV', 'q',
                             'qV_q', 'dD_int', 'A_V', 'a_V', 'alpha_LO_V',
                             ('A_R', 'A_R_V'), ('delta', 'delta_V'),
                             ('C_D0', 'C_D0_V')],
            promotes_outputs=['psi_V', 'C_LV', 'L_V', 'D_V'])
        self.add_subsystem('vert_stab_forces',
                           VertStabForcesComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])

        # ------------------------------------------------------- fuselage
        self.add_subsystem('fuselage_derivatives',
                           FuselageDerivativesComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('fuselage_forces',
                           FuselageForcesComp(num_nodes=nn, linearized=lin),
                           promotes=['*'])

        if feedback:
            # Y_V -> dD_int -> D_V -> Y_V, p. 509. Gauss-Seidel rather than
            # Newton: one loop, weakly coupled, and the |T_T Y_V| kink of
            # p. 509 is not something to hand a Jacobian-based solver.
            self.nonlinear_solver = om.NonlinearBlockGS(maxiter=30, rtol=1e-10,
                                                        iprint=0)
            self.linear_solver = om.DirectSolver()

    def _add_charts(self, nn, ones):
        """Replace the free chart inputs with the digitised figures."""
        # ------------------------------------------------ aspect ratios
        self.add_subsystem(
            'end_plate', EndPlateFactorComp(),
            promotes_inputs=[('A_R_geo', 'A_R_H_geo'), 'h_bH'],
            promotes_outputs=[('A_R_eff', 'A_R_H'), ('factor', 'end_plate_factor')])
        self.add_subsystem(
            'fin_aspect_ratio', VertStabEffectiveARComp(),
            promotes_inputs=[('A_R_geo', 'A_R_V_geo'), 'bV_2r1', 'lambda_V',
                             'ZH_bV', 'x_cV', 'SH_SV'],
            promotes_outputs=[('A_R_eff', 'A_R_V'), 'f_B', 'f_H', 'K_H'])

        # ------------------------------------------- slopes and polars
        for tag in ('H', 'V'):
            self.add_subsystem(
                f'lift_slope_{tag}', LiftCurveSlopeComp(),
                promotes_inputs=[('A_R', f'A_R_{tag}'), ('sweep', f'sweep_{tag}')],
                promotes_outputs=[('a', f'a_{tag}')])
            self.add_subsystem(
                f'span_efficiency_{tag}', SpanEfficiencyComp(),
                promotes_inputs=[('lambda_taper', f'lambda_{tag}'),
                                 ('A_R', f'A_R_{tag}')],
                promotes_outputs=[('delta', f'delta_{tag}')])

        # ------------------------------------------------- interference
        self.add_subsystem('interference', InterferenceFactorComp(),
                           promotes_inputs=['y_V', 'R_T', 'b_V'],
                           promotes_outputs=['K_int'])

        # ------------------------------- local pressure at the stabiliser
        # V/v1_hover = 2 sqrt(q/D.L.); the density cancels
        self.add_subsystem(
            'stabiliser_flow',
            om.ExecComp(['DL = T_M / A_M',
                         'V_v1 = 2.0 * (q * A_M / T_M) ** 0.5'],
                        DL={'shape': (nn,), 'units': 'lbf/ft**2'},
                        V_v1={'shape': (nn,)},
                        T_M={'shape': (nn,), 'units': 'lbf', 'val': 20000.0},
                        q={'shape': (nn,), 'units': 'lbf/ft**2', 'val': 45.0},
                        A_M={'units': 'ft**2', 'val': 2827.0}),
            promotes=['*'])
        self.add_subsystem('dynamic_pressure',
                           DynPressureIncreaseComp(num_nodes=nn),
                           promotes_inputs=['V_v1', 'q', 'qH_q_momentum', 'DL'],
                           promotes_outputs=['qH_q', 'q_H', 'dq_DL'])

        # R_T and b_V now reach the chart as well as the drag increment
        self.set_input_defaults('R_T', val=6.5, units='ft')
        self.set_input_defaults('b_V', val=7.7, units='ft')
        self.set_input_defaults('lambda_V', val=0.21)
        self.set_input_defaults('qH_q_momentum', val=0.6 * ones)
