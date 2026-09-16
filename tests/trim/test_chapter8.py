"""Chapter 8 -- The Helicopter in Trim, p. 481-539.

Anchors are the example helicopter at 115 knots (mu = 0.3): Table 8.4
pp. 518-521, Table 8.5 p. 523, Table 8.11 pp. 536-537, and the hover
equations of p. 517 and p. 531.
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.trim import (ControlPositionsComp,
                         DynPressureIncreaseComp,
                         EndPlateFactorComp,
                         FuselageAlphaComp,
                         FuselageDerivativesComp,
                         FuselageDownwashAtHorizStabComp,
                         FuselageForcesComp,
                         FuselageSidewashComp,
                         HorizStabAlphaComp,
                         HorizStabForcesComp,
                         HorizStabLiftDragComp,
                         HoverLateralTrimComp,
                         HoverLongitudinalTrimComp,
                         LatCyclicPitchComp,
                         LatNEquilibriumComp,
                         LatREquilibriumComp,
                         LatYEquilibriumComp,
                         LateralDirectionalTrimGroup,
                         InterferenceFactorComp,
                         LiftCurveSlopeComp,
                         LongCyclicPitchComp,
                         LongMEquilibriumComp,
                         LongXEquilibriumComp,
                         LongZEquilibriumComp,
                         LongitudinalTrimGroup,
                         MainRotorForcesComp,
                         ManeuverWeightComp,
                         RotorDownwashComp,
                         TailRotorForcesComp,
                         TrimElementsGroup,
                         TrimGradientComp,
                         SpanEfficiencyComp,
                         TailRotorSidewashComp,
                         TailRotorTppAngleComp,
                         VertStabEffectiveARComp,
                         VertStabForcesComp,
                         VertStabInterferenceDragComp,
                         VertStabLiftDragComp)


# ------------------------------------------------------------------------
# the example helicopter at 115 knots
# ------------------------------------------------------------------------

# Table 8.5, p. 523: initial trim forces and physical dimensions
H_A1S0_BAR, T_M_BAR, Q_M_EX = -145.0, 20606.0, 34726.0
H_T_EX, Q_T_EX, T_T_EX, B1S_T_EX = 40.0, 127.0, 661.0, -0.0054
L_T_EX, L_V_EX, H_T_ARM_EX = 37.0, 35.0, 6.0
L_V_FORCE_EX = 287.0                      # vertical stabiliser lift

# Table 8.5 derived: rotor stiffness from Chapter 7, p. 477
DMM_DA1S_EX = 200940.0

# converged longitudinal solution, p. 522
T_M_EX, A1S_EX = 20586.0, -0.019
B1S_EX = -0.0155                          # lateral flapping, order of Fig 8.31


def _main_rotor(nn=1, linearized=False, **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', MainRotorForcesComp(num_nodes=nn, linearized=linearized),
        promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('T_M', np.full(nn, T_M_EX))
    p.set_val('a1s_M', np.full(nn, A1S_EX))
    p.set_val('b1s_M', np.full(nn, B1S_EX))
    p.set_val('H_a1s0', np.full(nn, H_A1S0_BAR))
    p.set_val('Q_M', np.full(nn, Q_M_EX))
    p.set_val('dMM_da1s', np.full(nn, DMM_DA1S_EX))
    p.set_val('dRM_db1s', np.full(nn, DMM_DA1S_EX))
    if linearized:
        p.set_val('T_M_bar', np.full(nn, T_M_BAR))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _tail_rotor(nn=1, thrust='given', blade_closest='up', **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', TailRotorForcesComp(num_nodes=nn, thrust=thrust,
                                    blade_closest=blade_closest),
        promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('H_T', np.full(nn, H_T_EX))
    p.set_val('Q_T', np.full(nn, Q_T_EX))
    p.set_val('b1s_T', np.full(nn, B1S_T_EX))
    if thrust == 'antitorque':
        p.set_val('Q_M', np.full(nn, Q_M_EX))
        p.set_val('Y_V', np.full(nn, L_V_FORCE_EX))
        p.set_val('l_T', L_T_EX)
        p.set_val('l_V', L_V_EX)
    else:
        p.set_val('T_T', np.full(nn, T_T_EX))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


# ------------------------------------------------------------------------
# main_rotor_forces_comp -- p. 485
# ------------------------------------------------------------------------

def test_main_rotor_linearized_reproduces_table_84():
    """X and Z rows of Table 8.4, p. 518-519, with i_M = 0."""
    p = _main_rotor(linearized=True)
    assert_near_equal(p.get_val('X_M')[0], 145.0 - 20606.0 * A1S_EX, 1e-12)
    assert_near_equal(p.get_val('Z_M')[0], -T_M_EX, 1e-12)


def test_main_rotor_linearized_reproduces_table_811():
    """Y, R and N rows of Table 8.11, p. 536-537."""
    p = _main_rotor(linearized=True)
    assert_near_equal(p.get_val('Y_M')[0], 20606.0 * B1S_EX, 1e-12)
    assert_near_equal(p.get_val('R_M')[0], 200940.0 * B1S_EX, 1e-12)
    assert_near_equal(p.get_val('M_M')[0], 200940.0 * A1S_EX, 1e-12)
    assert_near_equal(p.get_val('N_M')[0], Q_M_EX, 1e-12)


def test_main_rotor_sideslip_coefficient_of_table_811():
    """Table 8.11 prints +537 beta in the Y equation.

    The coefficient is -(H + T_bar a1s + T_bar i_M) built on the *converged
    longitudinal* flapping, which is what makes the two trim problems
    separable: a1s is frozen when the lateral set is solved.
    """
    p = _main_rotor(linearized=True, beta=1.0)
    coefficient = p.get_val('Y_M')[0] - 20606.0 * B1S_EX
    assert_near_equal(coefficient, 537.0, 2e-3)


def test_main_rotor_nonlinear_matches_small_angles():
    """p. 485 collapses onto Table 8.4 as the angles go to zero."""
    tiny = dict(a1s_M=1e-7, b1s_M=1e-7, beta=1e-7, T_M=T_M_BAR)
    exact = _main_rotor(**tiny)
    linear = _main_rotor(linearized=True, **tiny)
    for name in ('X_M', 'Y_M', 'Z_M'):
        assert_near_equal(exact.get_val(name)[0], linear.get_val(name)[0], 1e-6)


def test_main_rotor_torque_coupling_is_dropped_by_the_tables():
    """C8-1. Tables 8.4 and 8.11 drop Q_M sin(a1s) from R_M and
    -Q_M sin(b1s) from M_M, and at 115 knots they are not small.

    The roll term is 21 % of R_M and half the constant of the R equilibrium
    equation. Asserted as a disagreement so that a change which happens to
    close it is noticed rather than absorbed.
    """
    exact, linear = _main_rotor(), _main_rotor(linearized=True)

    dR = exact.get_val('R_M')[0] - linear.get_val('R_M')[0]
    dM = exact.get_val('M_M')[0] - linear.get_val('M_M')[0]
    assert_near_equal(dR, Q_M_EX * np.sin(A1S_EX), 1e-12)
    assert_near_equal(dM, -Q_M_EX * np.sin(B1S_EX), 1e-12)
    assert abs(dR / linear.get_val('R_M')[0]) > 0.20
    assert abs(dM / linear.get_val('M_M')[0]) > 0.13


def test_main_rotor_shaft_incidence_tilts_thrust_into_x():
    """X_M = -H cos(a1s + i_M) - T sin(a1s + i_M), p. 485."""
    i_M = np.deg2rad(3.0)
    p = _main_rotor(i_M=i_M)
    x = A1S_EX + i_M
    expected = -H_A1S0_BAR * np.cos(x) - T_M_EX * np.sin(x)
    assert_near_equal(p.get_val('X_M')[0], expected, 1e-12)


@pytest.mark.parametrize('linearized', [False, True])
def test_main_rotor_partials(linearized):
    p = _main_rotor(nn=3, linearized=linearized, beta=np.full(3, 0.05),
                    b1s_M=np.full(3, -0.027), i_M=0.03)
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# tail_rotor_forces_comp -- p. 487
# ------------------------------------------------------------------------

def test_tail_rotor_reproduces_table_84():
    """X_T, Z_T and M_T rows, p. 518-521, blade closest going up."""
    p = _tail_rotor()
    assert_near_equal(p.get_val('X_T')[0], -40.0, 1e-12)
    assert_near_equal(p.get_val('M_T')[0], -127.0, 1e-12)
    assert_near_equal(p.get_val('Z_T')[0], B1S_T_EX * T_T_EX, 1e-12)
    assert_near_equal(p.get_val('Z_T')[0], -4.0, 0.11)      # -3.57 printed -4
    assert_near_equal(-p.get_val('X_T')[0] * H_T_ARM_EX, 240.0, 1e-12)


def test_tail_rotor_ZT_lT_row_of_table_84_is_a_rounding_artifact():
    """C8-2. Table 8.4 prints Z_T l_T = -148, which is the rounded
    Z_T = -4 times l_T = 37, not the product of the unrounded values.
    """
    p = _tail_rotor()
    assert_near_equal(p.get_val('Z_T')[0] * L_T_EX, -132.1, 1e-3)
    assert_near_equal(-4.0 * L_T_EX, -148.0, 1e-12)


def test_tail_rotor_antitorque_relation():
    """T_T = Q_M/l_T - Y_V l_V/l_T, p. 487 and p. 510.

    Q_M/l_T reproduces the 939 lb of Table 8.5, so the 0.9 % gap on T_T
    sits in Prouty's Y_V, not in the relation.
    """
    p = _tail_rotor(thrust='antitorque')
    assert_near_equal(Q_M_EX / L_T_EX, 939.0, 1e-3)
    assert_near_equal(p.get_val('T_T')[0], 661.0, 1e-2)


def test_tail_rotor_antitorque_without_fin_is_the_hover_N_equation():
    """p. 531 gives Q_M - l_T T_T = 0 when the airframe is neglected."""
    p = _tail_rotor(thrust='antitorque', Y_V=0.0)
    assert_near_equal(Q_M_EX - L_T_EX * p.get_val('T_T')[0], 0.0, 1e-12)


def test_tail_rotor_blade_direction_flips_ZT_and_MT_together():
    """p. 487 gives both signs; they turn over as a pair."""
    up, down = _tail_rotor(), _tail_rotor(blade_closest='down')
    for name in ('Z_T', 'M_T'):
        assert_near_equal(down.get_val(name)[0], -up.get_val(name)[0], 1e-12)
    for name in ('X_T', 'Y_T'):
        assert_near_equal(down.get_val(name)[0], up.get_val(name)[0], 1e-12)


@pytest.mark.parametrize('thrust', ['given', 'antitorque'])
@pytest.mark.parametrize('blade_closest', ['up', 'down'])
def test_tail_rotor_partials(thrust, blade_closest):
    p = _tail_rotor(nn=3, thrust=thrust, blade_closest=blade_closest,
                    b1s_T=np.full(3, -0.02))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# rotor_downwash_horiz_stab_comp -- p. 493
# ------------------------------------------------------------------------

# Table 8.5 p. 523 flight conditions, Table 8.2 p. 501 stabiliser
Q_EX, A_M_EX = 45.0, 2827.0
VH_V1_EX = 1.5                              # Figure 8.11 as Prouty reads it
X_OVER_R_EX, Z_OVER_R_EX = -1.08, 0.3       # p. 493


def _downwash(nn=1, downwash_model='input', **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', RotorDownwashComp(num_nodes=nn,
                                  downwash_model=downwash_model),
        promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('T_M', np.full(nn, T_M_BAR))
    p.set_val('q', np.full(nn, Q_EX))
    p.set_val('A_M', A_M_EX)
    if downwash_model == 'vortex_axis':
        p.set_val('X_over_R', X_OVER_R_EX)
        p.set_val('Z_over_R', Z_OVER_R_EX)
    else:
        p.set_val('v_ratio', np.full(nn, VH_V1_EX))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_downwash_reproduces_page_493():
    """0.06 rad, 3.5 deg, at 115 knots and design gross weight."""
    p = _downwash()
    assert_near_equal(p.get_val('eps_M')[0], 0.06, 2e-2)
    assert_near_equal(p.get_val('eps_M', units='deg')[0], 3.5, 2e-2)


def test_downwash_uses_the_table_84_grouping():
    """Table 8.4 carries the angle as (v_H/v1) T_M/(4 q A_M)."""
    p = _downwash()
    expected = VH_V1_EX * T_M_BAR / (4.0 * Q_EX * A_M_EX)
    assert_near_equal(p.get_val('eps_M')[0], expected, 1e-12)


def test_downwash_two_printed_forms_agree():
    """p. 493 gives (v_H/v1)(v1/V) and (v_H/v1) D.L./(4q).

    They are the same statement: the high-speed momentum induced velocity of
    p. 167 makes v1/V = D.L./(4q) identically.
    """
    rho, V = 0.002377, np.sqrt(2.0 * Q_EX / 0.002377)
    v1 = T_M_BAR / (2.0 * rho * A_M_EX * V)
    p = _downwash()
    assert_near_equal(p.get_val('eps_M')[0], VH_V1_EX * v1 / V, 1e-12)


def test_vortex_axis_runs_high_against_figure_811():
    """C8-3. The level-one wake model gives 1.737 where Prouty reads 1.5.

    Asserted as a disagreement: the model keeps the rotor radius as the
    cylinder radius although the wake axis is nearly horizontal at mu = 0.3,
    so it is expected to be generous, and the size of that is worth watching.
    """
    p = _downwash(downwash_model='vortex_axis')
    assert_near_equal(p.get_val('v_ratio')[0], 1.737, 1e-3)
    assert 1.10 < p.get_val('v_ratio')[0] / VH_V1_EX < 1.25


def test_vortex_axis_is_unity_at_the_disc_and_two_far_downstream():
    """v/v1 = 1 + s/sqrt(s^2 + R^2), the on-axis vortex cylinder."""
    at_disc = _downwash(downwash_model='vortex_axis', X_over_R=0.0,
                        Z_over_R=0.0)
    assert_near_equal(at_disc.get_val('v_ratio')[0], 1.0, 1e-12)

    far = _downwash(downwash_model='vortex_axis', X_over_R=-1e5)
    assert_near_equal(far.get_val('v_ratio')[0], 2.0, 1e-4)


@pytest.mark.parametrize('downwash_model', ['input', 'vortex_axis'])
def test_downwash_partials(downwash_model):
    p = _downwash(nn=3, downwash_model=downwash_model,
                  T_M=np.linspace(18000.0, 22000.0, 3),
                  q=np.linspace(30.0, 60.0, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# fuselage_downwash_horiz_stab_comp -- p. 498
# ------------------------------------------------------------------------

EPS_F0_EX, DEPSF_DALPHAF_EX = 0.024, 0.23   # Table 8.5, p. 524
THETA_EX = -0.0165                          # p. 522


def _fuse_downwash(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', FuselageDownwashAtHorizStabComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_fuselage_downwash_defaults_are_table_85():
    """eps_F0 = 0.024 and (d eps_F/d alpha_F)_H = 0.23 out of the box."""
    p = _fuse_downwash()
    assert_near_equal(p.get_val('eps_F0')[0], EPS_F0_EX, 1e-12)
    assert_near_equal(p.get_val('depsF_dalphaF')[0], DEPSF_DALPHAF_EX, 1e-12)
    assert_near_equal(p.get_val('eps_FH')[0], EPS_F0_EX, 1e-12)


def test_fuselage_downwash_at_the_level_flight_point():
    """alpha_F = Theta - gamma_c - T_M/(4 q A_M) = -3.26 deg, p. 513.

    Table 8.8 p. 530 lists -3.3 deg for level flight at 115 knots.
    """
    alpha_F = THETA_EX - T_M_EX / (4.0 * Q_EX * A_M_EX)
    assert_near_equal(np.rad2deg(alpha_F), -3.3, 2e-2)

    p = _fuse_downwash(alpha_F=alpha_F)
    expected = EPS_F0_EX + DEPSF_DALPHAF_EX * alpha_F
    assert_near_equal(p.get_val('eps_FH')[0], expected, 1e-12)
    assert_near_equal(p.get_val('eps_FH', units='deg')[0], 0.625, 1e-2)


def test_fuselage_downwash_figure_815_wing_sizes():
    """no wing 0.06, small 0.23, medium 0.39, large 0.41, p. 500."""
    alpha_F = -0.0569
    for slope in (0.06, 0.23, 0.39, 0.41):
        p = _fuse_downwash(alpha_F=alpha_F, depsF_dalphaF=slope)
        assert_near_equal(p.get_val('eps_FH')[0],
                          EPS_F0_EX + slope * alpha_F, 1e-12)


def test_fuselage_downwash_partials():
    p = _fuse_downwash(nn=3, alpha_F=np.linspace(-0.20, 0.10, 3),
                       eps_F0=0.03, depsF_dalphaF=0.39)
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# horiz_stab_alpha_comp -- p. 489
# ------------------------------------------------------------------------

I_H_EX = -0.052                             # Table 8.5, p. 523
EPS_MH_EX = VH_V1_EX * T_M_BAR / (4.0 * Q_EX * A_M_EX)
EPS_FH_EX = EPS_F0_EX + DEPSF_DALPHAF_EX * (
    THETA_EX - T_M_EX / (4.0 * Q_EX * A_M_EX))


def _alpha_H(nn=1, reference='attitude', **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', HorizStabAlphaComp(num_nodes=nn, reference=reference),
        promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('i_H', I_H_EX)
    p.set_val('eps_MH', np.full(nn, EPS_MH_EX))
    p.set_val('eps_FH', np.full(nn, EPS_FH_EX))
    if reference == 'attitude':
        p.set_val('Theta', np.full(nn, THETA_EX))
    else:
        p.set_val('alpha_TPP', np.full(nn, THETA_EX + A1S_EX))
        p.set_val('a1s_M', np.full(nn, A1S_EX))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_alpha_H_at_the_level_flight_point():
    """Table 8.8 p. 530 lists -7.9 deg at 115 knots, level."""
    p = _alpha_H()
    assert_near_equal(p.get_val('alpha_H')[0], -0.1401, 1e-3)
    assert_near_equal(p.get_val('alpha_H', units='deg')[0], -7.9, 2e-2)


def test_alpha_H_two_forms_agree():
    """p. 489 prints both; they are one relation via alpha_TPP, p. 525."""
    attitude, tpp = _alpha_H(), _alpha_H(reference='tpp')
    assert_near_equal(tpp.get_val('alpha_H')[0],
                      attitude.get_val('alpha_H')[0], 1e-12)


def test_alpha_H_two_forms_agree_in_a_climb():
    """alpha_TPP = Theta + a1s_M + i_M - gamma_c carries the climb angle.

    The tpp form takes no gamma_c on purpose. Double-counting it is invisible
    in level flight and shows up only here.
    """
    gamma_c, i_M = np.deg2rad(9.7), np.deg2rad(2.0)   # Table 8.8 climb case
    attitude = _alpha_H(gamma_c=gamma_c)
    tpp = _alpha_H(reference='tpp', i_M=i_M,
                   alpha_TPP=THETA_EX + A1S_EX + i_M - gamma_c)
    assert_near_equal(tpp.get_val('alpha_H')[0],
                      attitude.get_val('alpha_H')[0], 1e-12)


def test_alpha_H_shaft_incidence_cancels_in_the_attitude_form():
    """The attitude form measures from the body, so i_M does not appear."""
    p = _alpha_H()
    assert 'i_M' not in p.model.comp._var_rel_names['input']


@pytest.mark.parametrize('reference', ['attitude', 'tpp'])
def test_alpha_H_partials(reference):
    p = _alpha_H(nn=3, reference=reference,
                 eps_MH=np.linspace(0.04, 0.08, 3),
                 eps_FH=np.linspace(-0.01, 0.03, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# lift_curve_slope_comp -- Figure 8.6, p. 490
# ------------------------------------------------------------------------

def _slope(**ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', LiftCurveSlopeComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_lift_curve_slope_horizontal_stabiliser():
    """Table 8.2 p. 501 reads 4.0 /rad at A.R. = 4.5, Lambda = 13 deg."""
    p = _slope(A_R=4.5, sweep=np.deg2rad(13.0))
    assert_near_equal(p.get_val('a')[0], 4.020, 1e-3)
    assert_near_equal(p.get_val('a')[0], 4.0, 1e-2)


def test_lift_curve_slope_vertical_stabiliser_runs_high():
    """Open anchor. Table 8.3 p. 511 reads 3.0 /rad at A.R.eff = 3.2,
    Lambda = 27 deg; the relation Figure 8.6 plots gives 3.290.

    The formula is not the suspect: it lands on the horizontal stabiliser
    exactly, and both surfaces come off the same chart.
    """
    p = _slope(A_R=3.2, sweep=np.deg2rad(27.0))
    assert_near_equal(p.get_val('a')[0], 3.290, 1e-3)
    assert 1.05 < p.get_val('a')[0] / 3.0 < 1.15


def test_lift_curve_slope_is_the_asymptote_of_the_printed_chart():
    """Figure 8.6's ordinate tops out at 1.6; a/A.R. -> pi/2 as A.R. -> 0."""
    p = _slope(A_R=1e-6)
    assert_near_equal(p.get_val('a')[0] / 1e-6, np.pi / 2.0, 1e-6)


def test_lift_curve_slope_partials():
    p = _slope(A_R=3.2, sweep=np.deg2rad(27.0))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# horiz_stab_lift_drag_comp -- p. 488
# ------------------------------------------------------------------------

# Table 8.2 p. 501 and Table 8.5 p. 524
QH_Q_EX, A_H_EX, A_H_AR_EX = 0.6, 18.0, 4.5
A_SLOPE_EX, DELTA_H_EX, CD0_H_EX = 4.0, 0.02, 0.0064
ALPHA_H_EX = THETA_EX + I_H_EX - (EPS_MH_EX + EPS_FH_EX)


def _lift_drag(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', HorizStabLiftDragComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('alpha_H', np.full(nn, ALPHA_H_EX))
    p.set_val('q', np.full(nn, Q_EX))
    p.set_val('qH_q', np.full(nn, QH_Q_EX))
    p.set_val('A_H', A_H_EX)
    p.set_val('a_H', A_SLOPE_EX)
    p.set_val('A_R', A_H_AR_EX)
    p.set_val('delta', DELTA_H_EX)
    p.set_val('C_D0', CD0_H_EX)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_horiz_stab_lift_matches_table_85():
    """L_H = -273 lb at 115 knots."""
    p = _lift_drag()
    assert_near_equal(p.get_val('L_H')[0], -273.0, 3e-3)
    assert_near_equal(p.get_val('C_LH')[0], A_SLOPE_EX * ALPHA_H_EX, 1e-12)


def test_horiz_stab_lift_confirms_the_alpha_H_chain():
    """-273 lb over (q_H/q) q A_H = 486 is alpha_H = -8.05 deg.

    That is what this package computes, -8.03 deg, and not the -7.9 deg of
    Table 8.8 p. 530. Prouty's own lift settles the disagreement in favour
    of the chain.
    """
    implied = -273.0 / (QH_Q_EX * Q_EX * A_H_EX) / A_SLOPE_EX
    assert_near_equal(np.rad2deg(implied), -8.05, 3e-3)
    assert_near_equal(np.rad2deg(ALPHA_H_EX), -8.03, 3e-3)


def test_horiz_stab_drag_runs_below_table_85():
    """Open anchor: 14.1 lb against the 15 printed, 6 %."""
    p = _lift_drag()
    assert_near_equal(p.get_val('D_H')[0], 14.13, 1e-2)
    assert 0.90 < p.get_val('D_H')[0] / 15.0 < 0.97


def test_horiz_stab_drag_splits_into_induced_and_profile():
    """C_D = C_L^2 (1 + delta)/(pi A.R.) + C_D0, p. 488."""
    profile = _lift_drag(alpha_H=np.zeros(1))
    qA = QH_Q_EX * Q_EX * A_H_EX
    assert_near_equal(profile.get_val('D_H')[0], qA * CD0_H_EX, 1e-12)

    full = _lift_drag()
    induced = full.get_val('D_H')[0] - profile.get_val('D_H')[0]
    C_L = full.get_val('C_LH')[0]
    assert_near_equal(induced,
                      qA * C_L ** 2 * (1 + DELTA_H_EX)
                      / (np.pi * A_H_AR_EX), 1e-12)


def test_horiz_stab_lift_drag_partials():
    p = _lift_drag(nn=3, alpha_H=np.linspace(-0.20, 0.10, 3),
                   q=np.linspace(30.0, 60.0, 3),
                   qH_q=np.linspace(0.5, 0.8, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# horiz_stab_forces_comp -- p. 488
# ------------------------------------------------------------------------

L_H_EX, D_H_EX = -273.0, 15.0               # Table 8.5


def _hs_forces(nn=1, linearized=False, **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', HorizStabForcesComp(num_nodes=nn, linearized=linearized),
        promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('L_H', np.full(nn, L_H_EX))
    p.set_val('D_H', np.full(nn, D_H_EX))
    p.set_val('Theta', np.full(nn, THETA_EX))
    p.set_val('eps_MH', np.full(nn, EPS_MH_EX))
    p.set_val('eps_FH', np.full(nn, EPS_FH_EX))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_horiz_stab_forces_at_the_level_flight_point():
    """phi = Theta - (eps_MH + eps_FH + gamma_c) = -0.0881 rad, p. 488."""
    phi = THETA_EX - (EPS_MH_EX + EPS_FH_EX)
    assert_near_equal(phi, -0.0881, 2e-3)

    p = _hs_forces()
    assert_near_equal(p.get_val('X_H')[0],
                      L_H_EX * np.sin(phi) - D_H_EX * np.cos(phi), 1e-12)
    assert_near_equal(p.get_val('Z_H')[0],
                      -L_H_EX * np.cos(phi) - D_H_EX * np.sin(phi), 1e-12)


def test_horiz_stab_forces_phi_is_alpha_H_less_the_incidence():
    """The forces resolve onto the airframe, so i_H never enters."""
    phi = THETA_EX - (EPS_MH_EX + EPS_FH_EX)
    assert_near_equal(phi, ALPHA_H_EX - I_H_EX, 1e-12)


def test_horiz_stab_forces_linearized_matches_at_small_phi():
    """Table 8.4 puts sin(phi) -> phi and cos(phi) -> 1."""
    tiny = dict(Theta=np.full(1, 1e-7), eps_MH=np.zeros(1),
                eps_FH=np.zeros(1))
    exact, linear = _hs_forces(**tiny), _hs_forces(linearized=True, **tiny)
    for name in ('X_H', 'Z_H'):
        assert_near_equal(exact.get_val(name)[0],
                          linear.get_val(name)[0], 1e-6)


@pytest.mark.parametrize('linearized', [False, True])
def test_horiz_stab_forces_partials(linearized):
    p = _hs_forces(nn=3, linearized=linearized,
                   Theta=np.linspace(-0.05, 0.05, 3),
                   gamma_c=np.linspace(-0.10, 0.17, 3),
                   eps_MH=np.linspace(0.04, 0.08, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# vertical stabiliser -- pp. 502-511
# ------------------------------------------------------------------------

# Table 8.3 p. 511 and Table 8.5 p. 524
QV_Q_EX, A_V_EX, A_V_SLOPE_EX, A_V_AR_EX = 0.6, 33.0, 3.0, 3.2
DELTA_V_EX, CD0_V_EX, ALPHA_LO_V_EX = 0.01, 0.0064, -0.1012
ETA_MV_EX, R_T_EX, B_V_EX, K_INT_EX = -0.052, 6.5, 7.7, 0.4
A_T_EX = np.pi * R_T_EX ** 2
LIFT_V_EX, D_V_EX = 287.0, 58.0


def _one(cls, nn=1, opts=None, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', cls(num_nodes=nn, **(opts or {})),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_tail_rotor_sidewash_matches_table_85():
    """eta_TV = T_T / [4 (q_V/q) q A_T], p. 508. Table 8.5 lists .045."""
    p = _one(TailRotorSidewashComp, T_T=np.full(1, T_T_EX),
             q=np.full(1, Q_EX), qV_q=np.full(1, QV_Q_EX), A_T=A_T_EX)
    assert_near_equal(p.get_val('eta_TV')[0], 0.04611, 1e-3)
    assert_near_equal(p.get_val('eta_TV')[0], 0.045, 3e-2)


def test_tail_rotor_sidewash_uses_local_dynamic_pressure():
    """The (q_V/q) is in the denominator, not only in the fin lift."""
    slowed = _one(TailRotorSidewashComp, T_T=np.full(1, T_T_EX),
                  q=np.full(1, Q_EX), qV_q=np.full(1, QV_Q_EX), A_T=A_T_EX)
    free = _one(TailRotorSidewashComp, T_T=np.full(1, T_T_EX),
                q=np.full(1, Q_EX), qV_q=np.ones(1), A_T=A_T_EX)
    assert_near_equal(slowed.get_val('eta_TV')[0],
                      free.get_val('eta_TV')[0] / QV_Q_EX, 1e-12)


def test_fuselage_sidewash_slope_is_the_006_of_page_509():
    """C8-5. Table 8.11's beta coefficient in the Y_V lift row is -2833.

    The row is -q(q_V/q)A_V a_V (1 + d eta_F/d beta) = -2673 (1 + x), which
    lands on -2833.4 at x = 0.06 and on -3288 at x = 0.23. The tables were
    computed with p. 509's 0.06, not with Table 8.5's -.23 entry, which
    belongs to the longitudinal downwash instead.
    """
    p = _one(FuselageSidewashComp, beta=np.ones(1))
    assert_near_equal(p.get_val('detaF_dbeta')[0], 0.06, 1e-12)

    row = -Q_EX * QV_Q_EX * A_V_EX * A_V_SLOPE_EX
    assert_near_equal(row * (1.0 + 0.06), -2833.0, 2e-4)
    assert abs(row * (1.0 + 0.23) - (-2833.0)) > 400.0


def _fin(nn=1, **ivc):
    vals = dict(eta_MV=np.full(nn, ETA_MV_EX),
                eta_TV=np.full(nn, T_T_EX / (4.0 * QV_Q_EX * Q_EX * A_T_EX)),
                q=np.full(nn, Q_EX), qV_q=np.full(nn, QV_Q_EX),
                A_V=A_V_EX, a_V=A_V_SLOPE_EX, alpha_LO_V=ALPHA_LO_V_EX,
                A_R=A_V_AR_EX, delta=DELTA_V_EX, C_D0=CD0_V_EX)
    return _one(VertStabLiftDragComp, nn=nn, **{**vals, **ivc})


def test_vert_stab_lift_matches_table_85():
    """L_V = 287 lb at 115 knots, beta = 0."""
    p = _fin()
    assert_near_equal(p.get_val('L_V')[0], 287.0, 4e-3)
    assert_near_equal(p.get_val('psi_V')[0], -0.00589, 1e-2)


def test_vert_stab_drag_is_mostly_interference():
    """Table 8.5 gives D_V = 58; the fin's own polar produces 14.9."""
    clean = _fin()
    dD = _one(VertStabInterferenceDragComp,
              T_T=np.full(1, T_T_EX), Y_V=np.full(1, LIFT_V_EX),
              q=np.full(1, Q_EX), R_T=R_T_EX, b_V=B_V_EX, K_int=K_INT_EX)
    assert_near_equal(clean.get_val('D_V')[0], 14.94, 1e-2)
    assert_near_equal(dD.get_val('dD_int')[0], 42.79, 1e-2)

    total = _fin(dD_int=dD.get_val('dD_int'))
    assert_near_equal(total.get_val('D_V')[0], 58.0, 1e-2)
    assert dD.get_val('dD_int')[0] / total.get_val('D_V')[0] > 0.70


def test_vert_stab_table_811_T_T_coefficient():
    """-q(q_V/q)A_V a_V / [4 (q_V/q) q A_T] is printed as -.187."""
    coefficient = -(Q_EX * QV_Q_EX * A_V_EX * A_V_SLOPE_EX
                    / (4.0 * QV_Q_EX * Q_EX * A_T_EX))
    assert_near_equal(coefficient, -0.187, 3e-3)


def test_interference_drag_kink_at_zero_side_force():
    """|T_T Y_V| is not differentiable where either force crosses zero."""
    kw = dict(T_T=np.full(1, T_T_EX), q=np.full(1, Q_EX), R_T=R_T_EX,
              b_V=B_V_EX, K_int=K_INT_EX)
    plus = _one(VertStabInterferenceDragComp, Y_V=np.full(1, 50.0), **kw)
    minus = _one(VertStabInterferenceDragComp, Y_V=np.full(1, -50.0), **kw)
    assert_near_equal(plus.get_val('dD_int')[0],
                      minus.get_val('dD_int')[0], 1e-12)
    assert plus.get_val('dD_int')[0] > 0.0


def _fin_forces(nn=1, linearized=False, **ivc):
    eps_FV = EPS_F0_EX + 0.23 * (THETA_EX - T_M_EX / (4.0 * Q_EX * A_M_EX))
    vals = dict(L_V=np.full(nn, LIFT_V_EX), D_V=np.full(nn, D_V_EX),
                psi_V=np.full(nn, ETA_MV_EX
                              + T_T_EX / (4.0 * QV_Q_EX * Q_EX * A_T_EX)),
                Theta=np.full(nn, THETA_EX),
                eps_MV=np.full(nn, 1.5 * T_M_EX / (4.0 * Q_EX * A_M_EX)),
                eps_FV=np.full(nn, eps_FV))
    return _one(VertStabForcesComp, nn=nn, opts={'linearized': linearized},
                **{**vals, **ivc})


def test_vert_stab_forces_match_table_84_X_rows():
    """-D_V - L_V(eta_MV + eta_TV) sums to -56 in Table 8.4 p. 518."""
    p = _fin_forces()
    assert_near_equal(p.get_val('X_V')[0], -56.0, 2e-2)
    assert_near_equal(p.get_val('Y_V')[0], LIFT_V_EX, 1e-2)


def test_vert_stab_Z_is_the_tilt_of_its_own_X_force():
    """Z_V = X_V sin(phi_V), p. 502: the fin has no lift in the Z plane."""
    p = _fin_forces()
    phi = (THETA_EX - 1.5 * T_M_EX / (4.0 * Q_EX * A_M_EX)
           - (EPS_F0_EX + 0.23 * (THETA_EX - T_M_EX / (4.0 * Q_EX * A_M_EX))))
    assert_near_equal(p.get_val('Z_V')[0],
                      p.get_val('X_V')[0] * np.sin(phi), 1e-12)
    assert_near_equal(p.get_val('Z_V')[0], 4.93, 1e-2)


def test_vert_stab_forces_linearized_matches_at_small_angles():
    tiny = dict(psi_V=np.full(1, 1e-7), Theta=np.full(1, 1e-7),
                eps_MV=np.zeros(1), eps_FV=np.zeros(1))
    exact, linear = _fin_forces(**tiny), _fin_forces(linearized=True, **tiny)
    for name in ('X_V', 'Y_V', 'Z_V'):
        assert_near_equal(exact.get_val(name)[0],
                          linear.get_val(name)[0], 1e-6)


def test_vert_stab_partials():
    assert_check_partials(
        _fin(nn=3, beta=np.linspace(-0.1, 0.1, 3),
             eta_FV=np.linspace(-0.006, 0.006, 3),
             q=np.linspace(30.0, 60.0, 3)).check_partials(
                 method='cs', out_stream=None), atol=1e-9, rtol=1e-9)

    assert_check_partials(
        _one(TailRotorSidewashComp, nn=3, T_T=np.linspace(400.0, 900.0, 3),
             q=np.linspace(30.0, 60.0, 3), qV_q=np.linspace(0.5, 0.8, 3),
             A_T=A_T_EX).check_partials(method='cs', out_stream=None),
        atol=1e-9, rtol=1e-9)

    assert_check_partials(
        _one(FuselageSidewashComp, nn=3, beta=np.linspace(-0.1, 0.1, 3),
             detaF_dbeta=0.06).check_partials(method='cs', out_stream=None),
        atol=1e-9, rtol=1e-9)

    assert_check_partials(
        _one(VertStabInterferenceDragComp, nn=3,
             T_T=np.linspace(400.0, 900.0, 3),
             Y_V=np.linspace(-300.0, 300.0, 3), q=np.linspace(30.0, 60.0, 3),
             R_T=R_T_EX, b_V=B_V_EX,
             K_int=K_INT_EX).check_partials(method='cs', out_stream=None),
        atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('linearized', [False, True])
def test_vert_stab_forces_partials(linearized):
    p = _fin_forces(nn=3, linearized=linearized,
                    psi_V=np.linspace(-0.1, 0.1, 3),
                    Theta=np.linspace(-0.05, 0.05, 3),
                    gamma_c=np.linspace(-0.10, 0.17, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# fuselage -- pp. 512-515
# ------------------------------------------------------------------------

# Table 8.5 pp. 524-525, all credited to Appendix A
LF_Q_0_EX, DLF_Q_EX = -1.5, 75.0
MF_Q_0_EX, DMF_Q_EX = -160.0, 1780.0
DSF_Q_EX, DRF_Q_EX, DNF_Q_EX = -220.0, 230.0, -820.0
EPS_MF_EX = T_M_EX / (4.0 * Q_EX * A_M_EX)      # v_F/v1 = 1, Table 8.4
ALPHA_F_EX = THETA_EX - EPS_MF_EX
L_F_EX, D_F_EX = -281.0, 794.0                  # Table 8.5 resultants


def test_fuselage_alpha_at_the_level_flight_point():
    """alpha_F = Theta - gamma_c - eps_MF = -3.26 deg, p. 513."""
    p = _one(FuselageAlphaComp, Theta=np.full(1, THETA_EX),
             eps_MF=np.full(1, EPS_MF_EX))
    assert_near_equal(p.get_val('alpha_F')[0], ALPHA_F_EX, 1e-12)
    assert_near_equal(p.get_val('alpha_F', units='deg')[0], -3.3, 2e-2)


def test_fuselage_alpha_agrees_with_chapter_3():
    """Chapter 3 p. 192 gives alpha_F = alpha_TPP - i_M - a1s - v1/V.

    Substituting alpha_TPP = Theta + a1s + i_M - gamma_c from p. 525 leaves
    p. 513's form, with eps_MF = v1/V. The two chapters are one relation.
    """
    i_M, gamma_c = np.deg2rad(2.0), np.deg2rad(5.0)
    alpha_TPP = THETA_EX + A1S_EX + i_M - gamma_c
    chapter_3 = alpha_TPP - i_M - A1S_EX - EPS_MF_EX

    p = _one(FuselageAlphaComp, Theta=np.full(1, THETA_EX),
             gamma_c=np.full(1, gamma_c), eps_MF=np.full(1, EPS_MF_EX))
    assert_near_equal(p.get_val('alpha_F')[0], chapter_3, 1e-12)


def _fuse(nn=1, **ivc):
    vals = dict(alpha_F=np.full(nn, ALPHA_F_EX), q=np.full(nn, Q_EX))
    return _one(FuselageDerivativesComp, nn=nn, **{**vals, **ivc})


def test_fuselage_derivative_defaults_are_table_85():
    p = _fuse()
    for name, value in (('LF_q_0', LF_Q_0_EX), ('dLF_q_dalpha', DLF_Q_EX),
                        ('MF_q_0', MF_Q_0_EX), ('dMF_q_dalpha', DMF_Q_EX),
                        ('dSF_q_dbeta', DSF_Q_EX), ('dRF_q_dbeta', DRF_Q_EX),
                        ('dNF_q_dbeta', DNF_Q_EX)):
        assert_near_equal(p.get_val(name)[0], value, 1e-12)


def test_fuselage_pitching_moment_matches_table_85():
    """M_F = -11,722 ft-lb at 115 knots."""
    p = _fuse()
    assert_near_equal(p.get_val('M_F')[0], -11722.0, 4e-3)


def test_fuselage_lift_runs_below_table_85():
    """Open anchor: -259.5 lb against the -281 printed, 7.6 %.

    Table 8.5's own resultant implies alpha_F = -3.62 deg where the same
    table's inputs give -3.26 deg, so the gap is in the book's iteration.
    """
    p = _fuse()
    assert_near_equal(p.get_val('L_F')[0], -259.5, 1e-3)
    implied = (-281.0 / Q_EX - LF_Q_0_EX) / DLF_Q_EX
    assert_near_equal(np.rad2deg(implied), -3.62, 5e-3)


def test_fuselage_lateral_slopes_match_table_811():
    """q times the three slopes are printed as -9900, -36,900 and 10,350."""
    p = _fuse(beta=np.ones(1))
    assert_near_equal(p.get_val('SF_F')[0], -9900.0, 1e-12)
    assert_near_equal(p.get_val('N_F')[0], -36900.0, 1e-12)
    assert_near_equal(p.get_val('R_F')[0], 10350.0, 1e-12)


def test_fuselage_lateral_terms_vanish_at_zero_sideslip():
    """p. 513 gives a symmetric airframe no side force at beta = 0."""
    p = _fuse()
    for name in ('SF_F', 'N_F', 'R_F'):
        assert_near_equal(p.get_val(name)[0], 0.0, 1e-12)


def test_chapter_3_figure_a2_is_empennage_on():
    """C8-7. prouty.forward_flight.FuselageAeroComp cannot feed this one.

    Its calibrated lift slope is 1.953 ft^2/deg for the airframe *with*
    empennage. The horizontal stabiliser contributes (q_H/q) A_H a_H, and
    the remainder lands within 9 % of Table 8.5's bare fuselage slope.
    """
    per_deg = np.pi / 180.0
    stabiliser = QH_Q_EX * A_H_EX * A_SLOPE_EX * per_deg
    assert_near_equal(stabiliser, 0.754, 1e-2)
    assert_near_equal(1.953 - stabiliser, 1.199, 1e-2)
    assert abs(1.199 / (DLF_Q_EX * per_deg) - 1.0) < 0.10


def _fuse_forces(nn=1, linearized=False, **ivc):
    vals = dict(L_F=np.full(nn, L_F_EX), D_F=np.full(nn, D_F_EX),
                alpha_F=np.full(nn, ALPHA_F_EX))
    return _one(FuselageForcesComp, nn=nn, opts={'linearized': linearized},
                **{**vals, **ivc})


def test_fuselage_forces_at_the_level_flight_point():
    """X_F = -D_F cos(alpha_F) + L_F sin(alpha_F), p. 512."""
    p = _fuse_forces()
    assert_near_equal(p.get_val('X_F')[0],
                      -D_F_EX * np.cos(ALPHA_F_EX)
                      + L_F_EX * np.sin(ALPHA_F_EX), 1e-12)
    assert_near_equal(p.get_val('Z_F')[0],
                      -L_F_EX * np.cos(ALPHA_F_EX)
                      - D_F_EX * np.sin(ALPHA_F_EX), 1e-12)


def test_fuselage_forces_resolve_through_alpha_F_itself():
    """p. 512's bracket, Theta - gamma_c - eps_MF, is p. 513's alpha_F.

    No incidence has to be taken back out, unlike the stabiliser, so no
    attitude is an input here at all.
    """
    p = _fuse_forces()
    assert 'Theta' not in p.model.comp._var_rel_names['input']
    assert 'eps_MF' not in p.model.comp._var_rel_names['input']


def test_fuselage_side_force_carries_the_drag_tilt():
    """Y_F = SF_F cos(beta) - D_F sin(beta), p. 512.

    Table 8.11 prints the row as -9900 beta, folding the second term into
    the coefficient. At D_F = 794 it is 8 % of it, not nothing.
    """
    beta = 1.0
    p = _fuse_forces(SF_F=np.full(1, DSF_Q_EX * Q_EX * beta),
                     beta=np.full(1, beta), linearized=True)
    assert_near_equal(p.get_val('Y_F')[0], -9900.0 - D_F_EX, 1e-12)


def test_fuselage_forces_linearized_matches_at_small_angles():
    tiny = dict(alpha_F=np.full(1, 1e-7), beta=np.full(1, 1e-7),
                SF_F=np.full(1, 10.0))
    exact, linear = _fuse_forces(**tiny), _fuse_forces(linearized=True, **tiny)
    for name in ('X_F', 'Y_F', 'Z_F'):
        assert_near_equal(exact.get_val(name)[0],
                          linear.get_val(name)[0], 1e-6)


def test_fuselage_partials():
    assert_check_partials(
        _one(FuselageAlphaComp, nn=3, Theta=np.linspace(-0.05, 0.05, 3),
             gamma_c=np.linspace(-0.1, 0.17, 3),
             eps_MF=np.linspace(0.02, 0.06, 3)).check_partials(
                 method='cs', out_stream=None), atol=1e-9, rtol=1e-9)

    assert_check_partials(
        _fuse(nn=3, alpha_F=np.linspace(-0.2, 0.1, 3),
              beta=np.linspace(-0.1, 0.1, 3), q=np.linspace(30.0, 60.0, 3),
              f=np.linspace(19.0, 23.0, 3)).check_partials(
                  method='cs', out_stream=None), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('linearized', [False, True])
def test_fuselage_forces_partials(linearized):
    p = _fuse_forces(nn=3, linearized=linearized,
                     alpha_F=np.linspace(-0.2, 0.1, 3),
                     beta=np.linspace(-0.1, 0.1, 3),
                     SF_F=np.linspace(-500.0, 500.0, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# trim_elements_group -- pp. 485-515 assembled
# ------------------------------------------------------------------------

# Table 8.5 pp. 523-525, moment arms about the c.g.
ARMS = dict(b_M=7.5, l_M=-0.5, b_T=6.0, l_T=37.0, b_H=-1.5, l_H=33.0,
            b_V=3.0, l_V=35.0, b_F=0.5, l_F=-0.5)
GW_EX = 20000.0


def _elements(nn=1, **opts):
    p = om.Problem()
    p.model.add_subsystem('g', TrimElementsGroup(num_nodes=nn, **opts),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    one = np.ones(nn)
    for name, value in (
            ('T_M', T_M_EX), ('q', Q_EX), ('Theta', THETA_EX),
            ('a1s_M', A1S_EX), ('b1s_M', B1S_EX), ('H_a1s0', H_A1S0_BAR),
            ('Q_M', Q_M_EX), ('dMM_da1s', DMM_DA1S_EX),
            ('dRM_db1s', DMM_DA1S_EX), ('H_T', H_T_EX), ('Q_T', Q_T_EX),
            ('b1s_T', B1S_T_EX), ('qH_q', QH_Q_EX), ('qV_q', QV_Q_EX),
            ('eta_MV', ETA_MV_EX), ('f', D_F_EX / Q_EX)):
        p.set_val(name, value * one)
    for name, value in (
            ('A_M', A_M_EX), ('A_T', A_T_EX), ('i_H', I_H_EX),
            ('A_H', A_H_EX), ('a_H', A_SLOPE_EX), ('A_R_H', A_H_AR_EX),
            ('delta_H', DELTA_H_EX), ('C_D0_H', CD0_H_EX), ('A_V', A_V_EX),
            ('a_V', A_V_SLOPE_EX), ('A_R_V', A_V_AR_EX),
            ('delta_V', DELTA_V_EX), ('C_D0_V', CD0_V_EX),
            ('alpha_LO_V', ALPHA_LO_V_EX), ('R_T', R_T_EX), ('b_V', B_V_EX),
            ('K_int', K_INT_EX)):
        p.set_val(name, value)
    if opts.get('thrust', 'given') == 'given':
        p.set_val('T_T', T_T_EX * one)
    else:
        p.set_val('l_T', ARMS['l_T'])
        p.set_val('l_V', ARMS['l_V'])
    if not opts.get('tail_rotor_feedback', False):
        p.set_val('Y_V_bar', LIFT_V_EX * one)
    if opts.get('charts', False):
        return p                      # caller supplies the chart geometry
    p.run_model()
    return p


def test_elements_group_reproduces_table_85():
    """The whole of pp. 485-515 at 115 knots, in one pass."""
    p = _elements()
    for name, book, tol in (('L_H', -273.0, 5e-3), ('D_H', 15.0, 7e-2),
                            ('L_V', 287.0, 5e-3), ('D_V', 58.0, 5e-3),
                            ('M_F', -11722.0, 5e-3), ('eta_TV', 0.045, 3e-2),
                            ('X_T', -40.0, 1e-12), ('M_T', -127.0, 1e-12),
                            ('X_V', -56.0, 5e-3)):
        assert_near_equal(p.get_val(name)[0], book, tol)

    assert_near_equal(p.get_val('alpha_F', units='deg')[0], -3.3, 2e-2)
    assert_near_equal(p.get_val('alpha_H', units='deg')[0], -7.9, 2e-2)


def test_elements_group_closes_the_X_and_Z_equations():
    """Fed Prouty's converged unknowns, the residuals should be small.

    This is the test the individual components cannot do: every sign,
    every moment arm and every downwash path has to be right at once for
    the sums to cancel.
    """
    p = _elements()
    X = sum(p.get_val(f'X_{c}')[0] for c in 'MTHVF')
    Z = sum(p.get_val(f'Z_{c}')[0] for c in 'MTHVF')

    assert abs(X - GW_EX * np.sin(THETA_EX)) < 5.0        # against ~800 lb
    assert abs(Z + GW_EX) < 30.0                          # against 20,000 lb


def test_elements_group_M_residual_carries_the_C8_1_term():
    """The M equation closes to 612 ft-lb, of which 538 is C8-1.

    Prouty's own linearised M equation, evaluated at his converged unknowns,
    leaves 129 ft-lb. Removing the Q_M sin(b1s_M) term that Table 8.4 drops
    brings this group to 74, which is the same order. The residual is
    therefore not a wiring error: it is the coupling term the book omits.
    """
    p = _elements()
    g = p.get_val
    M = (g('M_M')[0] - g('X_M')[0] * ARMS['b_M'] + g('Z_M')[0] * ARMS['l_M']
         + g('M_T')[0] - g('X_T')[0] * ARMS['b_T'] + g('Z_T')[0] * ARMS['l_T']
         - g('X_H')[0] * ARMS['b_H'] + g('Z_H')[0] * ARMS['l_H']
         - g('X_V')[0] * ARMS['b_V']
         + g('M_F')[0] + g('Z_F')[0] * ARMS['l_F']
         - g('X_F')[0] * ARMS['b_F'])

    dropped = -Q_M_EX * np.sin(B1S_EX)
    assert_near_equal(M, 612.0, 2e-2)
    assert abs(M - dropped) < 150.0


def test_elements_group_antitorque_mode_finds_the_tail_rotor_thrust():
    """thrust='antitorque' solves T_T from p. 487 instead of taking it."""
    p = _elements(thrust='antitorque')
    assert_near_equal(p.get_val('T_T')[0], 661.0, 1e-2)


def test_elements_group_feedback_loop_converges():
    """tail_rotor_feedback=True closes Y_V -> dD_int -> D_V -> Y_V, p. 509."""
    opened, closed = _elements(), _elements(tail_rotor_feedback=True)
    assert closed.model.g.nonlinear_solver._iter_count < 30
    assert_near_equal(closed.get_val('Y_V')[0],
                      opened.get_val('Y_V')[0], 2e-2)


def test_elements_group_linearized_tracks_the_nonlinear_chain():
    """At the real flight point the two forms agree to about a percent.

    They cannot be compared at vanishing angles the way a single component
    can: the downwash angles do not vanish with the unknowns, so phi_H stays
    at -0.072 rad however small Theta is made. What matters here is that the
    linearisation Tables 8.4 and 8.11 apply is worth what Prouty says it is
    at the condition he applies it to.
    """
    exact, linear = _elements(), _elements(linearized=True)
    linear.set_val('T_M_bar', np.full(1, T_M_BAR))
    linear.run_model()

    for name in ('X_H', 'Z_H', 'X_V', 'Y_V', 'X_F', 'Z_F', 'Y_M', 'Z_M'):
        a, b = exact.get_val(name)[0], linear.get_val(name)[0]
        assert abs(a - b) < 0.02 * max(abs(a), 1.0), name

    # except the two the tables drop outright, C8-1
    assert abs(exact.get_val('R_M')[0]
               - linear.get_val('R_M')[0]) > 600.0


def test_elements_group_totals():
    """Derivatives of the six sums wrt the trim unknowns, across the group."""
    p = _elements()
    data = p.check_totals(
        of=['X_M', 'Z_M', 'M_M', 'X_H', 'Z_H', 'Y_V', 'Z_V', 'X_F', 'Z_F'],
        wrt=['T_M', 'Theta', 'a1s_M', 'b1s_M', 'beta'],
        method='cs', out_stream=None)
    for key, value in data.items():
        assert value['abs error'].forward < 1e-8, key


# ------------------------------------------------------------------------
# hover_longitudinal_trim_comp -- pp. 516-517
# ------------------------------------------------------------------------

# p. 517, worked example
HOVER = dict(GW=20000.0, Dv_GW_H=0.0, Dv_GW_F=0.042, M_T=-990.0,
             dMM_da1s=DMM_DA1S_EX)
HOVER_GEOM = dict(i_M=0.0, h_M=7.5, l_M=-0.5, l_H=33.0, l_F=-0.5)


def _hover_long(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', HoverLongitudinalTrimComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in HOVER.items():
        p.set_val(k, np.full(nn, v))
    for k, v in HOVER_GEOM.items():
        p.set_val(k, v)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_hover_longitudinal_reproduces_page_517():
    """T_M = 20,877 lb, Theta = 0.026 rad, a1s_M = -0.025 rad."""
    p = _hover_long()
    assert_near_equal(p.get_val('T_M')[0], 20877.0, 1e-4)
    assert_near_equal(p.get_val('a1s_M')[0], -0.025, 1e-2)
    assert_near_equal(p.get_val('Theta')[0], 0.026, 2e-2)


def test_hover_longitudinal_printed_angles_disagree_with_themselves():
    """p. 517 prints -0.025 rad and -1.5 deg side by side; -0.025 rad is
    -1.43 deg. The unrounded solution is -1.444 deg, so the radian figure is
    the faithful one and the degree figure is rounded the wrong way.
    """
    p = _hover_long()
    assert_near_equal(p.get_val('a1s_M', units='deg')[0], -1.444, 1e-3)
    assert_near_equal(np.rad2deg(-0.025), -1.432, 1e-3)


def test_hover_thrust_carries_the_vertical_drag():
    """T_M = G.W./[1 - (D_v/G.W.)_H - (D_v/G.W.)_F], p. 517.

    Same expression as the lateral hover solution of p. 532.
    """
    p = _hover_long()
    assert_near_equal(p.get_val('T_M')[0],
                      20000.0 / (1.0 - 0.042), 1e-12)
    assert p.get_val('T_M')[0] - 20000.0 > 800.0

    clean = _hover_long(Dv_GW_F=np.zeros(1))
    assert_near_equal(clean.get_val('T_M')[0], 20000.0, 1e-12)


def test_hover_vertical_drag_also_pitches_the_helicopter():
    """It acts at l_H and l_F from the c.g., not only along Z."""
    p = _hover_long()
    moved = _hover_long(l_F=+0.5)
    assert abs(moved.get_val('a1s_M')[0] - p.get_val('a1s_M')[0]) > 1e-4


def test_hover_pitch_stiffness_is_half_thrust_times_height():
    """The denominator is (dM_M/da1s) + T_M h_M, p. 517.

    156,576 of it comes from the rotor sitting 7.5 ft above the c.g., against
    200,940 from the hub, so a teetering rotor still trims in pitch.
    """
    p = _hover_long()
    T_M = p.get_val('T_M')[0]
    assert_near_equal(T_M * HOVER_GEOM['h_M'], 156576.0, 1e-4)

    teetering = _hover_long(dMM_da1s=np.zeros(1))
    assert np.isfinite(teetering.get_val('a1s_M')[0])
    assert abs(teetering.get_val('a1s_M')[0]) < 0.15


def test_hover_tail_rotor_torque_sign_comes_from_M_T():
    """p. 517 writes it as +/- Q_T; the choice lives in blade_closest."""
    up = _hover_long()
    down = _hover_long(M_T=np.full(1, +990.0))

    # a1s_M = -(T_M c + M_T)/D, so reversing M_T shifts it by -2 M_T/D
    D = DMM_DA1S_EX + up.get_val('T_M')[0] * HOVER_GEOM['h_M']
    assert_near_equal(down.get_val('a1s_M')[0] - up.get_val('a1s_M')[0],
                      -2.0 * 990.0 / D, 1e-10)


def test_hover_longitudinal_partials():
    p = _hover_long(nn=3, GW=np.linspace(18000.0, 22000.0, 3),
                    Dv_GW_H=np.linspace(0.0, 0.02, 3),
                    Dv_GW_F=np.linspace(0.03, 0.05, 3),
                    M_T=np.linspace(-1200.0, 1200.0, 3),
                    dMM_da1s=np.linspace(1.0e5, 3.0e5, 3),
                    i_M=0.03, h_M=7.5, l_M=-0.5, l_H=33.0, l_F=-0.5)
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# longitudinal equilibrium -- Table 8.4, pp. 518-521
# ------------------------------------------------------------------------

def _equilibrium(linearized=False):
    """Run the element group, then close the three equations on its output."""
    e = _elements(linearized=linearized)
    if linearized:
        e.set_val('T_M_bar', np.full(1, T_M_BAR))
        e.run_model()

    p = om.Problem()
    model = p.model
    model.add_subsystem('x', LongXEquilibriumComp(linearized=linearized),
                        promotes=['*'])
    model.add_subsystem('z', LongZEquilibriumComp(linearized=linearized),
                        promotes=['*'])
    model.add_subsystem('m', LongMEquilibriumComp(linearized=linearized),
                        promotes=['*'])
    p.setup(force_alloc_complex=True)

    for c in 'MTHVF':
        p.set_val(f'X_{c}', e.get_val(f'X_{c}'))
        p.set_val(f'h_{c}', ARMS[f'b_{c}'])
        if c != 'V' or not linearized:
            p.set_val(f'Z_{c}', e.get_val(f'Z_{c}'))
            p.set_val(f'l_{c}', ARMS[f'l_{c}'])
    for name in ('M_M', 'M_T', 'M_F'):
        p.set_val(name, e.get_val(name))
    p.set_val('GW', np.full(1, GW_EX))
    p.set_val('Theta', np.full(1, THETA_EX))
    p.run_model()
    return e, p


def test_longitudinal_equilibrium_closes_on_the_element_group():
    """Prouty's converged unknowns should nearly annihilate all three."""
    _, p = _equilibrium()
    assert abs(p.get_val('res_X')[0]) < 5.0            # against ~800 lb
    assert abs(p.get_val('res_Z')[0]) < 40.0           # against 20,000 lb
    assert abs(p.get_val('res_M')[0]) < 900.0          # against ~12,000 ft-lb


def test_M_equilibrium_sign_convention():
    """Two rows of Table 8.4 pin l positive aft and h positive up."""
    p = om.Problem()
    p.model.add_subsystem('c', LongMEquilibriumComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)

    # Z_M l_M with Z_M = -T_M and l_M = -0.5 must give +.5 T_M
    p.set_val('Z_M', np.full(1, -T_M_EX))
    p.set_val('l_M', ARMS['l_M'])
    p.run_model()
    assert_near_equal(p.get_val('res_M')[0], 0.5 * T_M_EX, 1e-12)

    # -X_T h_T with X_T = -H_T must give +H_T h_T = 240
    p.set_val('Z_M', np.zeros(1))
    p.set_val('X_T', np.full(1, -H_T_EX))
    p.set_val('h_T', ARMS['b_T'])
    p.run_model()
    assert_near_equal(p.get_val('res_M')[0], 240.0, 1e-12)


def test_M_equilibrium_ZV_lV_row_is_missing_from_table_84():
    """C8-8. Every component contributes -X h and +Z l; the fin only -X h.

    Worth 173 ft-lb at 115 knots, against an equilibrium the book closes to
    about 129. linearized=True reproduces the omission, as it does for C8-1.
    """
    e, exact = _equilibrium()
    _, table = _equilibrium(linearized=True)

    Z_V_l_V = e.get_val('Z_V')[0] * ARMS['l_V']
    assert_near_equal(Z_V_l_V, 173.0, 3e-2)
    assert 'Z_V' not in table.model.m._var_rel_names['input']
    assert 'Z_V' in exact.model.m._var_rel_names['input']


def test_X_equilibrium_is_fuselage_drag_against_rotor_H_force():
    """The two large terms; everything else is at the 1 % level."""
    e, _ = _equilibrium()
    big = abs(e.get_val('X_F')[0]) + abs(e.get_val('X_M')[0])
    small = sum(abs(e.get_val(f'X_{c}')[0]) for c in 'THV')
    assert small / big < 0.10


def test_Z_equilibrium_cosine_is_the_mildest_linearisation():
    """cos(Theta) -> 1 is worth 6 lb at 1.4 degrees of pitch."""
    _, exact = _equilibrium()
    _, table = _equilibrium(linearized=True)
    weight_gap = GW_EX * (1.0 - np.cos(THETA_EX))
    assert_near_equal(weight_gap, 2.7, 5e-2)
    assert abs(exact.get_val('res_Z')[0]
               - table.get_val('res_Z')[0]) < 50.0


@pytest.mark.parametrize('linearized', [False, True])
def test_longitudinal_equilibrium_partials(linearized):
    _, p = _equilibrium(linearized=linearized)
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


def test_table_84_mode_reproduces_proutys_own_residual():
    """Prouty's linearised M equation at his converged unknowns leaves
    129 ft-lb: -2871 + 355485 a1s + 32253 Theta + .5004 T_M.

    linearized=True leaves 103, so the package closes the book's equations
    as well as the book does. The nonlinear residual of 785 is then almost
    entirely the two terms Table 8.4 drops: 538 from C8-1 and 173 from C8-8.
    """
    prouty = (-2871.0 + 355485.0 * A1S_EX + 32253.0 * THETA_EX
              + 0.5004 * T_M_EX)
    assert_near_equal(prouty, 144.0, 5e-2)

    e, exact = _equilibrium()
    _, table = _equilibrium(linearized=True)
    assert abs(table.get_val('res_M')[0]) < 150.0

    dropped = (-Q_M_EX * np.sin(B1S_EX)) + e.get_val('Z_V')[0] * ARMS['l_V']
    gap = exact.get_val('res_M')[0] - table.get_val('res_M')[0]
    assert_near_equal(gap, dropped, 5e-2)


# ------------------------------------------------------------------------
# longitudinal_trim_group -- pp. 516-522 solved
# ------------------------------------------------------------------------

def _long_trim(nn=1, **opts):
    p = om.Problem()
    p.model.add_subsystem('t', LongitudinalTrimGroup(num_nodes=nn, **opts),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    one = np.ones(nn)
    for name, value in (
            ('GW', GW_EX), ('q', Q_EX), ('H_a1s0', H_A1S0_BAR),
            ('Q_M', Q_M_EX), ('dMM_da1s', DMM_DA1S_EX),
            ('dRM_db1s', DMM_DA1S_EX), ('H_T', H_T_EX), ('Q_T', Q_T_EX),
            ('b1s_T', B1S_T_EX), ('qH_q', QH_Q_EX), ('qV_q', QV_Q_EX),
            ('eta_MV', ETA_MV_EX), ('f', D_F_EX / Q_EX)):
        p.set_val(name, value * one)
    for name, value in (
            ('A_M', A_M_EX), ('A_T', A_T_EX), ('i_H', I_H_EX),
            ('A_H', A_H_EX), ('a_H', A_SLOPE_EX), ('A_R_H', A_H_AR_EX),
            ('delta_H', DELTA_H_EX), ('C_D0_H', CD0_H_EX), ('A_V', A_V_EX),
            ('a_V', A_V_SLOPE_EX), ('A_R_V', A_V_AR_EX),
            ('delta_V', DELTA_V_EX), ('C_D0_V', CD0_V_EX),
            ('alpha_LO_V', ALPHA_LO_V_EX), ('R_T', R_T_EX), ('b_V', B_V_EX),
            ('K_int', K_INT_EX)):
        p.set_val(name, value)
    for name, arm in ARMS.items():
        p.set_val(name.replace('b_', 'h_'), arm)
    p.set_val('Y_V_bar', LIFT_V_EX * one)
    if opts.get('linearized', False):
        p.set_val('T_M_bar', T_M_BAR * one)
    p.run_model()
    return p


@pytest.mark.parametrize('linearized', [True, False])
def test_longitudinal_trim_reproduces_page_522(linearized):
    """T_M = 20,586 lb, Theta = -0.9 deg, a1s_M = -1.1 deg."""
    p = _long_trim(linearized=linearized)
    assert_near_equal(p.get_val('T_M')[0], 20586.0, 3e-3)
    assert_near_equal(p.get_val('Theta', units='deg')[0], -0.9, 4e-2)
    assert_near_equal(p.get_val('a1s_M', units='deg')[0], -1.1, 4e-2)


def test_longitudinal_trim_drives_all_three_residuals_to_zero():
    p = _long_trim()
    for name in ('res_X', 'res_Z', 'res_M'):
        assert abs(p.get_val(name)[0]) < 1e-8, name


def test_longitudinal_trim_converges_in_a_few_newton_steps():
    """From a cold start 600 lb and half a degree away from the answer."""
    p = _long_trim()
    assert p.model.t.nonlinear_solver._iter_count <= 6


def test_longitudinal_trim_is_insensitive_to_the_starting_point():
    """Bounds and scaling should make the basin wide, not just deep."""
    p = _long_trim()
    reference = p.get_val('T_M')[0]

    for guess in (12000.0, 35000.0):
        q = _long_trim()
        q.set_val('T_M', np.full(1, guess))
        q.set_val('Theta', np.full(1, 0.3))
        q.set_val('a1s_M', np.full(1, 0.2))
        q.run_model()
        assert_near_equal(q.get_val('T_M')[0], reference, 1e-6)


def test_longitudinal_trim_two_methods_of_page_516_agree():
    """p. 516 offers a simultaneous linear solve and an iteration.

    linearized=True is the first, linearized=False the second. They differ
    by the terms of C8-1 and C8-8 and by the small angles, and land within
    0.03 deg of each other on both angles.
    """
    linear, exact = _long_trim(linearized=True), _long_trim()
    assert abs(linear.get_val('T_M')[0] - exact.get_val('T_M')[0]) < 20.0
    for name in ('Theta', 'a1s_M'):
        gap = (linear.get_val(name, units='deg')[0]
               - exact.get_val(name, units='deg')[0])
        assert abs(gap) < 0.05, name


def test_longitudinal_trim_totals():
    """Derivatives of the converged states wrt the design inputs."""
    p = _long_trim()
    data = p.check_totals(of=['T_M', 'Theta', 'a1s_M'],
                          wrt=['GW', 'q', 'A_H', 'i_H', 'f'],
                          method='fd', out_stream=None)
    for key, value in data.items():
        analytic = value.get('J_fwd', value.get('J_rev'))
        finite = value['J_fd']
        scale = max(np.abs(finite).max(), 1e-8)
        assert np.abs(analytic - finite).max() / scale < 1e-4, key


# ------------------------------------------------------------------------
# long_cyclic_pitch_comp -- p. 522
# ------------------------------------------------------------------------

B1_A1S_EX = np.deg2rad(7.8)                 # Chapter 3 performance charts


def test_long_cyclic_pitch_reproduces_page_522():
    """B_1 = 7.8 - (-1.1) = 8.9 deg."""
    p = _one(LongCyclicPitchComp, B1_a1s=np.full(1, B1_A1S_EX),
             a1s_M=np.full(1, np.deg2rad(-1.1)))
    assert_near_equal(p.get_val('B_1', units='deg')[0], 8.9, 1e-6)


def test_long_cyclic_pitch_splits_what_chapter_3_could_not():
    """Chapter 3 sets only B_1 + a1s; the split needs the M equation.

    Reading the stick position off the Chapter 3 sum alone would understate
    the forward cyclic by the whole of a1s_M, 12 % of the travel in use.
    """
    p = _long_trim()
    p.set_val('B1_a1s', np.full(1, B1_A1S_EX))
    p.run_model()

    a1s = p.get_val('a1s_M', units='deg')[0]
    B_1 = p.get_val('B_1', units='deg')[0]
    assert_near_equal(B_1, 7.8 - a1s, 1e-9)
    assert_near_equal(B_1, 8.9, 4e-2)
    assert abs(a1s) / B_1 > 0.10


def test_long_cyclic_pitch_is_outside_the_newton_loop():
    """Nothing feeds back from B_1.

    Changing B1_a1s on a converged model and re-running must leave the three
    states untouched and cost Newton no iterations at all. If the cyclic
    were inside the loop, it would cost some.
    """
    p = _long_trim()
    before = [p.get_val(n)[0] for n in ('T_M', 'Theta', 'a1s_M')]

    p.set_val('B1_a1s', np.full(1, B1_A1S_EX))
    p.run_model()

    assert p.model.t.nonlinear_solver._iter_count == 0
    for name, value in zip(('T_M', 'Theta', 'a1s_M'), before):
        assert_near_equal(p.get_val(name)[0], value, 1e-12)
    assert_near_equal(p.get_val('B_1', units='deg')[0], 8.9, 4e-2)


def test_long_cyclic_pitch_partials():
    p = _one(LongCyclicPitchComp, nn=3,
             B1_a1s=np.linspace(0.05, 0.20, 3),
             a1s_M=np.linspace(-0.05, 0.01, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-12, rtol=1e-12)


# ------------------------------------------------------------------------
# hover_lateral_trim_comp -- pp. 531-532
# ------------------------------------------------------------------------

# Table 8.10 p. 533, example helicopter hovering out of ground effect
HOVER_LAT = dict(GW=20000.0, T_M=20840.0, T_T=1540.0,
                 dRM_db1s=DMM_DA1S_EX)
HOVER_LAT_GEOM = dict(y_M=0.0, h_M=7.5, h_T=6.0)


def _hover_lat(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', HoverLateralTrimComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in HOVER_LAT.items():
        p.set_val(k, np.full(nn, v))
    for k, v in HOVER_LAT_GEOM.items():
        p.set_val(k, v)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_hover_lateral_roll_angle_matches_page_532():
    """Phi = -0.050 rad = -2.9 deg."""
    p = _hover_lat()
    assert_near_equal(p.get_val('Phi')[0], -0.050, 2e-3)
    assert_near_equal(p.get_val('Phi', units='deg')[0], -2.9, 2e-2)


def test_hover_lateral_flapping_disagrees_with_its_own_roll_angle():
    """C8-9. p. 532 prints b1s_M = -0.027; Table 8.10's numbers give -0.0259.

    The printed roll angle settles it. Rebuilding Phi from -0.027 gives
    -0.0489, which would print as -0.049 rad and -2.8 deg, not the -0.050
    and -2.9 the same page shows.
    """
    p = _hover_lat()
    assert_near_equal(p.get_val('b1s_M')[0], -0.025865, 1e-4)
    assert_near_equal(p.get_val('b1s_M', units='deg')[0], -1.5, 2e-2)

    from_printed = -(1540.0 + 20840.0 * -0.027) / 20000.0
    assert_near_equal(from_printed, -0.04887, 1e-3)
    assert abs(from_printed - (-0.050)) > abs(p.get_val('Phi')[0] - (-0.050))


@pytest.mark.parametrize('case,geometry,stiffness', [
    ('teetering, tail rotor as high as the main rotor',
     dict(y_M=0.0, h_M=7.5, h_T=7.5), 0.0),
    ('any rotor, tail rotor as low as the c.g.',
     dict(y_M=0.0, h_M=7.5, h_T=0.0), DMM_DA1S_EX),
    ('very high offset, any tail rotor height',
     dict(y_M=0.4, h_M=7.5, h_T=6.0), 1.0e12),
])
def test_hover_lateral_special_cases_of_table_89(case, geometry, stiffness):
    """Table 8.9 p. 532. Each row zeroes one of the two unknowns."""
    p = _hover_lat(dRM_db1s=np.full(1, stiffness), **geometry)
    b1s, phi = p.get_val('b1s_M')[0], p.get_val('Phi')[0]

    if 'teetering' in case:
        assert_near_equal(phi, 0.0, 1e-12)
        assert_near_equal(b1s, -1540.0 / 20840.0, 1e-12)   # -T_T/T_M ~ /G.W.
    else:
        assert abs(b1s) < 1e-9, case
        assert_near_equal(phi, -1540.0 / 20000.0, 1e-7)


def test_hover_lateral_the_two_ways_of_carrying_the_tail_rotor():
    """Flapping and bank share the job; nothing else can do it.

    Y: T_M b1s_M + T_T = -G.W. Phi, p. 531.
    """
    p = _hover_lat()
    side_force = (HOVER_LAT['T_M'] * p.get_val('b1s_M')[0]
                  + HOVER_LAT['T_T'])
    assert_near_equal(side_force,
                      -HOVER_LAT['GW'] * p.get_val('Phi')[0], 1e-10)


def test_hover_lateral_partials():
    p = _hover_lat(nn=3, GW=np.linspace(18000.0, 22000.0, 3),
                   T_M=np.linspace(19000.0, 23000.0, 3),
                   T_T=np.linspace(1200.0, 1800.0, 3),
                   dRM_db1s=np.linspace(1.0e5, 3.0e5, 3),
                   y_M=0.3, h_M=7.5, h_T=6.0)
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# lateral-directional equilibrium -- Table 8.11, pp. 536-537
# ------------------------------------------------------------------------

# Table 8.11 as printed, collected per equation: constant, b1s_M, beta, T_T
ROW_Y = (415.0, 20606.0, -13068.0, 0.809)
ROW_R = (1245.0, 355485.0, 341.0, 5.43)
ROW_N = (20201.0, 12363.0, 69973.0, -30.32)


def _printed_rows(b1s, beta, T_T):
    return tuple(c + p * b1s + q * beta + r * T_T
                 for c, p, q, r in (ROW_Y, ROW_R, ROW_N))


def test_table_811_rows_reproduce_figure_831_at_zero_sideslip():
    """Fig. 8.31 p. 538 annotates b1s_M = -0.78 deg, T_T = 661 lb, Phi = -1.9.

    Solving the printed R and N rows for b1s_M and T_T at beta = 0 lands on
    the first two, which is what makes the third meaningful as a check on
    the sign of the weight term.
    """
    A = np.array([[ROW_R[1], ROW_R[3]], [ROW_N[1], ROW_N[3]]])
    b = -np.array([ROW_R[0], ROW_N[0]])
    b1s, T_T = np.linalg.solve(A, b)

    assert_near_equal(np.rad2deg(b1s), -0.78, 2e-2)
    assert_near_equal(T_T, 661.0, 1e-3)


def test_Y_equilibrium_weight_sign_follows_page_531_not_table_811():
    """C8-10. Table 8.11 prints -G.W. Phi; p. 531 and Figure 8.31 say plus.

    With the printed minus the roll angle comes out +1.92 deg, right side
    down. Figure 8.31 shows -1.9 deg, left.
    """
    A = np.array([[ROW_R[1], ROW_R[3]], [ROW_N[1], ROW_N[3]]])
    b1s, T_T = np.linalg.solve(A, -np.array([ROW_R[0], ROW_N[0]]))
    aero_Y = ROW_Y[0] + ROW_Y[1] * b1s + ROW_Y[3] * T_T

    p = _one(LatYEquilibriumComp, opts={'linearized': True},
             Y_M=np.full(1, aero_Y), GW=np.full(1, GW_EX),
             Phi=np.full(1, np.deg2rad(-1.9)))
    assert abs(p.get_val('res_Y')[0]) < 10.0

    wrong_sign = aero_Y - GW_EX * np.deg2rad(-1.9)
    assert abs(wrong_sign) > 600.0


def test_lateral_equilibrium_at_the_zero_bank_point():
    """Fig. 8.31: Phi = 0 at beta = 3.2 deg, b1s_M = -0.89, T_T = 789."""
    residuals = _printed_rows(np.deg2rad(-0.89), np.deg2rad(3.2), 789.0)
    for name, value in zip(('Y', 'R', 'N'), residuals):
        assert abs(value) < 30.0, name


def _lat_eq(cls, linearized=False, **ivc):
    return _one(cls, opts={'linearized': linearized}, **ivc)


def test_R_equilibrium_has_the_lateral_cg_term():
    """Z_M y_M has no counterpart in the pitching moment equation."""
    p = _lat_eq(LatREquilibriumComp, Z_M=np.full(1, -T_M_EX), y_M=0.4)
    assert_near_equal(p.get_val('res_R')[0], -T_M_EX * 0.4, 1e-12)

    centred = _lat_eq(LatREquilibriumComp, Z_M=np.full(1, -T_M_EX), y_M=0.0)
    assert_near_equal(centred.get_val('res_R')[0], 0.0, 1e-12)


def test_R_equilibrium_stiffness_matches_table_811():
    """dR_M/db1s + T_M h_M = 200,940 + 154,545 = 355,485, as printed."""
    assert_near_equal(DMM_DA1S_EX + T_M_BAR * ARMS['b_M'], 355485.0, 1e-3)


def test_N_equilibrium_reduces_to_the_hover_antitorque_relation():
    """p. 531: Q_M - l_T T_T = 0 when the airframe is dropped."""
    p = _lat_eq(LatNEquilibriumComp, N_M=np.full(1, Q_M_EX),
                Y_T=np.full(1, T_T_EX), l_T=ARMS['l_T'])
    assert_near_equal(p.get_val('res_N')[0],
                      Q_M_EX - ARMS['l_T'] * T_T_EX, 1e-12)


def test_N_equilibrium_directional_stability_is_the_fin():
    """Table 8.11 p. 537 splits the beta coefficient across four rows.

    The fin is the only large positive one, and the fuselage weathercocks
    the wrong way. Without the fin the helicopter is directionally unstable.
    """
    fin, fuselage_moment = 101290.0, -36900.0
    rotor, fuselage_side = 228.0, 5355.0
    assert_near_equal(fin + fuselage_moment + rotor + fuselage_side,
                      ROW_N[2], 1e-12)
    assert fin + fuselage_moment + rotor + fuselage_side > 0.0
    assert rotor + fuselage_side + fuselage_moment < 0.0


@pytest.mark.parametrize('linearized', [False, True])
def test_lateral_equilibrium_partials(linearized):
    three = np.linspace(-500.0, 500.0, 3)
    kw = dict(Y_M=three, Y_T=three + 700.0, Y_V=three, Y_F=three)

    y = _one(LatYEquilibriumComp, nn=3, opts={'linearized': linearized},
             GW=np.full(3, GW_EX), Phi=np.linspace(-0.1, 0.1, 3),
             Theta=np.linspace(-0.05, 0.05, 3), **kw)
    r = _one(LatREquilibriumComp, nn=3, opts={'linearized': linearized},
             Z_M=np.full(3, -T_M_EX), y_M=0.4, h_M=7.5, h_T=6.0, h_V=3.0,
             h_F=0.5, R_M=three, R_F=three, **kw)
    n = _one(LatNEquilibriumComp, nn=3, opts={'linearized': linearized},
             l_M=-0.5, l_T=37.0, l_V=35.0, l_F=-0.5,
             N_M=np.full(3, Q_M_EX), N_F=three, **kw)
    for p in (y, r, n):
        assert_check_partials(p.check_partials(method='cs', out_stream=None),
                              atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# lateral_directional_trim_group -- pp. 531-538 solved
# ------------------------------------------------------------------------

LAT_ARMS = dict(h_M=7.5, l_M=-0.5, h_T=6.0, l_T=37.0, h_V=3.0, l_V=35.0,
                h_F=0.5, l_F=-0.5)


def _lat_trim(nn=1, **opts):
    p = om.Problem()
    p.model.add_subsystem(
        't', LateralDirectionalTrimGroup(num_nodes=nn, **opts), promotes=['*'])
    p.setup(force_alloc_complex=True)

    one = np.ones(nn)
    for name, value in (
            ('GW', GW_EX), ('q', Q_EX), ('H_a1s0', H_A1S0_BAR),
            ('Q_M', Q_M_EX), ('dMM_da1s', DMM_DA1S_EX),
            ('dRM_db1s', DMM_DA1S_EX), ('H_T', H_T_EX), ('Q_T', Q_T_EX),
            ('b1s_T', B1S_T_EX), ('qH_q', QH_Q_EX), ('qV_q', QV_Q_EX),
            ('eta_MV', ETA_MV_EX), ('f', D_F_EX / Q_EX),
            ('T_M', T_M_EX), ('Theta', THETA_EX), ('a1s_M', A1S_EX)):
        p.set_val(name, value * one)
    for name, value in (
            ('A_M', A_M_EX), ('A_T', A_T_EX), ('i_H', I_H_EX),
            ('A_H', A_H_EX), ('a_H', A_SLOPE_EX), ('A_R_H', A_H_AR_EX),
            ('delta_H', DELTA_H_EX), ('C_D0_H', CD0_H_EX), ('A_V', A_V_EX),
            ('a_V', A_V_SLOPE_EX), ('A_R_V', A_V_AR_EX),
            ('delta_V', DELTA_V_EX), ('C_D0_V', CD0_V_EX),
            ('alpha_LO_V', ALPHA_LO_V_EX), ('R_T', R_T_EX), ('b_V', B_V_EX),
            ('K_int', K_INT_EX), ('y_M', 0.0)):
        p.set_val(name, value)
    for name, arm in LAT_ARMS.items():
        p.set_val(name, arm)
    p.set_val('Y_V_bar', LIFT_V_EX * one)
    if opts.get('linearized', False):
        p.set_val('T_M_bar', T_M_BAR * one)
    p.run_model()
    return p


def test_lateral_trim_reproduces_figure_831_at_zero_sideslip():
    """b1s_M = -0.78 deg, T_T = 661 lb, Phi = -1.9 deg."""
    p = _lat_trim(mode='zero_sideslip', linearized=True)
    assert_near_equal(p.get_val('b1s_M', units='deg')[0], -0.78, 2e-2)
    assert_near_equal(p.get_val('T_T')[0], 661.0, 1e-2)
    assert_near_equal(p.get_val('Phi', units='deg')[0], -1.9, 2e-2)


def test_lateral_trim_reproduces_figure_831_at_zero_bank():
    """b1s_M = -0.89 deg, T_T = 789 lb, beta = 3.2 deg."""
    p = _lat_trim(mode='zero_bank', linearized=True)
    assert_near_equal(p.get_val('b1s_M', units='deg')[0], -0.89, 2e-2)
    assert_near_equal(p.get_val('beta', units='deg')[0], 3.2, 2e-2)
    assert_near_equal(p.get_val('T_T')[0], 789.0, 3e-2)
    assert_near_equal(p.get_val('Phi')[0], 0.0, 1e-12)


def test_lateral_trim_confirms_C8_1_through_the_whole_chain():
    """C8-1 predicted b1s_M would move from -0.78 to -0.67 deg.

    That estimate was made by eliminating T_T between the two printed rows
    of Table 8.11 and adding Q_M sin(a1s_M) to the constant. Solving the
    full nonlinear chain instead lands at -0.675, which the linearised run
    does not: the omission the tables make is worth 13 % of the lateral
    flapping, and nothing else in the model accounts for it.
    """
    table = _lat_trim(mode='zero_sideslip', linearized=True)
    exact = _lat_trim(mode='zero_sideslip')

    assert_near_equal(exact.get_val('b1s_M', units='deg')[0], -0.675, 3e-2)
    shift = (exact.get_val('b1s_M', units='deg')[0]
             - table.get_val('b1s_M', units='deg')[0])
    assert_near_equal(shift, 0.106, 1e-1)

    # the tail rotor barely notices; it is set by the torque balance
    assert abs(exact.get_val('T_T')[0] - table.get_val('T_T')[0]) < 2.0


def test_lateral_trim_zero_sideslip_is_given_sideslip_pinned_at_zero():
    """p. 535 names them separately; the arithmetic is the same."""
    pinned = _lat_trim(mode='zero_sideslip', linearized=True)
    given = _lat_trim(mode='given_sideslip', linearized=True)
    for name in ('b1s_M', 'Phi', 'T_T'):
        assert_near_equal(given.get_val(name)[0],
                          pinned.get_val(name)[0], 1e-10)


def test_lateral_trim_sideslip_costs_bank_and_tail_rotor_thrust():
    """Figure 8.31 is three straight lines; check the slopes have sense.

    Flying right sideslip rolls the helicopter right and loads the tail
    rotor, because the fin is now producing side force the wrong way.
    """
    left = _lat_trim(mode='given_sideslip', linearized=True)
    left.set_val('beta', np.deg2rad(-3.0))
    left.run_model()
    right = _lat_trim(mode='given_sideslip', linearized=True)
    right.set_val('beta', np.deg2rad(3.0))
    right.run_model()

    assert right.get_val('Phi')[0] > left.get_val('Phi')[0]
    assert right.get_val('T_T')[0] > left.get_val('T_T')[0]
    assert right.get_val('b1s_M')[0] < left.get_val('b1s_M')[0]


def test_lateral_trim_drives_all_three_residuals_to_zero():
    for mode in ('zero_sideslip', 'zero_bank'):
        p = _lat_trim(mode=mode)
        # forces in lbf, moments in lbf-ft against terms reaching 35,000
        for name in ('res_Y', 'res_R', 'res_N'):
            assert abs(p.get_val(name)[0]) < 1e-5, (mode, name)


def test_lateral_trim_converges_in_a_few_newton_steps():
    p = _lat_trim(mode='zero_bank')
    assert p.model.t.nonlinear_solver._iter_count <= 6


def test_lateral_trim_totals():
    p = _lat_trim(mode='zero_sideslip')
    data = p.check_totals(of=['b1s_M', 'Phi', 'T_T'],
                          wrt=['GW', 'q', 'A_V', 'l_T', 'Q_M'],
                          method='cs', out_stream=None)
    for key, value in data.items():
        analytic = value.get('J_fwd', value.get('J_rev'))
        finite = value['J_fd']
        scale = max(np.abs(finite).max(), 1e-8)
        assert np.abs(analytic - finite).max() / scale < 1e-4, key


# ------------------------------------------------------------------------
# lat_cyclic_pitch_comp and tail_rotor_tpp_angle_comp -- p. 535
# ------------------------------------------------------------------------

A1_B1S_EX = np.deg2rad(-2.3)                # Chapter 3, p. 169
B_1_EX = np.deg2rad(8.9)                    # p. 522


def test_lat_cyclic_pitch_at_zero_sideslip():
    """A_1 = (A_1 - b1s) + b1s, with b1s = -0.78 deg from Figure 8.31."""
    p = _one(LatCyclicPitchComp, A1_b1s=np.full(1, A1_B1S_EX),
             b1s_M=np.full(1, np.deg2rad(-0.78)))
    assert_near_equal(p.get_val('A_1', units='deg')[0], -3.08, 1e-2)


def test_lat_cyclic_pitch_uses_the_chapter_3_convention():
    """C8-11. p. 535 prints (A_1 + b1s) - b1s; p. 169 sets A_1 - b1s.

    The two differ by twice the flapping, 1.56 deg here against a 2.3 deg
    aerodynamic requirement, so the choice is not cosmetic.
    """
    b1s = np.deg2rad(-0.78)
    ours = _one(LatCyclicPitchComp, A1_b1s=np.full(1, A1_B1S_EX),
                b1s_M=np.full(1, b1s)).get_val('A_1')[0]
    as_printed = A1_B1S_EX - b1s
    assert_near_equal(np.rad2deg(ours - as_printed), 2.0 * -0.78, 1e-6)


def test_lat_cyclic_pitch_sideslip_term_tilts_figure_831():
    """B_1 sin(beta) is the slope of the lateral control line, p. 538."""
    straight = _one(LatCyclicPitchComp, A1_b1s=np.full(1, A1_B1S_EX),
                    b1s_M=np.full(1, np.deg2rad(-0.78)),
                    B_1=np.full(1, B_1_EX))
    yawed = _one(LatCyclicPitchComp, A1_b1s=np.full(1, A1_B1S_EX),
                 b1s_M=np.full(1, np.deg2rad(-0.78)),
                 B_1=np.full(1, B_1_EX), beta=np.full(1, np.deg2rad(3.0)))
    shift = (yawed.get_val('A_1', units='deg')[0]
             - straight.get_val('A_1', units='deg')[0])
    assert_near_equal(shift, 8.9 * np.sin(np.deg2rad(3.0)), 1e-6)
    assert_near_equal(shift, 0.47, 2e-2)


def test_lat_cyclic_trim_contributes_less_than_longitudinal():
    """0.78 deg of 3.08 laterally, against 1.1 deg of 8.9 longitudinally."""
    lateral = 0.78 / 3.08
    longitudinal = 1.1 / 8.9
    assert lateral > longitudinal


def test_lat_cyclic_pitch_partials():
    p = _one(LatCyclicPitchComp, nn=3, A1_b1s=np.linspace(-0.05, -0.03, 3),
             b1s_M=np.linspace(-0.02, 0.0, 3), B_1=np.full(3, B_1_EX),
             beta=np.linspace(-0.1, 0.1, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-12, rtol=1e-12)


def test_tail_rotor_tpp_angle_inverts_the_chapter_3_inflow_relation():
    """lambda' = mu tan(alpha_TPP) - sigma (C_T/sigma)/(2 mu), p. 167."""
    mu, sigma, CT_sigma = 0.25, 0.15, 0.09
    alpha = np.deg2rad(-4.0)
    lambda_p = mu * np.tan(alpha) - sigma * CT_sigma / (2.0 * mu)

    p = _one(TailRotorTppAngleComp, lambda_p_T=np.full(1, lambda_p),
             mu_T=np.full(1, mu), sigma_T=np.full(1, sigma),
             CT_sigma_T=np.full(1, CT_sigma))
    assert_near_equal(p.get_val('alpha_TPP_T')[0], alpha, 1e-12)


def test_tail_rotor_flapping_carries_the_sideslip():
    """a1s_T = alpha_TPP_T + beta, p. 535."""
    beta = np.deg2rad(3.2)
    p = _one(TailRotorTppAngleComp, lambda_p_T=np.full(1, -0.05),
             mu_T=np.full(1, 0.25), sigma_T=np.full(1, 0.15),
             CT_sigma_T=np.full(1, 0.09), beta=np.full(1, beta))
    assert_near_equal(p.get_val('a1s_T')[0] - p.get_val('alpha_TPP_T')[0],
                      beta, 1e-12)


def test_tail_rotor_tpp_angle_partials():
    p = _one(TailRotorTppAngleComp, nn=3,
             lambda_p_T=np.linspace(-0.10, -0.02, 3),
             mu_T=np.linspace(0.15, 0.35, 3),
             sigma_T=np.full(3, 0.15),
             CT_sigma_T=np.linspace(0.05, 0.12, 3),
             beta=np.linspace(-0.05, 0.05, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# longitudinal post-processing -- pp. 525-530
# ------------------------------------------------------------------------

def test_maneuver_weight_at_the_table_87_condition():
    """p. 529 trims at n GW; Table 8.7 uses n = 1.3 at 20,000 lb."""
    p = _one(ManeuverWeightComp, GW=np.full(1, GW_EX), n=np.full(1, 1.3))
    assert_near_equal(p.get_val('GW_eff')[0], 26000.0, 1e-12)


def test_maneuver_weight_partials():
    p = _one(ManeuverWeightComp, nn=3, GW=np.linspace(18000.0, 22000.0, 3),
             n=np.linspace(1.0, 2.0, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


def _gradient(sweep, B_1_deg, sweep_var):
    return _one(TrimGradientComp, nn=len(B_1_deg), opts={'sweep': sweep},
                B_1=np.deg2rad(B_1_deg), sweep_var=np.array(sweep_var))


def test_speed_stability_of_page_527():
    """B_1 goes 8.9 -> 9.3 deg between 115 and 135 knots: stable.

    p. 527 reads it as the pilot having to hold more forward cyclic to dive
    at 135 knots than to fly level at 115.
    """
    p = _gradient('speed', [8.9, 9.3], [115.0, 135.0])
    assert p.get_val('stability')[0] > 0.0
    assert_near_equal(p.get_val('B1_shift', units='deg')[0], 0.4, 1e-6)


def test_angle_of_attack_stability_of_table_87():
    """B_1 goes 8.9 -> 11.4 deg between 1.0 and 1.3 g: unstable.

    p. 529 calls the forward stick in the turn an indication that the
    stabiliser is not large enough for positive angle of attack stability.
    """
    p = _gradient('load_factor', [8.9, 11.4], [1.0, 1.3])
    assert p.get_val('stability')[0] < 0.0
    assert_near_equal(p.get_val('B1_shift', units='deg')[0], 2.5, 1e-6)


def test_the_two_sign_conventions_are_opposite():
    """Rising B_1 is stable in speed and unstable in load factor.

    The same numbers, read two ways. Getting this backwards produces a model
    that calls an unstable helicopter stable.
    """
    rising = [8.9, 11.4]
    speed = _gradient('speed', rising, [1.0, 1.3])
    load = _gradient('load_factor', rising, [1.0, 1.3])
    assert_near_equal(speed.get_val('stability')[0],
                      -load.get_val('stability')[0], 1e-12)


def test_power_effects_of_table_88():
    """B_1 = 8.5, 8.8, 9.4 deg from autorotation through climb, p. 530.

    0.9 deg across the whole power range, which p. 530 calls practically no
    stick motion -- and attributes to the stabiliser being too small, the
    same deficiency the p. 529 test exposes.
    """
    p = _gradient('climb', [8.5, 8.8, 9.4], [-9.2, 0.0, 9.7])
    assert_near_equal(p.get_val('B1_shift', units='deg')[0], 0.9, 1e-6)
    assert p.get_val('dB1_dsweep').shape == (2,)
    assert 'stability' not in p.model.comp._var_rel_names['output']


def test_trim_gradient_needs_two_nodes():
    with pytest.raises(ValueError):
        _one(TrimGradientComp, nn=1)


@pytest.mark.parametrize('sweep', ['speed', 'load_factor', 'climb'])
def test_trim_gradient_partials(sweep):
    p = _one(TrimGradientComp, nn=3, opts={'sweep': sweep},
             B_1=np.deg2rad([8.5, 8.8, 9.4]),
             sweep_var=np.array([-9.2, 0.0, 9.7]))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# vert_stab_effective_ar_comp -- p. 504, Figure 8.19 p. 505
# ------------------------------------------------------------------------

# p. 504, example helicopter
FIN = dict(A_R_geo=1.8, bV_2r1=7.7 / 1.5, lambda_V=0.21, ZH_bV=0.0,
           x_cV=2.5 / 4.25, SH_SV=18.0 / 33.0)


def _fin_ar(**ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', VertStabEffectiveARComp(), promotes=['*'])
    p.setup()
    for k, v in {**FIN, **ivc}.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_figure_819_reproduces_the_three_readings_of_page_504():
    """f_B = 1.05, f_H = 1.1, K_H = 0.64, giving A.R.eff = 3.2."""
    p = _fin_ar()
    assert_near_equal(p.get_val('f_B')[0], 1.05, 2e-2)
    assert_near_equal(p.get_val('f_H')[0], 1.10, 4e-2)
    assert_near_equal(p.get_val('K_H')[0], 0.64, 4e-2)
    assert_near_equal(p.get_val('A_R_eff')[0], 3.2, 5e-2)


def test_end_plating_nearly_doubles_the_fin_aspect_ratio():
    """1.8 geometric against 3.2 effective, p. 504.

    Sizing a fin on its geometric aspect ratio would undersize it: through
    the Helmbold relation the difference is worth 40 % on the lift slope.
    """
    p = _fin_ar()
    assert p.get_val('A_R_eff')[0] / FIN['A_R_geo'] > 1.7

    isolated = _slope(A_R=FIN['A_R_geo'], sweep=np.deg2rad(27.0))
    plated = _slope(A_R=p.get_val('A_R_eff')[0], sweep=np.deg2rad(27.0))
    assert plated.get_val('a')[0] / isolated.get_val('a')[0] > 1.35


def test_figure_819a_peaks_where_the_printed_curves_do():
    """Both curves top out near b_V/2r_1 = 2, at 1.63 and 1.50."""
    low = [_fin_ar(bV_2r1=x, lambda_V=0.3).get_val('f_B')[0]
           for x in np.arange(0.5, 7.0, 0.1)]
    peak = 0.5 + 0.1 * int(np.argmax(low))
    assert_near_equal(peak, 2.0, 1e-1)
    assert_near_equal(max(low), 1.63, 2e-2)

    unit = _fin_ar(bV_2r1=2.0, lambda_V=1.0).get_val('f_B')[0]
    assert_near_equal(unit, 1.50, 2e-2)


def test_figure_819a_curves_converge_without_meeting():
    """They touch on paper past b_V/2r_1 = 5.2 but stay two lines.

    The dark band there is 5 to 6 pixels where an isolated line is 2 to 3,
    and splitting it by thickness recovers a separation that falls steadily
    to 0.008 at the right frame without ever reaching zero.
    """
    sep = []
    for x in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        low = _fin_ar(bV_2r1=x, lambda_V=0.3).get_val('f_B')[0]
        unit = _fin_ar(bV_2r1=x, lambda_V=1.0).get_val('f_B')[0]
        sep.append(low - unit)
    assert all(s > 0.005 for s in sep)
    assert all(sep[i] > sep[i + 1] for i in range(len(sep) - 1))
    assert_near_equal(sep[-1], 0.008, 3e-1)


def test_figure_819b_has_a_minimum_near_minus_half():
    """The horizontal tail helps least when it sits mid-fin, p. 505."""
    z = np.arange(0.0, -0.95, -0.02)
    f = [_fin_ar(ZH_bV=v).get_val('f_H')[0] for v in z]
    assert_near_equal(z[int(np.argmin(f))], -0.5, 3e-1)
    assert 0.85 < min(f) < 0.92
    assert _fin_ar(ZH_bV=-0.95).get_val('f_H')[0] > 1.4


def test_figure_819b_curves_stay_ordered_to_the_end_of_the_chart():
    """A higher x/c_V gives a larger factor at every Z_H/b_V.

    The four look merged past -0.78 only because they turn steep there; row
    tracking keeps them apart, and the ordering is the check that it worked.
    """
    for z in (0.0, -0.3, -0.6, -0.8, -0.95):
        f = [_fin_ar(ZH_bV=z, x_cV=x).get_val('f_H')[0]
             for x in (0.5, 0.6, 0.7, 0.8)]
        assert all(f[i] <= f[i + 1] + 2e-3 for i in range(3)), z
        assert f[3] - f[0] > 0.02, z


def test_taper_blend_is_smooth_across_the_two_curves():
    """lambda_V between 0.6 and 1.0 is a smoothstep, not a corner."""
    lam = np.linspace(0.4, 1.2, 81)
    f = np.array([_fin_ar(bV_2r1=2.0, lambda_V=v).get_val('f_B')[0]
                  for v in lam])
    second = np.diff(f, 2)
    assert np.max(np.abs(second)) < 5e-3
    assert f[0] > f[-1]


def test_vert_stab_effective_ar_partials():
    p = _fin_ar(A_R_geo=2.0, bV_2r1=3.4, lambda_V=0.8, ZH_bV=-0.35,
                x_cV=0.65, SH_SV=0.9)
    assert_check_partials(p.check_partials(method='fd', step=1e-6,
                                           out_stream=None),
                          atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------------
# the remaining charts -- Figures 8.8, 8.10, 8.17 and 8.22
# ------------------------------------------------------------------------

def _scalar(cls, **ivc):
    """For the chart components that are scalar geometry, with no num_nodes."""
    p = om.Problem()
    p.model.add_subsystem('comp', cls(), promotes=['*'])
    p.setup()
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _end_plate(**ivc):
    """EndPlateFactorComp is scalar geometry and takes no num_nodes."""
    p = om.Problem()
    p.model.add_subsystem('comp', EndPlateFactorComp(), promotes=['*'])
    p.setup()
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _jacobian_ok(problem, tol=1e-5):
    """Compare analytic and finite-difference Jacobians entry by entry.

    Scaled, because check_partials reports an infinite relative error on
    the off-diagonal zeros of a sparse declaration.
    """
    data = problem.check_partials(method='fd', step=1e-6, out_stream=None)
    for key, value in data['comp'].items():
        a = np.atleast_1d(value['J_fwd']).ravel()
        b = np.atleast_1d(value['J_fd']).ravel()
        scale = max(np.abs(b).max(), 1.0)
        assert np.abs(a - b).max() / scale < tol, key


def test_end_plate_factor_is_a_straight_line():
    """Figure 8.8 p. 491: no end plate, no effect; slope 1.655."""
    p = _end_plate(h_bH=0.0, A_R_geo=4.5)
    assert_near_equal(p.get_val('factor')[0], 1.0, 1e-12)
    assert_near_equal(p.get_val('A_R_eff')[0], 4.5, 1e-12)

    half = _end_plate(h_bH=0.2, A_R_geo=4.5)
    full = _end_plate(h_bH=0.4, A_R_geo=4.5)
    assert_near_equal(full.get_val('factor')[0] - 1.0,
                      2.0 * (half.get_val('factor')[0] - 1.0), 1e-12)


def test_end_plate_factor_is_below_hoerner():
    """The classical result is 1 + 1.9 h/b; Figure 8.8 gives 1.655.

    13 % less, and the figure is credited to reference 8.3 rather than to
    Hoerner. The chart is what Prouty used.
    """
    p = _end_plate(h_bH=0.4, A_R_geo=4.5)
    assert_near_equal(p.get_val('factor')[0], 1.662, 1e-2)
    assert p.get_val('factor')[0] < 1.0 + 1.9 * 0.4


def test_figure_810_passes_through_its_measured_points():
    """The figure plots five test points; the digitised curve hits them."""
    p = _one(DynPressureIncreaseComp, nn=6,
             V_v1=np.array([0.3, 1.15, 1.9, 2.7, 3.7, 4.2]))
    got = p.get_val('dq_DL')
    for value, book, tol in ((got[1], 0.80, 3e-2), (got[2], 1.11, 2e-2),
                             (got[3], 0.35, 6e-2)):
        assert_near_equal(value, book, tol)
    assert got[0] == 0.0 and got[4] == 0.0 and got[5] == 0.0


def test_figure_810_wake_more_than_doubles_the_local_pressure():
    """At 20,000 lb over 2,827 sq ft the peak increment is 7.9 lb/sq ft."""
    DL = GW_EX / A_M_EX
    p = _one(DynPressureIncreaseComp, V_v1=np.full(1, 1.8),
             q=np.full(1, 6.6), qH_q_momentum=np.full(1, QH_Q_EX),
             DL=np.full(1, DL))
    assert_near_equal(p.get_val('dq_DL')[0] * DL, 7.9, 3e-2)
    assert p.get_val('q_H')[0] > 1.7 * 6.6


def test_span_efficiency_minimum_sits_near_quarter_taper():
    """Figure 8.17 p. 503: straight taper approaches elliptic near 0.4."""
    lam = np.arange(0.10, 1.20, 0.02)
    for A_R in (4.0, 12.0, 20.0):
        d = np.array([_scalar(SpanEfficiencyComp, lambda_taper=v,
                              A_R=A_R).get_val('delta')[0] for v in lam])
        assert_near_equal(lam[int(np.argmin(d))], 0.42, 2e-1)


def test_span_efficiency_penalty_grows_with_aspect_ratio():
    """0.5 % at A.R. 4 against 3.9 % at 20, both at the optimum taper."""
    low = _scalar(SpanEfficiencyComp, lambda_taper=0.4,
                  A_R=4.0).get_val('delta')[0]
    high = _scalar(SpanEfficiencyComp, lambda_taper=0.4,
                   A_R=20.0).get_val('delta')[0]
    assert_near_equal(low, 0.005, 4e-1)
    assert_near_equal(high, 0.039, 2e-1)
    assert high > 5.0 * low


def test_span_efficiency_clamps_below_the_printed_curves():
    """The fin is A.R. 3.2 effective; Figure 8.17 starts at 4."""
    below = _scalar(SpanEfficiencyComp, lambda_taper=0.21, A_R=3.2)
    edge = _scalar(SpanEfficiencyComp, lambda_taper=0.21, A_R=4.0)
    assert_near_equal(below.get_val('delta')[0],
                      edge.get_val('delta')[0], 1e-12)


def test_figure_822_reproduces_the_reading_of_page_509():
    """K_int = 0.4 for the example helicopter.

    b_V/2R_T = 0.59 puts it on the leftmost printed curve, which gives 0.4
    at a separation ratio of 0.192, or y_V = 2.0 ft. Prouty does not print
    y_V, so this is a consistency check rather than an anchor.
    """
    p = _scalar(InterferenceFactorComp, y_V=1.99, R_T=R_T_EX, b_V=B_V_EX)
    assert_near_equal(p.get_val('K_int')[0], 0.4, 2e-2)


def test_figure_822_trends_have_sense():
    """Interference falls with separation and rises with fin span."""
    near = _scalar(InterferenceFactorComp, y_V=1.0, R_T=R_T_EX,
                   b_V=B_V_EX).get_val('K_int')[0]
    far = _scalar(InterferenceFactorComp, y_V=2.4, R_T=R_T_EX,
                  b_V=B_V_EX).get_val('K_int')[0]
    assert near > far

    big = _scalar(InterferenceFactorComp, y_V=2.0, R_T=R_T_EX,
                  b_V=13.0).get_val('K_int')[0]
    assert big > _scalar(InterferenceFactorComp, y_V=2.0, R_T=R_T_EX,
                         b_V=B_V_EX).get_val('K_int')[0]


def test_figure_822_feeds_the_fin_drag():
    """The factor is three quarters of D_V, so it is not a detail, C8-6."""
    kw = dict(T_T=np.full(1, T_T_EX), Y_V=np.full(1, LIFT_V_EX),
              q=np.full(1, Q_EX), R_T=R_T_EX, b_V=B_V_EX)
    close = _scalar(InterferenceFactorComp, y_V=0.93, R_T=R_T_EX,
                    b_V=B_V_EX).get_val('K_int')[0]
    nominal = _scalar(InterferenceFactorComp, y_V=1.99, R_T=R_T_EX,
                      b_V=B_V_EX).get_val('K_int')[0]
    drag = [_one(VertStabInterferenceDragComp,
                 K_int=k, **kw).get_val('dD_int')[0] for k in (nominal, close)]
    assert_near_equal(close, 0.50, 5e-2)
    assert drag[1] / drag[0] > 1.2


def test_remaining_chart_partials():
    _jacobian_ok(_end_plate(h_bH=0.35, A_R_geo=4.5))
    _jacobian_ok(_one(DynPressureIncreaseComp, nn=3,
                      V_v1=np.array([1.0, 1.8, 3.0]),
                      q=np.linspace(5.0, 45.0, 3),
                      qH_q_momentum=np.full(3, 0.6),
                      DL=np.full(3, 7.1)))
    _jacobian_ok(_scalar(SpanEfficiencyComp, lambda_taper=0.5, A_R=9.0))
    _jacobian_ok(_scalar(InterferenceFactorComp, y_V=2.2, R_T=6.5, b_V=9.8))


# ------------------------------------------------------------------------
# control_positions_comp -- Appendix A Figure A.5 p. 682
# ------------------------------------------------------------------------

def _controls(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', ControlPositionsComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_rigging_matches_table_88_stick_positions():
    """Table 8.8 p. 530: B_1 = 8.5, 8.8, 9.4 gives 38, 37, 35 %.

    The fitted line is a constant 1.1 points high with the right slope,
    which is digitising precision on a chart drawn at 2.2 pixels per per
    cent.
    """
    p = _controls(nn=3, B_1=np.array([8.5, 8.8, 9.4]))
    got = p.get_val('pct_B1')
    assert_near_equal(got, np.array([39.1, 38.1, 36.2]), 5e-3)

    printed = np.array([38.0, 37.0, 35.0])
    assert np.abs(got - printed).max() < 1.3
    assert np.abs(np.diff(got) - np.diff(printed)).max() < 0.15


def test_rigging_settles_C8_11():
    """C8-11. Figure 8.31 p. 538 gives 44 % lateral stick at zero sideslip.

    Chapter 3's combination is A_1 - b1s_M = -2.3 deg and the lateral trim
    gives b1s_M = -0.78 deg. The two readings of p. 535 put the stick nine
    points apart, and only one of them lands on the figure.
    """
    ours = A1_B1S_EX + np.deg2rad(-0.78)              # p. 169 convention
    printed = A1_B1S_EX - np.deg2rad(-0.78)           # p. 535 as written

    p = _controls(nn=2, A_1=np.rad2deg([ours, printed]))
    got = p.get_val('pct_A1')

    assert_near_equal(got[0], 42.3, 2e-2)
    assert_near_equal(got[1], 51.4, 2e-2)
    assert abs(got[0] - 44.0) < 2.0
    assert abs(got[1] - 44.0) > 7.0


def test_rigging_directional_position_of_figure_831():
    """55 % pedal at zero sideslip gives 6.0 deg of tail rotor pitch."""
    p = _controls(theta_75_T=np.full(1, 6.02))
    assert_near_equal(p.get_val('pct_pedal')[0], 55.0, 2e-2)


def test_rigging_senses_are_not_all_the_same():
    """More B_1 is more forward stick; more A_1 is more right stick;
    more tail rotor pitch is more left pedal.
    """
    a = _controls(B_1=np.full(1, 5.0), A_1=np.full(1, -5.0),
                  theta_75_T=np.full(1, 5.0), theta_75_M=np.full(1, 10.0))
    b = _controls(B_1=np.full(1, 10.0), A_1=np.full(1, 0.0),
                  theta_75_T=np.full(1, 10.0), theta_75_M=np.full(1, 15.0))
    assert b.get_val('pct_B1')[0] < a.get_val('pct_B1')[0]
    assert b.get_val('pct_A1')[0] > a.get_val('pct_A1')[0]
    assert b.get_val('pct_pedal')[0] < a.get_val('pct_pedal')[0]
    assert b.get_val('pct_coll')[0] > a.get_val('pct_coll')[0]


def test_rigging_gives_the_three_inch_criterion_of_page_530():
    """p. 530 sets the minimum control shift, climb to autorotation, at 3 in.

    Table 8.8 spans B_1 = 8.5 to 9.4, which the 10 inch longitudinal travel
    turns into 0.3 in. The example helicopter is nowhere near the
    specification, and p. 530 calls that a benefit of its small stabiliser.
    """
    p = _controls(nn=2, B_1=np.array([8.5, 9.4]))
    shift = abs(np.diff(p.get_val('in_B1'))[0])
    assert_near_equal(shift, 0.30, 5e-2)
    assert shift < 3.0


def test_rigging_full_travel_spans_the_printed_range():
    """Figure A.5: 0 % at B_1 = 20.4 deg and 100 % at -10.0 deg."""
    p = _controls(nn=2, B_1=np.array([20.4, -10.05]))
    assert_near_equal(p.get_val('pct_B1'), np.array([0.0, 100.0]), 5e-2)


def test_control_positions_partials():
    p = _controls(nn=3, B_1=np.linspace(5.0, 12.0, 3),
                  A_1=np.linspace(-5.0, 2.0, 3),
                  theta_75_T=np.linspace(4.0, 20.0, 3),
                  theta_75_M=np.linspace(10.0, 18.0, 3))
    assert_check_partials(p.check_partials(method='cs', out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------------
# rigging wired into the two trim groups
# ------------------------------------------------------------------------

def test_longitudinal_group_gives_a_stick_position():
    """From gross weight and geometry to per cent of stick, in one solve."""
    p = _long_trim(linearized=True)
    p.set_val('B1_a1s', np.full(1, B1_A1S_EX))
    p.run_model()

    assert_near_equal(p.get_val('B_1', units='deg')[0], 8.9, 4e-2)
    assert_near_equal(p.get_val('pct_B1')[0], 37.8, 2e-2)
    assert_near_equal(p.get_val('in_B1')[0], 3.78, 2e-2)

    # Table 8.8 p. 530 puts level flight at 37 %
    assert abs(p.get_val('pct_B1')[0] - 37.0) < 1.5


def test_lateral_group_reproduces_the_lateral_stick_of_figure_831():
    """44 % at zero sideslip, and this is what settles C8-11.

    The chain is complete here: gross weight and geometry give b1s_M, the
    Chapter 3 combination gives A_1, and Figure A.5 gives the stick.
    """
    p = _lat_trim(mode='zero_sideslip', linearized=True)
    p.set_val('A1_b1s', np.full(1, A1_B1S_EX))
    p.set_val('B_1', np.full(1, B_1_EX))
    p.run_model()

    assert_near_equal(p.get_val('A_1', units='deg')[0], -3.08, 3e-2)
    assert_near_equal(p.get_val('pct_A1')[0], 42.3, 2e-2)
    assert abs(p.get_val('pct_A1')[0] - 44.0) < 2.0


def test_lateral_stick_tilts_with_sideslip_as_figure_831_shows():
    """Right sideslip moves the lateral stick right, through B_1 sin(beta).

    Figure 8.31 draws the lateral control line rising from about 41 % at
    5 deg left to 47 % at 5 deg right.
    """
    left = _lat_trim(mode='given_sideslip', linearized=True)
    right = _lat_trim(mode='given_sideslip', linearized=True)
    for p, beta in ((left, -3.0), (right, 3.0)):
        p.set_val('A1_b1s', np.full(1, A1_B1S_EX))
        p.set_val('B_1', np.full(1, B_1_EX))
        p.set_val('beta', np.full(1, np.deg2rad(beta)))
        p.run_model()

    rise = right.get_val('pct_A1')[0] - left.get_val('pct_A1')[0]
    assert rise > 0.0
    assert_near_equal(rise, 2.0 * 8.9 * np.sin(np.deg2rad(3.0)) / 0.17144
                      + (right.get_val('b1s_M', units='deg')[0]
                         - left.get_val('b1s_M', units='deg')[0]) / 0.17144,
                      5e-2)


def test_pedal_position_is_not_an_independent_check():
    """Figure 8.31 gives 55 % directional at zero sideslip.

    The component reproduces it, but only because theta_75_T was read back
    off the same chart: Chapter 8 stops at alpha_TPP_T and sends the tail
    rotor collective to the Chapter 3 method, which is not wired here. This
    is a round trip, not a validation, and it is asserted as such.
    """
    p = _lat_trim(mode='zero_sideslip', linearized=True)
    p.set_val('A1_b1s', np.full(1, A1_B1S_EX))
    p.set_val('B_1', np.full(1, B_1_EX))
    p.set_val('theta_75_T', np.full(1, 6.02), units='deg')
    p.run_model()
    assert_near_equal(p.get_val('pct_pedal')[0], 55.0, 2e-2)

    assert 'theta_75_T' not in p.model.t.cyclic._var_rel_names['output']


def test_controls_stay_outside_both_newton_loops():
    """Changing a rigging constant must cost no iterations."""
    for build, name, value in ((lambda: _long_trim(linearized=True),
                                'long_pct_0', 70.0),
                               (lambda: _lat_trim(mode='zero_sideslip',
                                                  linearized=True),
                                'lat_A1_0', -11.0)):
        p = build()
        before = p.get_val('b1s_M' if 'lat_A1_0' == name else 'T_M')[0]
        p.set_val(name, value)
        p.run_model()
        assert p.model.t.nonlinear_solver._iter_count == 0
        assert_near_equal(
            p.get_val('b1s_M' if 'lat_A1_0' == name else 'T_M')[0],
            before, 1e-12)


# ------------------------------------------------------------------------
# charts wired into TrimElementsGroup
# ------------------------------------------------------------------------

# geometry the charts are drawn against, p. 504 and Tables 8.2 and 8.3
CHART_GEO = dict(A_R_H_geo=A_H_AR_EX, h_bH=0.0, A_R_V_geo=1.8,
                 bV_2r1=B_V_EX / 1.5, lambda_V=0.21, ZH_bV=0.0,
                 x_cV=2.5 / 4.25, SH_SV=A_H_EX / A_V_EX, lambda_H=0.6,
                 y_V=1.99)
CHART_SWEEP = dict(sweep_H=np.deg2rad(13.0), sweep_V=np.deg2rad(27.0))
CHART_QUANTITIES = ('A_R_H', 'a_H', 'delta_H', 'qH_q',
                    'A_R_V', 'a_V', 'delta_V', 'K_int')


def _elements_charts(nn=1, **opts):
    p = _elements(nn=nn, charts=True, **opts)
    for k, v in {**CHART_GEO, **CHART_SWEEP}.items():
        p.set_val(k, v)
    p.set_val('qH_q_momentum', QH_Q_EX * np.ones(nn))
    p.run_model()
    return p


def test_charts_off_leaves_the_readings_alone():
    """The anchors of this package were set on Prouty's own readings.

    With charts off the eight chart quantities are free inputs and the group
    returns exactly what it was given; with charts on they are computed and
    whatever was set is ignored.
    """
    off = _elements()
    for name, given in (('a_V', A_V_SLOPE_EX), ('A_R_V', A_V_AR_EX),
                        ('delta_V', DELTA_V_EX), ('K_int', K_INT_EX)):
        assert_near_equal(off.get_val(name)[0], given, 1e-12)

    on = _elements_charts()
    on.set_val('a_V', 99.0)
    on.run_model()
    assert on.get_val('a_V')[0] < 5.0


def test_charts_on_reproduces_proutys_readings_to_a_few_per_cent():
    """Each digitised value against the number Prouty read off the same
    figure. They err the same way, so the fin chain compounds them.
    """
    p = _elements_charts()
    for name, read, tol in (('A_R_H', 4.5, 1e-9), ('a_H', 4.0, 1e-2),
                            ('qH_q', 0.6, 1e-9), ('A_R_V', 3.2, 5e-2),
                            ('a_V', 3.0, 1.3e-1), ('K_int', 0.4, 1e-2)):
        assert_near_equal(p.get_val(name)[0], read, tol), name


def test_figure_810_is_inert_at_115_knots():
    """V/v1_hover = 2 sqrt(q/D.L.) = 4.97, past the 3.7 where it returns
    to zero, so wiring Figure 8.10 leaves the anchors alone.
    """
    p = _elements_charts()
    assert_near_equal(p.get_val('V_v1')[0], 4.97, 1e-2)
    assert_near_equal(p.get_val('dq_DL')[0], 0.0, 1e-12)
    assert_near_equal(p.get_val('qH_q')[0], QH_Q_EX, 1e-12)


def test_charts_move_the_fin_and_leave_the_stabiliser_alone():
    """The horizontal stabiliser chain is insensitive; the fin is not.

    Figure 8.6 gives a_H = 4.02 against a read 4.0, so L_H barely moves.
    The fin compounds Figure 8.19 with Figure 8.6 and gains 12 %.
    """
    read, charts = _elements(), _elements_charts()
    assert abs(charts.get_val('L_H')[0] / read.get_val('L_H')[0] - 1) < 0.01
    assert charts.get_val('L_V')[0] / read.get_val('L_V')[0] > 1.10


def test_charts_leave_the_longitudinal_trim_where_it_was():
    """The stabiliser is what trims pitch, and the charts hardly touch it."""
    read, charts = _long_trim(linearized=True), _long_trim(linearized=True,
                                                           charts=True)
    for p in (read, charts):
        p.set_val('B1_a1s', np.full(1, B1_A1S_EX))
    for k, v in {**CHART_GEO, **CHART_SWEEP}.items():
        charts.set_val(k, v)
    charts.set_val('qH_q_momentum', np.full(1, QH_Q_EX))
    read.run_model()
    charts.run_model()

    assert abs(charts.get_val('T_M')[0] - read.get_val('T_M')[0]) < 5.0
    assert abs(charts.get_val('B_1', units='deg')[0]
               - read.get_val('B_1', units='deg')[0]) < 0.05
    assert_near_equal(charts.get_val('pct_B1')[0], 37.7, 2e-2)


def test_charts_cost_the_lateral_trim_six_per_cent_of_tail_rotor_thrust():
    """A 12 % stronger fin carries more antitorque, so T_T falls.

    Everything the pilot sees stays put: b1s_M, Phi and the lateral stick
    all move by less than the width of a line on Figure 8.31. The tail
    rotor thrust is the one observable that shifts, and it shifts away from
    the printed 661 lb, which is the price of reading the charts.
    """
    read = _lat_trim(mode='zero_sideslip', linearized=True)
    charts = _lat_trim(mode='zero_sideslip', linearized=True, charts=True)
    for p in (read, charts):
        p.set_val('A1_b1s', np.full(1, A1_B1S_EX))
        p.set_val('B_1', np.full(1, B_1_EX))
    for k, v in {**CHART_GEO, **CHART_SWEEP}.items():
        charts.set_val(k, v)
    charts.set_val('qH_q_momentum', np.full(1, QH_Q_EX))
    read.run_model()
    charts.run_model()

    assert charts.get_val('T_T')[0] < read.get_val('T_T')[0]
    assert abs(charts.get_val('T_T')[0] / read.get_val('T_T')[0] - 1) > 0.04

    for name in ('b1s_M', 'Phi'):
        gap = abs(charts.get_val(name, units='deg')[0]
                  - read.get_val(name, units='deg')[0])
        assert gap < 0.05, name
    assert abs(charts.get_val('pct_A1')[0] - read.get_val('pct_A1')[0]) < 0.3
