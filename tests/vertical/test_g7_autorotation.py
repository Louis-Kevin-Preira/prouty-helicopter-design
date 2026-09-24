"""Chapter 2, G7 -- VerticalAutorotationGroup, pp. 109-115, Figures 2.13-2.14."""

import warnings

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import (assert_check_partials, assert_check_totals,
                                         assert_near_equal)

from prouty.vertical import (AutorotationDescentComp, ClimbCollectiveComp, ClimbPowerGroup,
                             FlowStatesGroup, ThrustDampingGroup, VerticalAutorotationGroup,
                             VortexRingBoundariesComp, TailRotorVortexRingGroup,
                             TailRotorDriveTorqueComp)

# Example helicopter, Appendix A; Chapter 1 lift slope 5.73 (reproduces Figure 2.14)
GW, A_M, SIGMA, RHO_SL, A_BLADE = 20000.0, 2827.4, 0.085, 0.002377, 5.73

# Figure 2.14 p. 114, digitized: tip speed ft/s, rate of descent ft/min, collective deg
FIG_2_14 = np.array([[450, 4550, 16.33], [490, 4533, 15.00], [530, 4524, 13.78],
                     [550, 4524, 13.25], [590, 4529, 12.28], [630, 4547, 11.48],
                     [650, 4559, 11.14], [690, 4592, 10.55]])


def _run(V_tip=650.0, T=GW, dP=0.0, drag='airfoil', cd=None, nn=1, **opts):
    p = om.Problem()
    p.model.add_subsystem('g7', VerticalAutorotationGroup(num_nodes=nn, drag=drag, **opts),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('T', np.full(nn, T), units='lbf')
    p.set_val('rho', RHO_SL)
    p.set_val('A', A_M, units='ft**2')
    p.set_val('sigma', SIGMA)
    p.set_val('a', A_BLADE)
    p.set_val('V_tip', V_tip, units='ft/s')
    p.set_val('theta_1', -10.0, units='deg')
    p.set_val('dP_auto', np.broadcast_to(dP, (nn,)).astype(float), units='hp')
    if cd is not None:
        p.set_val('cd_bar', np.full(nn, cd))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p.run_model()
    return p


def _descent(y, v_hov=1.0, theta_1_deg=-10.0):
    y = np.atleast_1d(np.asarray(y, dtype=float))
    p = om.Problem()
    p.model.add_subsystem('d', AutorotationDescentComp(num_nodes=y.size), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('VD_minus_v1_bar', y)
    p.set_val('v_hov', np.full(y.size, v_hov), units='ft/s')
    p.set_val('theta_1', theta_1_deg, units='deg')
    p.run_model()
    return p


# ------------------------------------------------------------------------
# book anchors
# ------------------------------------------------------------------------

def test_figure_2_13_reading_p112():
    """0.22 at -10 deg --> V_D_bar = 1.97; with v_1hov = 40.4 ft/s, 4,780 ft/min."""
    p = _descent(0.22, v_hov=40.4)
    assert_near_equal(p.get_val('V_D_bar_auto')[0], 1.97, 0.005)
    assert_near_equal(p.get_val('V_D_auto', units='ft/min')[0], 4780.0, 0.005)


def test_parameter_with_book_drag_p112():
    """The p. 112 algebra: 0.22 is the parameter for c_d = 0.00865 at 20,000 lb."""
    p = _run(drag='input', cd=0.00865)
    assert_near_equal(p.get_val('VD_minus_v1_bar')[0], 0.22, 0.005)


def test_parameter_with_chapter6_drag():
    """C2-8: the NACA 0012 model gives c_d = 0.0104 at cl_bar = 0.50 and a parameter of 0.264."""
    p = _run()
    assert_near_equal(p.get_val('cl_bar')[0], 0.497, 0.002)
    assert_near_equal(p.get_val('cd_bar')[0], 0.0104, 0.01)
    assert_near_equal(p.get_val('VD_minus_v1_bar')[0], 0.264, 0.01)


def test_extra_power_p115():
    """40 hp raises V_D_bar - v1_bar by 11 %; the rate of descent by 0.4 %, not 1.5 % (C2-9)."""
    p = _run(dP=[0.0, 40.0], nn=2)
    y, vd = p.get_val('VD_minus_v1_bar'), p.get_val('V_D_auto')
    assert_near_equal(y[1] / y[0] - 1.0, 0.11, 0.03)
    assert 0.002 < vd[1] / vd[0] - 1.0 < 0.008


def test_parachute_p115():
    """C_D = 1.2 --> 4,260 ft/min; C_D = 1.0 --> exactly 2 v_1hov."""
    p = _run()
    assert_near_equal(p.get_val('RD_parachute', units='ft/min')[0], 4260.0, 0.01)
    p.set_val('C_D_chute', 1.0)
    p.run_model()
    assert_near_equal(p.get_val('RD_parachute')[0], 2.0 * p.get_val('v_hov')[0], 1e-12)


def test_figure_2_14_rate_of_descent_and_collective():
    """Within 1.5 % on R/D and 0.3 deg on collective over 490-690 ft/s (C2-10)."""
    rows = FIG_2_14[1:]
    for V_tip, rd, theta in rows:
        p = _run(V_tip=V_tip)
        assert_near_equal(p.get_val('V_D_auto', units='ft/min')[0], rd, 0.015)
        assert abs(p.get_val('theta_0_auto', units='deg')[0] - theta) < 0.3


def test_figure_2_14_uses_20000_lb_at_sea_level():
    """C2-3 (closed): at 650 ft/s the figure reads 4,559 ft/min = 1.97 x 38.6 ft/s, not 40.4."""
    p = _run(drag='input', cd=0.00865)
    assert_near_equal(p.get_val('V_D_auto', units='ft/min')[0], 4559.0, 0.005)


def test_rule_of_thumb_twice_hover_induced_velocity():
    V_D_bar = np.array([_run(V_tip=v).get_val('V_D_bar_auto')[0] for v in (500.0, 650.0, 700.0)])
    assert np.all(np.abs(V_D_bar - 2.0) < 0.06)


# ------------------------------------------------------------------------
# internal consistency
# ------------------------------------------------------------------------

def test_descent_inverts_g0():
    """V_D_bar - v1_bar recomputed with G0 at the solution returns the input."""
    from prouty.vertical import axial_inflow
    y = np.array([0.0, 0.22, 0.6, 1.5, 3.0])
    for th in (0.0, -6.0, -12.0):
        x = _descent(y, theta_1_deg=th).get_val('V_D_bar_auto')
        assert_near_equal(x - axial_inflow(x, np.radians(th))[0], y, 1e-10)


def test_above_the_hook_is_held_with_a_warning():
    """theta_1 = 0: y(1.75) = -0.08, so -0.5 lies on the upper branch of Figure 2.13."""
    with pytest.warns(UserWarning, match='lower branch'):
        p = _descent(-0.5, theta_1_deg=0.0)
    assert_near_equal(p.get_val('V_D_bar_auto')[0], 1.75, 1e-14)


def test_untwisted_rotor_descends_slower():
    """p. 113: an untwisted rotor needs a lower rate of descent."""
    x0 = _descent(0.25, theta_1_deg=0.0).get_val('V_D_bar_auto')[0]
    x12 = _descent(0.25, theta_1_deg=-12.0).get_val('V_D_bar_auto')[0]
    assert x0 < x12


def test_stall_warning_at_low_tip_speed():
    p = _run(V_tip=440.0)
    with pytest.warns(UserWarning, match='alpha_L'):
        p.run_model()


def test_promoted_defaults_are_consistent_across_the_chapter():
    p = om.Problem()
    m = p.model
    m.add_subsystem('g0', FlowStatesGroup(num_nodes=2), promotes=['*'])
    m.add_subsystem('g2', ClimbPowerGroup(num_nodes=2, tail_rotor='hover_power', inflow='external'),
                    promotes=['*'])
    m.add_subsystem('g3', ClimbCollectiveComp(num_nodes=2), promotes=['*'])
    m.add_subsystem('g4', ThrustDampingGroup(num_nodes=2), promotes=['*'])
    m.add_subsystem('g7', VerticalAutorotationGroup(num_nodes=2, inflow='external'),
                    promotes=['*'])
    m.add_subsystem('g5', VortexRingBoundariesComp(num_nodes=2), promotes=['*'])
    m.add_subsystem('g6', TailRotorVortexRingGroup(num_nodes=1), promotes=['*'])
    m.add_subsystem('g8', TailRotorDriveTorqueComp(num_nodes=2), promotes=['*'])
    p.setup()
    p.final_setup()


# ------------------------------------------------------------------------
# derivatives
# ------------------------------------------------------------------------

@pytest.mark.parametrize('drag', ['airfoil', 'input'])
def test_partials(drag):
    p = _run(V_tip=600.0, dP=[0.0, 40.0], drag=drag, cd=0.01 if drag == 'input' else None, nn=2)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_partials_descent_across_branches():
    p = _descent([-0.4, 0.1, 0.22, 1.2, 2.5], v_hov=38.6, theta_1_deg=-7.0)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_totals():
    p = _run(V_tip=620.0, dP=[0.0, 30.0], nn=2)
    data = p.check_totals(of=['V_D_auto', 'theta_0_auto'],
                          wrt=['T', 'V_tip', 'sigma', 'theta_1', 'dP_auto'],
                          method='cs', out_stream=None)
    assert_check_totals(data, atol=1e-7, rtol=1e-7)
