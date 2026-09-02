"""Regression tests for prouty.hover against the values printed in the book.

The reference case throughout is the example helicopter main rotor of p. 669
run through the sample calculation of Figure 1.45, p. 76: R = 30 ft, c = 2 ft,
b = 4, V_tip = 650 ft/s, cutout 0.15 R, theta_1 = -10 deg, theta_0 = 17.5 deg,
NACA 0012, sea level standard day.

Where the model departs from the book it is on purpose and the tolerance says
so; see docs/validation_forward_flight.md for the C_T discrepancy at step 11.
"""

import numpy as np
import openmdao.api as om
import pytest

from prouty.hover import (HoverRotorGroup, RotorPreprocessGroup,
                          IntegrationWeightsComp, TipLossComp,
                          WakeRotationComp, WakeContractionComp, InflowGroup,
                          DimensionalPerfComp, ChordDistComp, BladeGridComp)

MAIN_ROTOR = dict(R=30.0, c_root=2.0, c_tip=2.0, r_1=30.0, r_cutout=4.5,
                  V_tip=650.0, b=4.0, theta_1=-10.0, altitude=0.0)
TAIL_ROTOR = dict(R=6.5, c_root=1.0, c_tip=1.0, r_1=6.5, r_cutout=0.975,
                  V_tip=650.0, b=3.0, theta_1=-5.0, altitude=0.0)


def build(group=HoverRotorGroup, complex_step=True, **opts):
    p = om.Problem()
    p.model.add_subsystem('rotor', group(**opts), promotes=['*'])
    p.setup(force_alloc_complex=complex_step)
    return p


def run_main(theta_0=17.5, ne=10, **overrides):
    p = build(num_elements=ne)
    for k, v in {**MAIN_ROTOR, **overrides}.items():
        p.set_val(k, v)
    p.set_val('theta_0', theta_0)
    p.run_model()
    return p


@pytest.fixture(scope='module')
def sample():
    """The Figure 1.45 sample calculation, run once."""
    return run_main()


# --- book anchors, steps 1 to 21 -----------------------------------------

def test_solidity_main_and_tail_rotor():
    """sigma = 0.085 p. 76 and 0.146 p. 670."""
    assert run_main().get_val('sigma')[0] == pytest.approx(0.085, abs=5e-4)

    p = build()
    for k, v in TAIL_ROTOR.items():
        p.set_val(k, v)
    p.set_val('theta_0', 12.0)
    p.run_model()
    assert p.get_val('sigma')[0] == pytest.approx(0.146, abs=1e-3)


def test_tip_mach_number(sample):
    """Tip Mach number 0.58, p. 75."""
    assert sample.get_val('M')[-1] == pytest.approx(0.58, abs=5e-3)


def test_pitch_at_root_and_tip(sample):
    """theta_0 = 17.5 with theta_1 = -10 gives 16.0 at 0.15 R and 7.5 at the tip."""
    theta = sample.get_val('theta')
    assert theta[0] == pytest.approx(16.0, abs=1e-9)
    assert theta[-1] == pytest.approx(7.5, abs=1e-9)


def test_thrust_coefficient_without_tip_loss(sample):
    """C_T no tip loss = 0.00741, Figure 1.45. Trapezoid is 0.3% low."""
    assert sample.get_val('CT_no_tip_loss')[0] == pytest.approx(0.00741, rel=5e-3)


def test_tip_loss_factor(sample):
    """B = 0.98, Figure 1.45, from the effective radius equations of p. 71."""
    assert sample.get_val('B')[0] == pytest.approx(0.98, abs=1e-3)


def test_profile_torque_coefficient(sample):
    """C_Q0 = 0.000111, Figure 1.45. Not affected by the step 11 gap."""
    assert sample.get_val('CQ0')[0] == pytest.approx(1.11e-4, rel=1e-2)


def test_swirl_ratio_from_figure_1_29(sample):
    """Swirl over thrust induced power = 0.017 at C_T = 0.0072, p. 52."""
    assert sample.get_val('swirl_ratio')[0] == pytest.approx(0.017, abs=1e-3)


def test_power_factor_from_figure_1_34(sample):
    """Measured over calculated power = 1.05 near 0.61, p. 58."""
    assert sample.get_val('power_factor')[0] == pytest.approx(1.05, abs=1e-2)


def test_thrust_and_power(sample):
    """T = 20,400 lb and 1,990 hp, Figure 1.45; the model runs 2.3% low."""
    assert sample.get_val('T')[0] == pytest.approx(20400.0, rel=3e-2)
    assert sample.get_val('power_hp')[0] == pytest.approx(1990.0, rel=3e-2)


def test_figure_of_merit_is_realistic(sample):
    """Hover F.M. of a transport rotor sits near 0.7, p. 22."""
    assert 0.65 < sample.get_val('FM')[0] < 0.78


# --- internal consistency -------------------------------------------------

def test_chord_equation_matches_the_book_form():
    """c_e of p. 17, written with the removable singularity cleared."""
    p = om.Problem()
    p.model.add_subsystem('grid', BladeGridComp(num_elements=10), promotes=['*'])
    p.model.add_subsystem('chord', ChordDistComp(num_nodes=11), promotes=['*'])
    p.setup()
    p.set_val('c_root_R', 0.08)
    p.set_val('c_tip_R', 0.04)
    for x_t in (0.0, 0.3, 0.6, 0.999):
        p.set_val('x_t', x_t)
        p.run_model()
        book = (0.04 + (0.08 - 0.04) * x_t ** 3
                - ((0.08 - 0.04) / (1.0 - x_t))
                * (-0.25 + x_t ** 3 - 0.75 * x_t ** 4))
        assert p.get_val('sigma_T')[0] * np.pi / 4.0 == pytest.approx(book, rel=1e-9)


def test_ideal_twist_gives_uniform_inflow():
    """theta = theta_t / (r/R) makes v1 constant, p. 19.

    The property is exact only for a constant lift curve slope: with
    theta = theta_t/x and P proportional to a/x, the radicand of step 5 becomes
    32 pi theta_t / (a b c/R), free of x, so v1_Or goes as 1/x and v1 is
    uniform. Once the airfoil model is coupled, a rises with Mach number toward
    the tip and the property degrades to about 6% spread, so the analytic check
    belongs on InflowGroup with a imposed, not on the coupled rotor.
    """
    nn = 11
    p = om.Problem()
    p.model.add_subsystem('pre', RotorPreprocessGroup(num_elements=nn - 1),
                          promotes=['*'])
    p.model.add_subsystem('inflow', InflowGroup(num_nodes=nn, twist_law='ideal'),
                          promotes=['*'])
    p.setup()
    for k, v in MAIN_ROTOR.items():
        p.set_val(k, v)
    p.set_val('a', np.full(nn, 0.1))
    p.set_val('theta_0', 8.0)
    p.run_model()
    lam = p.get_val('v1_Or') * p.get_val('r_R')
    assert np.ptp(lam) < 1e-12


def test_ideal_twist_inflow_spread_stays_small_when_coupled():
    """With a = a(M) the uniform inflow property is approximate, not exact."""
    p = build(twist_law='ideal')
    for k, v in MAIN_ROTOR.items():
        p.set_val(k, v)
    p.set_val('theta_0', 8.0)
    p.run_model()
    lam = p.get_val('v1_Or') * p.get_val('r_R')
    assert np.ptp(lam) / lam.mean() < 0.10


def test_quadrature_is_exact_on_a_linear_integrand():
    """Trapezoid weights, including the cut cell, integrate x exactly."""
    nn = 11
    x = np.linspace(0.15, 1.0, nn)
    p = om.Problem()
    p.model.add_subsystem('ivc', om.IndepVarComp('r_R', val=x), promotes=['*'])
    p.model.add_subsystem('w', IntegrationWeightsComp(num_nodes=nn),
                          promotes=['*'])
    p.setup()
    for x_max in (0.4, 0.7449, 0.745, 0.9812, 1.0):
        p.set_val('x_max', x_max)
        p.run_model()
        w = p.get_val('w')
        assert w.sum() == pytest.approx(x_max - x[0], abs=1e-12)
        assert w @ x == pytest.approx((x_max ** 2 - x[0] ** 2) / 2.0, abs=1e-12)


def test_tip_loss_reproduces_the_two_bladed_assumption():
    """The 0.06 constant of p. 71 gives 0.97 R for b = 2, as the note states."""
    p = om.Problem()
    p.model.add_subsystem('tl', TipLossComp(), promotes=['*'])
    p.setup()
    p.set_val('b', 2.0)
    for ct in (0.002, 0.004, 0.0059):
        p.set_val('CT_no_tip_loss', ct)
        p.run_model()
        assert p.get_val('B')[0] == pytest.approx(0.97, abs=1e-9)


def test_tip_loss_models_disagree_as_expected():
    """The general model of p. 71 gives 0.97, the effective radius one 0.98."""
    out = {}
    for model in ('general', 'effective_radius'):
        p = om.Problem()
        p.model.add_subsystem('tl', TipLossComp(model=model), promotes=['*'])
        p.setup()
        p.set_val('CT_no_tip_loss', 0.00741)
        p.run_model()
        out[model] = p.get_val('B')[0]
    assert out['general'] == pytest.approx(0.970, abs=1e-3)
    assert out['effective_radius'] == pytest.approx(0.979, abs=1e-3)


def test_tip_loss_is_monotone_through_the_break():
    """The smoothed branch point at C_T = 0.006 must not create a bump."""
    p = om.Problem()
    p.model.add_subsystem('tl', TipLossComp(), promotes=['*'])
    p.setup()
    B = []
    for ct in np.linspace(0.004, 0.012, 81):
        p.set_val('CT_no_tip_loss', ct)
        p.run_model()
        B.append(p.get_val('B')[0])
    assert np.all(np.diff(B) <= 1e-12)


def test_wake_charts_stay_monotone_over_their_range():
    """Both digitised splines must be monotone, Figures 1.29 and 1.34."""
    p = om.Problem()
    p.model.add_subsystem('wr', WakeRotationComp(), promotes=['*'])
    p.setup()
    vals = []
    for ct in np.linspace(0.0, 0.05, 101):
        p.set_val('CT', ct)
        p.run_model()
        vals.append(p.get_val('swirl_ratio')[0])
    assert np.all(np.diff(vals) > 0)

    q = om.Problem()
    q.model.add_subsystem('wc', WakeContractionComp(), promotes=['*'])
    q.setup()
    q.set_val('CT_sigma', 1.0)
    vals = []
    for x in np.linspace(0.10, 1.20, 111):
        q.set_val('DL', x)
        q.run_model()
        vals.append(q.get_val('power_factor')[0])
    assert np.all(np.diff(vals) > 0)


def test_dimensional_step_21(sample):
    """T and hp follow from the coefficients by the step 21 formulas."""
    rho, A, v = (sample.get_val(n)[0] for n in ('rho', 'A', 'V_tip'))
    ct, cq = sample.get_val('CT')[0], sample.get_val('CQ')[0]
    assert sample.get_val('T')[0] == pytest.approx(rho * A * v ** 2 * ct)
    assert sample.get_val('power_hp')[0] == pytest.approx(
        rho * A * v ** 3 * cq / 550.0)


def test_grid_convergence():
    """C_Q must be settled to better than 0.5% over the p. 69 element range."""
    cq = [run_main(ne=ne).get_val('CQ')[0] for ne in (8, 10, 12, 15)]
    assert (max(cq) - min(cq)) / cq[-1] < 5e-3


# --- solvers and derivatives ---------------------------------------------

def test_inflow_airfoil_cycle_converges_from_a_wrong_slope():
    """The cycle is weak: a is a function of Mach only."""
    ref = run_main().get_val('alpha')
    p = run_main()
    p.set_val('a', np.full(11, 0.5))
    p.run_model()
    np.testing.assert_allclose(p.get_val('alpha'), ref, atol=1e-12)
    assert p.model.rotor.blade_element.nonlinear_solver._iter_count <= 5


def test_trim_mode_inverts_analysis_mode():
    """Feeding back the thrust of an analysis run recovers its collective."""
    thrust = run_main().get_val('T')[0]
    p = build(mode='trim')
    for k, v in MAIN_ROTOR.items():
        p.set_val(k, v)
    p.set_val('T_target', thrust)
    p.run_model()
    assert p.get_val('theta_0')[0] == pytest.approx(17.5, abs=1e-3)
    assert p.get_val('T')[0] == pytest.approx(thrust, abs=1e-4)


def test_trim_converges_from_several_starts():
    """Newton must land on the same collective from anywhere in the domain."""
    got = []
    for start in (11.0, 14.0, 17.5, 24.0):
        p = build(mode='trim', theta_0_bounds=(10.5, 25.0))
        for k, v in MAIN_ROTOR.items():
            p.set_val(k, v)
        p.set_val('T_target', 20000.0)
        p.set_val('theta_0', start)
        p.run_model()
        got.append(p.get_val('theta_0')[0])
    assert np.ptp(got) < 1e-6


def test_preprocess_partials_are_analytic():
    """G0 is complex step clean on a tapered blade, where every branch is live."""
    p = build(RotorPreprocessGroup, num_elements=10)
    p.set_val('c_tip', 1.0)
    p.set_val('r_1', 18.0)
    p.run_model()
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    worst = max(float(np.max(np.abs(s['abs error'].forward)))
                for e in data.values() for s in e.values()
                if s['abs error'].forward is not None)
    assert worst < 1e-10


def test_total_derivatives_through_the_cycle_and_the_moving_bound():
    """dT/dtheta_0 must match a central difference across the whole model."""
    p = build()
    for k, v in MAIN_ROTOR.items():
        p.set_val(k, v)
    p.model.add_design_var('theta_0')
    p.model.add_objective('T')
    p.setup(force_alloc_complex=True)
    for k, v in MAIN_ROTOR.items():
        p.set_val(k, v)
    p.set_val('theta_0', 17.5)
    p.run_model()
    analytic = p.compute_totals(of=['T'], wrt=['theta_0'],
                                return_format='flat_dict')[('T', 'theta_0')]

    step = 1e-3
    thrust = []
    for th in (17.5 - step, 17.5 + step):
        thrust.append(run_main(theta_0=th).get_val('T')[0])
    fd = (thrust[1] - thrust[0]) / (2.0 * step)
    assert analytic.ravel()[0] == pytest.approx(fd, rel=1e-4)


def test_trim_derivative_needs_a_retrimmed_reference():
    """check_totals does not reconverge Newton, so compare against retrimmed runs."""
    def trimmed_hp(v_tip):
        p = build(mode='trim')
        for k, val in MAIN_ROTOR.items():
            p.set_val(k, val)
        p.set_val('V_tip', v_tip)
        p.set_val('T_target', 20000.0)
        p.run_model()
        return p.get_val('power_hp')[0]

    p = build(mode='trim')
    for k, v in MAIN_ROTOR.items():
        p.set_val(k, v)
    p.model.add_design_var('V_tip')
    p.model.add_objective('power_hp')
    p.setup(force_alloc_complex=True)
    for k, v in MAIN_ROTOR.items():
        p.set_val(k, v)
    p.set_val('T_target', 20000.0)
    p.run_model()
    analytic = p.compute_totals(of=['power_hp'], wrt=['V_tip'],
                                return_format='flat_dict')
    fd = (trimmed_hp(651.0) - trimmed_hp(649.0)) / 2.0
    assert analytic[('power_hp', 'V_tip')].ravel()[0] == pytest.approx(fd,
                                                                      rel=1e-4)
