"""Tests for G2, NumericalRotorGroup and its components.

Book anchors: Table 3.2 (p. 193), Figure 3.48 (p. 194), p. 200 (the p_n
groups), p. 211-214, Figure 3.56 (p. 218), p. 221, Figure 3.58 (p. 222),
p. 225, p. 228.

Departures from the printed text asserted here rather than tolerated, each
documented in the component that makes them:

  * the induced velocity varies as cos(psi), not the sin(psi) of p. 209;
  * the spanwise skin friction of the H-force uses U_TR, not U_B^2/U_T, which
    changes C_H/sigma by a factor of 43 and its sign;
  * the dynamic stall delay is saturated, because unbounded it drives the drag
    divergence angle to -116 deg and c_d to 110.
"""

import numpy as np
import openmdao.api as om
import pytest

from prouty.forward_flight.alpha_comp import AlphaComp
from prouty.forward_flight.chord_force_comp import ChordForceComp
from prouty.forward_flight.disc_airfoil_group import DiscAirfoilGroup
from prouty.forward_flight.disc_integral_comp import DiscIntegralComp
from prouty.forward_flight.gyro_moment_comp import GyroMomentComp
from prouty.forward_flight.lift_coef_bounds_comp import LiftCoefBoundsComp
from prouty.forward_flight.loadings_comp import LoadingsComp
from prouty.forward_flight.local_mach_comp import LocalMachComp
from prouty.forward_flight.normal_force_comp import NormalForceComp
from prouty.forward_flight.numerical_rotor_group import NumericalRotorGroup
from prouty.forward_flight.pitch_dist_comp import PitchDistComp
from prouty.forward_flight.resultant_vel_comp import ResultantVelComp
from prouty.forward_flight.rotor_disc_grid_comp import RotorDiscGridComp
from prouty.forward_flight.stall_delay_comp import StallDelayComp
from prouty.forward_flight.stall_indicators_comp import StallIndicatorsComp
from prouty.forward_flight.sweep_angle_comp import SweepAngleComp
from prouty.forward_flight.velocity_comps import (PerpVelComp, RadialVelComp,
                                                  TangentialVelComp)

# Example helicopter in level flight at mu = 0.3, Table 3.2 p. 193
REF = dict(mu=0.3, lambda_p=-0.0316, theta_0=np.deg2rad(15.8),
           theta_1=np.deg2rad(-10.0), V_tip=650.0, V_son=1116.0,
           c_R=2.0 / 30.0, B=0.97, x_0=0.15, gamma=8.05033, a=6.0)
IMPOSED = dict(vi_OR=0.012203, a0=np.deg2rad(4.3))
MOMENTUM = dict(R=30.0, sigma=0.084883)

FIELD_CHAIN = (('grid', RotorDiscGridComp), ('pitch', PitchDistComp),
               ('ut', TangentialVelComp), ('up', PerpVelComp),
               ('ur', RadialVelComp), ('ub', ResultantVelComp),
               ('mach', LocalMachComp), ('al', AlphaComp),
               ('sw', SweepAngleComp), ('sd', StallDelayComp))


def set_known(problem, values):
    """Set only the variables the assembled model actually owns.

    The kinematic chain has no use for gamma, a or sigma; passing the whole
    reference dictionary to every builder keeps the tests readable and costs
    a name check.
    """
    names = {meta['prom_name'] for _, meta in
             problem.model.list_inputs(out_stream=None, prom_name=True,
                                       val=False)}
    names |= {meta['prom_name'] for _, meta in
              problem.model.list_outputs(out_stream=None, prom_name=True,
                                         val=False)}
    for name, value in values.items():
        if name in names:
            problem.set_val(name, value)


def errors(entry):
    data = entry['abs error']
    return [e for e in (data.forward, data.reverse) if e is not None]


def build_field(n_psi=24, n_r=41, nn=1, **overrides):
    """The kinematic chain only: grid through stall delay."""
    grid = dict(num_nodes=nn, num_azimuth=n_psi, num_radial=n_r)
    p = om.Problem()
    for name, comp in FIELD_CHAIN:
        p.model.add_subsystem(name, comp(**grid), promotes=['*'])
    p.setup(force_alloc_complex=True)
    set_known(p, {**REF, **IMPOSED, 'A_1': np.deg2rad(-2.3),
                  'B_1': np.deg2rad(4.9), **overrides})
    p.run_model()
    return p


def build_rotor(n_psi=12, n_r=15, nn=1, inflow='imposed', trim=True,
                **overrides):
    grid = dict(num_nodes=nn, num_azimuth=n_psi, num_radial=n_r)
    p = om.Problem()
    p.model.add_subsystem('g2', NumericalRotorGroup(
        airfoil=DiscAirfoilGroup(**grid), trim=trim, inflow=inflow, **grid),
        promotes=['*'])
    p.setup()
    values = dict(REF)
    values.update(MOMENTUM if inflow == 'momentum' else IMPOSED)
    values.update(overrides)
    set_known(p, values)
    p.set_val('A_1', np.deg2rad(-2.3))
    p.set_val('B_1', np.deg2rad(4.9))
    return p


# ------------------------------------------------------- grid and quadrature
@pytest.mark.parametrize('B, x_0', [(1.0, 0.0), (0.97, 0.0), (0.97, 0.15),
                                    (0.94, 0.20)])
def test_quadrature_moments_match_the_closed_form_groups(B, x_0):
    """sum w r^n must reproduce the p_n groups of p. 200.

    This is a cross-check between G1 and G2, not an internal consistency
    test: the same integrals appear as p1..p4 in the closed-form equations.
    """
    p = om.Problem()
    p.model.add_subsystem('g', RotorDiscGridComp(num_radial=81), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('B', B)
    p.set_val('x_0', x_0)
    p.run_model()

    r, w = p.get_val('r_R'), p.get_val('w_r_lift')[0]
    exact = [B - x_0, (B ** 2 - x_0 ** 2) / 2, (B ** 3 - x_0 ** 3) / 3,
             (B ** 4 - x_0 ** 4) / 4]
    for order, reference in enumerate(exact):
        assert (w * r ** order).sum() == pytest.approx(reference, rel=2e-3)


def test_azimuth_average_is_exact_below_the_harmonic_count():
    p = om.Problem()
    p.model.add_subsystem('g', RotorDiscGridComp(num_azimuth=24),
                          promotes=['*'])
    p.setup()
    p.run_model()

    psi, w = p.get_val('psi'), p.get_val('w_psi')[0]
    assert w * np.sin(psi).sum() == pytest.approx(0.0, abs=1e-14)
    assert w * (np.sin(psi) ** 2).sum() == pytest.approx(0.5)
    assert w * (np.sin(psi) * np.cos(psi)).sum() == pytest.approx(0.0, abs=1e-14)


def test_tip_loss_weights_stay_differentiable_on_the_grid_nodes():
    """B = 1 and x_0 = 0 sit exactly on the outer nodes, the awkward case."""
    p = om.Problem()
    p.model.add_subsystem('g', RotorDiscGridComp(num_nodes=2), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('B', [1.0, 0.95])
    p.set_val('x_0', 0.0)
    p.run_model()

    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    checked = 0
    for derivatives in data.values():
        for key, value in derivatives.items():
            for error in errors(value):
                assert error < 1e-10, key
                checked += 1
    assert checked > 0


# --------------------------------------------------------- pitch and velocity
def test_pitch_corner_values():
    p = build_field(n_psi=24)
    theta = p.get_val('theta', units='deg')[0]
    psi = np.degrees(p.get_val('psi'))
    tip = -1

    at = lambda angle: theta[np.argmin(np.abs(psi - angle)), tip]
    assert at(270.0) == pytest.approx(15.8 - 10.0 + 4.9, abs=1e-6)
    assert at(90.0) == pytest.approx(15.8 - 10.0 - 4.9, abs=1e-6)
    assert at(0.0) == pytest.approx(15.8 - 10.0 + 2.3, abs=1e-6)
    assert theta[:, tip].mean() == pytest.approx(5.8, abs=1e-6)


def test_pitch_rates_are_the_azimuth_derivatives():
    p = build_field(n_psi=72)
    rate = p.get_val('theta_dot_Om')[0]
    second = p.get_val('theta_ddot_Om2')[0]
    d_psi = np.diff(p.get_val('psi'))[0]

    # central difference, so the residual is the O(d_psi^2) truncation:
    # (d_psi^2/6)|theta''''| is 1.2e-4 with 72 stations
    numeric = (np.roll(rate, -1) - np.roll(rate, 1)) / (2 * d_psi)
    assert np.abs(numeric - second).max() < 2e-4


def test_velocity_identities():
    p = build_field(n_psi=24)
    UT, UP = p.get_val('UT_bar')[0], p.get_val('UP_bar')[0]
    UR, UB = p.get_val('UR_bar')[0], p.get_val('UB_bar')[0]
    psi = np.degrees(p.get_val('psi'))
    i90 = np.argmin(np.abs(psi - 90.0))
    i270 = np.argmin(np.abs(psi - 270.0))

    assert UT[i90, -1] == pytest.approx(1.3)
    assert UT[i270, -1] == pytest.approx(0.7)
    assert UR[i90] == pytest.approx(0.0, abs=1e-15)
    # U_P reduces to lambda' wherever cos(psi) vanishes, at every station
    assert UP[i270] == pytest.approx(np.full(UP.shape[1], -0.0316))
    assert UB.min() > 0.0
    assert np.all(UB >= np.abs(UT) - 1e-12)


def test_reverse_flow_region():
    """U_T < 0 inside a circle of diameter mu on the retreating side."""
    p = build_field(n_psi=72, n_r=81)
    UT = p.get_val('UT_bar')[0]
    r, psi = p.get_val('r_R'), p.get_val('psi')

    reverse = UT < 0.0
    assert reverse.mean() == pytest.approx(0.10, abs=0.02)
    # nothing reversed on the advancing side
    assert not reverse[np.sin(psi) > 0.05].any()
    # the boundary is r/R = -mu sin(psi)
    for j in np.where(np.sin(psi) < -0.1)[0]:
        edge = -0.3 * np.sin(psi[j])
        assert r[reverse[j]].max() <= edge + 1e-9


# ------------------------------------------------ angle of attack and quadrant
def test_quadrant_rule_reproduces_proutys_example():
    """p. 214: U_P and U_T both negative must give 225 deg, not -135."""
    p = om.Problem()
    p.model.add_subsystem('a', AlphaComp(num_azimuth=2, num_radial=2),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('theta', np.zeros((1, 2, 2)))
    p.set_val('UT_bar', np.array([[[1.0, 1.0], [-1.0, -1.0]]]))
    p.set_val('UP_bar', np.array([[[1.0, -1.0], [1.0, -1.0]]]))
    p.run_model()

    phi = p.get_val('phi', units='deg')[0]
    assert phi[0, 0] == pytest.approx(45.0)
    assert phi[0, 1] == pytest.approx(-45.0)
    assert phi[1, 0] == pytest.approx(135.0)
    assert phi[1, 1] == pytest.approx(225.0)


def test_quadrant_rule_only_touches_the_reverse_flow_region():
    p = build_field(n_psi=24)
    reference = p.get_val('alpha', units='deg')[0]
    UT = p.get_val('UT_bar')[0]

    q = om.Problem()
    q.model.add_subsystem('a', AlphaComp(num_azimuth=24, num_radial=41,
                                         quadrant=False), promotes=['*'])
    q.setup()
    for name in ('theta', 'UT_bar', 'UP_bar'):
        q.set_val(name, p.get_val(name))
    q.run_model()

    changed = np.abs(reference - q.get_val('alpha', units='deg')[0]) > 1e-9
    assert np.all(UT[changed] < 0.0)
    assert np.abs(reference - q.get_val('alpha', units='deg')[0])[changed] \
        == pytest.approx(360.0)


def test_peak_angle_of_attack_is_inboard_of_the_tip():
    """p. 193: with 10 deg of twist the highest local angle of attack is at
    the 70 % station, not at the retreating tip."""
    p = build_field(n_psi=24, n_r=81)
    alpha = p.get_val('alpha', units='deg')[0]
    r = p.get_val('r_R')
    i270 = np.argmin(np.abs(np.degrees(p.get_val('psi')) - 270.0))

    outboard = r >= 0.35
    peak = np.argmax(alpha[i270, outboard])
    assert r[outboard][peak] == pytest.approx(0.72, abs=0.04)
    assert alpha[i270, outboard][peak] > alpha[i270, -1]
    # Table 3.2 gives alpha_1,270 = 8.3 deg
    assert alpha[i270, -1] == pytest.approx(8.3, abs=0.25)


# ------------------------------------------------------- sweep and stall delay
@pytest.mark.parametrize('mu, reach', [(0.30, 0.52), (0.45, 0.78)])
def test_sweep_region_matches_figure_356(mu, reach):
    """Figure 3.56: the 30 deg sweep lobe reaches r/R = mu/tan(30 deg)."""
    p = build_field(n_psi=72, n_r=81, mu=mu)
    sweep = np.abs(p.get_val('Lambda', units='deg')[0])
    r = p.get_val('r_R')

    inside = sweep[0] > 30.0          # psi = 0, the fore-aft line
    assert r[inside].max() == pytest.approx(reach, abs=0.03)


def test_sweep_vanishes_where_the_flow_is_chordwise():
    p = build_field(n_psi=24)
    sweep = p.get_val('Lambda', units='deg')[0]
    psi = np.degrees(p.get_val('psi'))

    # cos(psi) is 1e-16 rather than 0 at these stations, and at the root the
    # epsilon under |U_T| turns that into 3e-7 deg of sweep
    for angle in (90.0, 270.0):
        j = np.argmin(np.abs(psi - angle))
        assert np.abs(sweep[j]).max() < 1e-6
    assert np.allclose(p.get_val('cos_Lambda') * p.get_val('sec_Lambda'), 1.0)


def test_stall_delay_changes_sign_at_mach_06():
    """gamma = 1.76 ln(0.6/M): above Mach 0.6 shocks bring stall forward."""
    p = build_field(n_psi=24)
    M = p.get_val('M')[0]
    delay = p.get_val('d_alpha_stall', units='deg')[0]
    rate = p.get_val('alpha_rate')[0]

    outboard = slice(None), slice(20, None)
    gamma = 1.76 * np.log(0.6 / M[outboard])
    expected = np.sign(gamma * rate[outboard])
    assert np.all(np.sign(delay[outboard]) == expected)


def test_stall_delay_saturation():
    """Unbounded the delay reaches -200 deg, which drives c_d to 110."""
    loose = build_field(n_psi=24, n_r=41)
    delay = loose.get_val('d_alpha_stall', units='deg')[0]
    assert np.abs(delay).max() == pytest.approx(10.0, abs=1e-6)

    raw = om.Problem()
    raw.model.add_subsystem('sd', StallDelayComp(num_azimuth=24, num_radial=41,
                                                 max_delay=0.0), promotes=['*'])
    raw.setup()
    for name in ('alpha', 'M', 'UB_bar', 'psi', 'c_R'):
        raw.set_val(name, loose.get_val(name))
    raw.run_model()
    assert raw.get_val('d_alpha_stall', units='deg')[0].min() < -100.0


def test_azimuth_difference_is_unwrapped():
    """alpha jumps 360 deg across the reverse flow boundary by construction."""
    p = build_field(n_psi=24)
    rate = p.get_val('alpha_rate')[0]
    alpha = p.get_val('alpha')[0]
    d_psi = np.diff(p.get_val('psi'))[0]

    naive = (np.roll(alpha, -1, axis=0) - np.roll(alpha, 1, axis=0)) / (2 * d_psi)
    assert np.abs(naive).max() > 1.5 * np.abs(rate).max()


# --------------------------------------------------------------- lift bounds
def test_lift_bounds_match_figure_358():
    angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, -45.0])
    p = om.Problem()
    p.model.add_subsystem('b', LiftCoefBoundsComp(num_azimuth=1,
                                                  num_radial=angles.size),
                          promotes=['*'])
    p.setup()
    p.set_val('alpha', np.deg2rad(angles).reshape(1, 1, -1))
    p.set_val('cl', np.full((1, 1, angles.size), 100.0))
    p.run_model()

    cl_max = p.get_val('cl_max')[0, 0]
    assert cl_max[1] == pytest.approx(2.0 * np.pi)      # peak at 45 deg
    assert cl_max[0] == pytest.approx(0.0, abs=1e-12)
    assert cl_max[2] == pytest.approx(0.0, abs=1e-12)
    # folded beyond 90 deg, signs following sin(2 alpha)
    assert cl_max[3] == pytest.approx(-2.0 * np.pi)
    assert cl_max[5] == pytest.approx(2.0 * np.pi)
    assert cl_max[7] == pytest.approx(-2.0 * np.pi)
    # an unbounded coefficient is clipped, a reasonable one is not
    assert p.get_val('cl_bounded')[0, 0, 1] == pytest.approx(2.0 * np.pi,
                                                             abs=0.02)


# ------------------------------------------------------ forces and integration
def test_normal_force_product_is_regular():
    p = build_field(n_psi=24)
    UB = p.get_val('UB_bar')[0]
    n = UB.size

    q = om.Problem()
    q.model.add_subsystem('nf', NormalForceComp(num_azimuth=24, num_radial=41),
                          promotes=['*'])
    q.setup(force_alloc_complex=True)
    for name in ('UT_bar', 'UP_bar', 'UB_bar'):
        q.set_val(name, p.get_val(name))
    q.set_val('cl', np.full(UB.shape, 0.5).reshape(1, 24, 41))
    q.set_val('cd', np.full(UB.shape, 0.01).reshape(1, 24, 41))
    q.run_model()

    assert np.abs(q.get_val('cN_UB2')[0] - UB ** 2 * q.get_val('cN')[0]).max() \
        < 1e-15
    # c_N itself is amplified by 1/U_B, which is why the loadings use cN_UB2
    assert 1.0 / UB.min() > 50.0


def test_spanwise_friction_form_changes_the_h_force_outright():
    """p. 212 prints U_B^2/U_T, which is singular on the reverse flow
    boundary; the rederived U_TR form is finite. Nothing else moves."""
    field = build_field(n_psi=24, n_r=41)
    grid = dict(num_azimuth=24, num_radial=41)
    results = {}

    for form in ('book', 'regular'):
        p = om.Problem()
        p.model.add_subsystem('ld', LoadingsComp(spanwise_form=form, **grid),
                              promotes=['*'])
        p.model.add_subsystem('it', DiscIntegralComp(**grid), promotes=['*'])
        p.setup()
        for name in ('UT_bar', 'UTR_bar', 'UB_bar', 'r_R', 'psi', 'mu',
                     'w_r_lift', 'w_r_full', 'w_psi'):
            p.set_val(name, field.get_val(name))
        shape = (1, 24, 41)
        p.set_val('cN_UB2', np.full(shape, 0.5))
        p.set_val('cc0_UB2', np.full(shape, 0.01))
        p.set_val('ccind_UB2', np.full(shape, -0.005))
        p.run_model()
        results[form] = {name: p.get_val(name)[0] for name in
                         ('CT_sigma', 'CQ_sigma', 'CH_sigma', 'CM_sigma')}

    for name in ('CT_sigma', 'CQ_sigma', 'CM_sigma'):
        assert results['book'][name] == pytest.approx(results['regular'][name])
    assert results['book']['CH_sigma'] < 0.0 < results['regular']['CH_sigma']


def test_gyroscopic_moments_cross_the_rates():
    """The pitching moment comes from the roll rate, and the other way."""
    p = om.Problem()
    p.model.add_subsystem('g', GyroMomentComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('gamma', 8.05033)
    p.set_val('a', 6.0)
    p.set_val('Phi_dot_Om', 0.01)
    p.set_val('Theta_dot_Om', 0.02)
    p.run_model()

    assert p.get_val('CM_sigma_gyro')[0] == pytest.approx(2 * 6.0 * 0.01 / 8.05033)
    assert p.get_val('CR_sigma_gyro')[0] == pytest.approx(2 * 6.0 * 0.02 / 8.05033)


def test_stall_indicators():
    psi = np.arange(24) * 2 * np.pi / 24
    profile = 0.0025 * (1.0 + 0.4 * np.sin(psi))

    p = om.Problem()
    p.model.add_subsystem('s', StallIndicatorsComp(num_azimuth=24),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('theta_0', np.deg2rad(15.8))
    p.set_val('theta_1', np.deg2rad(-10.0))
    p.set_val('B_1', np.deg2rad(4.9))
    p.set_val('lambda_p', -0.0316)
    p.set_val('mu', 0.3)
    p.set_val('CQ0_sigma_psi', profile.reshape(1, -1))
    p.run_model()

    assert p.get_val('alpha_1270', units='deg')[0] == pytest.approx(8.3, abs=0.25)
    peak = p.get_val('CQ0_sigma_max')[0]
    assert peak >= profile.max()
    assert peak - profile.max() < 2.0e-5 * np.log(24)


# ------------------------------------------------------------ assembled group
def test_trim_drives_both_hub_moments_to_zero():
    p = build_rotor()
    p.run_model()

    assert abs(p.get_val('CM_sigma')[0]) < 1e-12
    assert abs(p.get_val('CR_sigma')[0]) < 1e-12
    # A_1 lands on the closed-form combination of Table 3.2
    assert p.get_val('A_1', units='deg')[0] == pytest.approx(-2.3, abs=0.2)


@pytest.mark.parametrize('A_1, B_1', [(0.0, 0.0), (-10.0, 15.0), (8.0, -8.0)])
@pytest.mark.slow
def test_trim_is_robust_to_the_starting_cyclic(A_1, B_1):
    p = build_rotor()
    p.set_val('A_1', np.deg2rad(A_1))
    p.set_val('B_1', np.deg2rad(B_1))
    p.run_model()

    assert abs(p.get_val('CM_sigma')[0]) < 1e-12
    assert abs(p.get_val('CR_sigma')[0]) < 1e-12


@pytest.mark.parametrize('CT_0', [0.02, 0.08, 0.20])
@pytest.mark.slow
def test_inflow_loop_is_robust_to_the_starting_thrust(CT_0):
    """p. 213 closes the loop by iteration; the fixed point must not depend
    on where it starts."""
    p = build_rotor(inflow='momentum')
    p.set_val('CT_sigma_ref', CT_0)
    p.run_model()

    assert p.get_val('CT_sigma')[0] == pytest.approx(
        p.get_val('CT_sigma_ref')[0], rel=1e-8)
    assert p.get_val('vi_OR')[0] == pytest.approx(
        0.084883 * p.get_val('CT_sigma')[0] / 0.6, rel=1e-8)


def test_closing_the_inflow_loop_barely_moves_the_thrust():
    """Documented finding: the loop is self-compensating. A lower C_T/sigma
    gives less downwash, which raises the angle of attack, which brings the
    thrust back. The 14 % gap against the closed form is the stall and
    reverse flow the closed form ignores, not the inflow."""
    imposed = build_rotor(inflow='imposed')
    imposed.run_model()
    closed = build_rotor(inflow='momentum')
    closed.run_model()

    assert closed.get_val('CT_sigma')[0] == pytest.approx(
        imposed.get_val('CT_sigma')[0], rel=0.01)
    assert closed.get_val('vi_OR')[0] < imposed.get_val('vi_OR')[0]


@pytest.mark.slow
def test_thrust_stalls_out_at_high_collective():
    """Beyond about 18 deg the disc stalls: thrust stops rising while torque
    runs away. That turnover is the shape of the chart curves of p. 254."""
    results = {}
    for theta_0 in (14.0, 18.0, 22.0):
        p = build_rotor(inflow='momentum', theta_0=np.deg2rad(theta_0))
        p.run_model()
        results[theta_0] = (p.get_val('CT_sigma')[0], p.get_val('CQ_sigma')[0])

    assert results[18.0][0] > results[14.0][0]
    assert results[22.0][0] < results[18.0][0]
    assert results[22.0][1] > 2.0 * results[18.0][1]


@pytest.mark.parametrize('theta_0', [0.0, 6.0, 10.0, 24.0, 30.0])
@pytest.mark.slow
def test_the_whole_collective_range_converges(theta_0):
    """This used to be a documented failure. Three things fixed it, none of
    them a tolerance:

      * C_T/sigma was bounded below at zero, which made the residual
        INFEASIBLE at low collective -- a rotor at theta_0 = 4 deg with
        -10 deg of twist genuinely produces downward thrust, and the disc
        returns -0.10;
      * maxiter was 40, and at theta_0 = 24 deg the residuals reach zero on
        the fortieth iteration exactly, so a converged answer was reported as
        a failure;
      * the Chapter 6 airfoil model returned NaN outside its fitted range,
        and one NaN in a solver state poisons the whole residual vector.
    """
    p = build_rotor(inflow='momentum', theta_0=np.deg2rad(theta_0))
    p.run_model()

    residual = max(abs(p.get_val(name)[0])
                   for name in ('res_CT', 'res_CM', 'res_CR'))
    assert residual < 1e-8


def test_thrust_reverses_at_low_collective():
    """-10 deg of twist puts the blade at negative pitch outboard well before
    the collective reaches zero."""
    low = build_rotor(inflow='momentum', theta_0=np.deg2rad(6.0))
    low.run_model()
    high = build_rotor(inflow='momentum', theta_0=np.deg2rad(16.0))
    high.run_model()

    assert low.get_val('CT_sigma')[0] < 0.0 < high.get_val('CT_sigma')[0]
    # the induced velocity follows the thrust, as momentum requires
    assert low.get_val('vi_OR')[0] < 0.0


def test_airfoil_model_never_returns_nan():
    """A solver state can carry any value between iterations, so the airfoil
    model has to answer with a number whatever it is handed."""
    from prouty.airfoil.lift_model_coefs_comp import LiftModelCoefsComp
    from prouty.airfoil.lift_coef_comp import LiftCoefComp

    M = np.array([-0.5, 0.0, 0.5, 0.99, 1.0, 1.5, 3.0])
    p = om.Problem()
    p.model.add_subsystem('c', LiftModelCoefsComp(num_nodes=M.size),
                          promotes=['*'])
    p.model.add_subsystem('l', LiftCoefComp(num_nodes=M.size), promotes=['*'])
    p.setup()
    p.set_val('M', M)
    p.set_val('alpha', np.full(M.size, 30.0))
    p.run_model()

    for name in ('a', 'alpha_L', 'K1', 'K2', 'cl'):
        assert np.all(np.isfinite(p.get_val(name))), name


def test_stall_indicators_are_carried_by_the_group():
    """p. 228: both chart limit lines come out of a converged solution."""
    p = build_rotor(inflow='momentum')
    p.run_model()

    # Table 3.2 gives alpha_1,270 = 8.3 deg for the reference condition
    assert p.get_val('alpha_1270', units='deg')[0] == pytest.approx(8.3, abs=0.4)
    assert p.get_val('CQ0_sigma_max')[0] > 0.0


def test_both_indicators_rise_into_stall():
    """They measure different things -- a single point on the disc against an
    integral over the blade -- so they cross the chart limits of p. 254 at
    different collectives, and that is the whole reason Prouty keeps both.
    Measured here: alpha_1,270 passes 12 deg between theta_0 = 16 and 18 deg,
    while the profile torque is still at 0.0032 at 18 and only passes 0.004
    between 18 and 20. The tip angle warns first."""
    tip, torque = {}, {}
    for theta_0 in (16.0, 18.0, 20.0):
        p = build_rotor(inflow='momentum', theta_0=np.deg2rad(theta_0))
        p.run_model()
        tip[theta_0] = p.get_val('alpha_1270', units='deg')[0]
        torque[theta_0] = p.get_val('CQ0_sigma_max')[0]

    assert tip[16.0] < 12.0 < tip[18.0] < 16.0 < tip[20.0]
    assert torque[18.0] < 0.004 < torque[20.0]
    assert torque[20.0] > 4.0 * torque[18.0]


# ----------------------------------------------------------------- gradients
def test_field_chain_partials():
    p = build_field(n_psi=12, n_r=11, nn=2, mu=[0.3, 0.45],
                    lambda_p=[-0.0316, -0.10], vi_OR=[0.0122, 0.0081],
                    a0=[0.075, 0.084], theta_0=[0.277, 0.40],
                    A_1=[-0.04, -0.05], B_1=[0.0855, 0.18], B=[0.97, 0.96])

    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    checked = 0
    for component, derivatives in data.items():
        for key, value in derivatives.items():
            for error in errors(value):
                assert error < 1e-8, f'{component} {key}'
                checked += 1
    assert checked > 0


def test_force_chain_partials():
    rng = np.random.default_rng(0)
    grid = dict(num_nodes=2, num_azimuth=6, num_radial=5)
    shape = (2, 6, 5)

    p = om.Problem()
    p.model.add_subsystem('nf', NormalForceComp(**grid), promotes=['*'])
    p.model.add_subsystem('cf', ChordForceComp(**grid), promotes=['*'])
    p.model.add_subsystem('ld', LoadingsComp(**grid), promotes=['*'])
    p.model.add_subsystem('it', DiscIntegralComp(**grid), promotes=['*'])
    p.setup(force_alloc_complex=True)

    UT = rng.uniform(-0.4, 1.4, shape)
    UP = rng.uniform(-0.2, 0.1, shape)
    UR = rng.uniform(-0.45, 0.45, shape)
    p.set_val('UT_bar', UT)
    p.set_val('UP_bar', UP)
    p.set_val('UB_bar', np.sqrt(UT ** 2 + UP ** 2))
    p.set_val('UTR_bar', np.hypot(UT, UR))
    p.set_val('cl', rng.uniform(-2.0, 2.0, shape))
    p.set_val('cd', rng.uniform(0.008, 0.05, shape))
    p.set_val('r_R', np.linspace(0.0, 1.0, 5))
    p.set_val('psi', np.arange(6) * 2 * np.pi / 6)
    p.set_val('mu', [0.3, 0.45])
    p.set_val('w_r_lift', np.full((2, 5), 0.2))
    p.set_val('w_r_full', np.full(5, 0.2))
    p.run_model()

    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    checked = 0
    for component, derivatives in data.items():
        for key, value in derivatives.items():
            for error in errors(value):
                assert error < 1e-8, f'{component} {key}'
                checked += 1
    assert checked > 0


# ----------------------------------------------------------------- totals
@pytest.mark.slow
def test_totals_through_the_trim_and_inflow_loops():
    """Analytic totals across a Newton that solves A_1, B_1 and C_T/sigma.

    Checked against check_totals with central differences, which for this
    group DOES reconverge the solver at each perturbation -- verified against
    manually retrimmed central differences on twelve pairs, agreeing to
    between 1e-6 and 1e-3 relative. That is worth stating because the trim
    groups of G1b behave differently and needed the manual route.

    Tolerances are relative and loose on C_H/sigma, whose derivatives are two
    orders of magnitude smaller than the thrust ones, so the central
    difference has correspondingly less signal.
    """
    p = build_rotor(inflow='momentum')
    p.run_model()

    data = p.check_totals(of=['CT_sigma', 'CQ_sigma', 'CH_sigma'],
                          wrt=['theta_0', 'lambda_p', 'mu'], method='fd',
                          step=1e-4, form='central', compact_print=True,
                          out_stream=None)

    checked = 0
    for key, entry in data.items():
        analytic = np.atleast_2d(entry['J_fwd'])[0, 0]
        difference = np.atleast_2d(entry['J_fd'])[0, 0]
        scale = max(abs(difference), 1e-4)
        assert abs(analytic - difference) / scale < 5e-3, key
        checked += 1
    assert checked == 9


def test_thrust_derivatives_have_the_right_sign_and_size():
    """A sanity net on the totals: more collective and less downwash both
    raise the thrust, and at mu = 0.3 the closed form puts dC_T/d(theta_0) at
    (a/4)(2/3 + mu^2) = 1.14 and dC_T/d(lambda') at a/4 = 1.5 for B = 1."""
    p = build_rotor(inflow='momentum')
    p.run_model()
    totals = p.compute_totals(of=['CT_sigma'], wrt=['theta_0', 'lambda_p'])

    collective = totals['CT_sigma', 'theta_0'][0, 0]
    inflow = totals['CT_sigma', 'lambda_p'][0, 0]
    assert 0.5 < collective < 1.2
    assert 0.8 < inflow < 1.6
