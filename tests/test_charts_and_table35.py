"""Tests for G3 (isolated rotor charts) and G4 (Table 3.5 cases 7 to 9).

Book anchors: p. 229-231 and the chart plates p. 254-271, Table 3.5 case 7
(p. 242-244), case 8 (p. 245-246) and case 9 (p. 247-249). Case 9 is the only
one in the chapter checked against measurement, test 276 run 3 of reference
3.27, so it is worth more than the rest put together.

Chart readings used here were digitised, not eyeballed. Four separate readings
were wrong by exactly one curve before digitisation; see validation_forward_flight
section 5.
"""

import numpy as np
import openmdao.api as om
import pytest

from prouty.forward_flight.chart_parameter_comp import ChartParameterComp
from prouty.forward_flight.climb_drag_area_comp import ClimbDragAreaComp
from prouty.forward_flight.disc_airfoil_group import DiscAirfoilGroup
from prouty.forward_flight.numerical_rotor_group import NumericalRotorGroup
from prouty.forward_flight.propulsive_balance_comp import PropulsiveBalanceComp
from prouty.forward_flight.rotor_drag_area_comp import RotorDragAreaComp
from prouty.forward_flight.rotor_side_force_comp import RotorSideForceComp
from prouty.forward_flight.tunnel_wall_correction_comp import (
    TunnelWallCorrectionComp)
from prouty.forward_flight.wind_tunnel_rotor_group import WindTunnelRotorGroup

# Chart rotor, p. 229. M_1,90 = 0.7 fixes the tip speed at each mu, so the
# plates are NOT at constant tip speed.
CHART = dict(theta_1=np.deg2rad(-5.0), c_R=0.079, B=0.97, x_0=0.0,
             gamma=8.0, a=6.0, sigma=0.062, V_son=1116.0)

# H-34 in the 40 by 80 tunnel, p. 247
DISC_AREA = 1820.0
TUNNEL = dict(mu=0.3, sigma=0.062, theta_75=np.deg2rad(9.0),
              theta_1=np.deg2rad(-8.0), alpha_s=np.deg2rad(5.0),
              a1s=np.deg2rad(0.3), b1s=np.deg2rad(0.2), gamma=9.3, a=6.0,
              R=np.sqrt(DISC_AREA / np.pi), V_tip=0.74 * 1116.0 / 1.3,
              V_son=1116.0, cd_bar=0.0100, B=0.97, x_0=0.0, c_R=0.05,
              area_ratio=DISC_AREA / 3200.0, delta_WL=-0.54)

MEASURED = dict(CT_sigma=0.110, CQ_sigma=0.0040, CXR_sigma=-0.0058,
                A_1=-3.3, B_1=10.0)


def errors(entry):
    data = entry['abs error']
    return [e for e in (data.forward, data.reverse) if e is not None]


def build_chart_rotor(mu, n_psi=12, n_r=25):
    grid = dict(num_nodes=1, num_azimuth=n_psi, num_radial=n_r)
    p = om.Problem()
    p.model.add_subsystem('g2', NumericalRotorGroup(
        airfoil=DiscAirfoilGroup(**grid), trim=True, inflow='momentum', **grid),
        promotes=['*'])
    p.setup()
    p.model.g2.nonlinear_solver.options['err_on_non_converge'] = False
    p.final_setup()
    for name, value in CHART.items():
        p.set_val(name, value)
    p.set_val('mu', mu)
    p.set_val('V_tip', 0.7 / (1.0 + mu) * 1116.0)
    p.set_val('R', 30.0)
    return p


def chart_point(problem, theta_0, lambda_p):
    problem.set_val('theta_0', np.deg2rad(theta_0))
    problem.set_val('lambda_p', lambda_p)
    problem.run_model()
    get = lambda name, units=None: problem.get_val(name, units=units)[0]
    residual = max(abs(get(name))
                   for name in ('res_CT', 'res_CM', 'res_CR'))
    return dict(CT=get('CT_sigma'), CQ=get('CQ_sigma'), CH=get('CH_sigma'),
                B_1=get('B_1', 'deg'), A_1=get('A_1', 'deg'),
                alpha_1270=get('alpha_1270', 'deg'),
                dCQ=get('CQ0_sigma_max'), converged=residual < 1e-8)


def build_tunnel(rotor='closed_form', **overrides):
    p = om.Problem()
    p.model.add_subsystem('g', WindTunnelRotorGroup(rotor=rotor),
                          promotes=['*'])
    p.setup()
    p.final_setup()
    for name, value in {**TUNNEL, **overrides}.items():
        try:
            p.set_val(name, value)
        except KeyError:
            pass                      # not every variable exists in both paths
    return p


# --------------------------------------------------- chart parameter identity
def test_chart_ordinate_is_lambda_prime_in_disguise():
    """p. 229 and Table 3.5 step e: the chart ordinate
    f/A_b + (sigma/mu^4)(C_T/sigma)^2 equals -2 lambda' (C_T/sigma)/mu^3
    exactly, for an isolated rotor carrying only parasite drag. That identity
    is what makes a converged G2 point a chart point with nothing to solve."""
    rho, tip_speed, radius, A_b = 0.002377, 650.0, 30.0, 240.0
    sigma = A_b / (np.pi * radius ** 2)

    for mu, CT, f in ((0.30, 0.085, 19.4), (0.10, 0.060, 15.0),
                      (0.45, 0.100, 25.0)):
        q = 0.5 * rho * (mu * tip_speed) ** 2
        thrust = CT * rho * A_b * tip_speed ** 2
        lambda_p = mu * (-q * f / thrust) - sigma * CT / (2 * mu)

        p = om.Problem()
        p.model.add_subsystem('c', ChartParameterComp(mode='from_inflow'),
                              promotes=['*'])
        p.setup()
        p.set_val('lambda_p', lambda_p)
        p.set_val('CT_sigma', CT)
        p.set_val('mu', mu)
        p.run_model()

        assert p.get_val('X_chart')[0] == pytest.approx(
            f / A_b + sigma / mu ** 4 * CT ** 2, rel=1e-12)


def test_chart_parameter_matches_table_35_step_e():
    """p. 235: 19.4/240 + 123(.085)(.085)^2 = 0.157."""
    p = om.Problem()
    p.model.add_subsystem('c', ChartParameterComp(mode='to_inflow'),
                          promotes=['*'])
    p.setup()
    p.set_val('f_Ab', 19.4 / 240.0)
    p.set_val('sigma', 0.084883)
    p.set_val('CT_sigma', 0.085)
    p.set_val('mu', 0.3)
    p.run_model()

    assert p.get_val('X_chart')[0] == pytest.approx(0.157, abs=0.001)


@pytest.mark.parametrize('mu, coefficient', [(0.10, 10000.0), (0.30, 123.0),
                                             (0.45, 24.4), (0.50, 16.0)])
def test_the_printed_coefficient_is_one_over_mu_fourth(mu, coefficient):
    """The plates print 10,000 at mu = 0.10 and 123 at 0.30; both are 1/mu^4."""
    assert 1.0 / mu ** 4 == pytest.approx(coefficient, rel=0.005)


# ------------------------------------------------------------- charts from G2
@pytest.mark.slow
def test_chart_4_inflow_curve():
    """Chart 4 at mu = 0.10, column C_T/sigma = 0.02, digitised from p. 255.

    The assignment was fixed by physics, not by counting: at theta_0 = 0 with
    -5 deg of twist the pitch is negative outboard, so lambda' must be POSITIVE
    to make C_T/sigma = 0.02, which rules out the two label artefacts above it.
    """
    reading = {0.0: 0.0599, 8.0: -0.0326, 12.0: -0.0822, 16.0: -0.1351}
    p = build_chart_rotor(0.10)

    for theta_0, expected in reading.items():
        points = []
        for lambda_p in np.linspace(expected - 0.05, expected + 0.05, 9):
            point = chart_point(p, theta_0, lambda_p)
            if point['converged']:
                points.append((point['CT'], lambda_p))
        assert len(points) >= 2, theta_0
        table = np.array(sorted(points))
        found = np.interp(0.02, table[:, 0], table[:, 1])
        assert found == pytest.approx(expected, abs=0.008), theta_0


@pytest.mark.slow
def test_chart_1_ordinate_at_the_clean_column():
    """Chart 1 at mu = 0.10, C_T/sigma = 0.11, where the plate yields exactly
    nine clusters and nine curves are in frame, so the assignment is forced."""
    p = build_chart_rotor(0.10)
    reading = {16.0: (13.8, -0.060), 20.0: (22.9, -0.110)}

    for theta_0, (expected, lambda_p) in reading.items():
        point = chart_point(p, theta_0, lambda_p)
        assert point['converged']
        X = -2.0 * lambda_p * point['CT'] / 0.10 ** 3
        assert X == pytest.approx(expected, rel=0.10), theta_0


@pytest.mark.slow
def test_chart_3_h_force_is_half_and_stays_half():
    """Documented failure, not a passing case: chart 3 gives roughly twice
    G2's C_H/sigma and seven candidate causes were eliminated by measurement
    (validation_forward_flight section 4). Asserted so that a future change which fixes
    it is noticed rather than absorbed silently."""
    p = build_chart_rotor(0.10)
    ratios = []
    for theta_0, lambda_p, chart in ((16.0, -0.060, -0.0022),
                                     (20.0, -0.110, -0.0042)):
        point = chart_point(p, theta_0, lambda_p)
        assert point['converged']
        ratios.append(point['CH'] / chart)

    assert all(0.4 < r < 0.7 for r in ratios), (
        'chart 3 agreement changed; see validation_forward_flight section 4')


@pytest.mark.slow
def test_the_inflow_gradient_phase_is_cos_not_sin():
    """The charts settle the p. 209 misprint, because they were produced by
    Prouty's own program: sin makes C_H/sigma change sign and cuts B_1 by a
    factor of four."""
    grid = dict(num_nodes=1, num_azimuth=12, num_radial=25)
    results = {}
    for phase in ('cos', 'sin'):
        p = om.Problem()
        p.model.add_subsystem('g2', NumericalRotorGroup(
            airfoil=DiscAirfoilGroup(**grid), trim=True, inflow='momentum',
            component_options={'perp_vel': {'gradient_phase': phase}}, **grid),
            promotes=['*'])
        p.setup()
        p.final_setup()
        for name, value in CHART.items():
            p.set_val(name, value)
        p.set_val('mu', 0.10)
        p.set_val('V_tip', 0.7 / 1.1 * 1116.0)
        p.set_val('R', 30.0)
        p.set_val('theta_0', np.deg2rad(12.0))
        p.set_val('lambda_p', -0.020)
        p.run_model()
        results[phase] = (p.get_val('CH_sigma')[0],
                          p.get_val('B_1', units='deg')[0])

    # chart wants C_H/sigma = -0.0007 and B_1 = 1.79 here
    assert results['sin'][0] > 0.0 > results['cos'][0]
    assert results['sin'][1] < 0.5 * results['cos'][1]


@pytest.mark.slow
def test_the_inflow_gradient_drives_the_longitudinal_cyclic_only():
    """A cos(psi) term in U_P is fore-and-aft, so it trims through A_1 and
    leaves B_1 alone. kappa scales it; A_1 is linear in kappa and B_1 is flat.
    """
    p = build_chart_rotor(0.10)
    lateral, longitudinal = [], []
    for kappa in (0.0, 1.0, 2.0):
        p.set_val('kappa', kappa)
        point = chart_point(p, 12.0, -0.020)
        assert point['converged']
        lateral.append(point['B_1'])
        longitudinal.append(point['A_1'])
    p.set_val('kappa', 1.0)

    assert max(lateral) - min(lateral) < 0.02
    steps = np.diff(longitudinal)
    assert steps[0] == pytest.approx(steps[1], rel=0.02)


# --------------------------------------------------- Table 3.5 cases 7 and 8
def test_rotor_drag_area_matches_case_7():
    """p. 244 step s."""
    p = om.Problem()
    p.model.add_subsystem('f', RotorDragAreaComp(source='chart'),
                          promotes=['*'])
    p.setup()
    p.set_val('X_chart', -0.10)
    p.set_val('CT_sigma', 0.083)
    p.set_val('mu', 0.3)
    p.set_val('sigma', 0.085)
    p.set_val('A_b', 240.0)
    p.run_model()

    assert p.get_val('f_M')[0] == pytest.approx(41.3, abs=0.1)


def test_rotor_drag_area_is_negative_in_level_flight():
    """p. 244 note: for level flight f_M = -(f + H_T/q). The rotor propels
    itself, so its equivalent flat plate area is negative."""
    f, H_T, q, CT, mu, sigma, A_b = 19.5, 59.0, 45.2, 0.085, 0.3, 0.085, 240.0
    total = f + H_T / q

    p = om.Problem()
    p.model.add_subsystem('f', RotorDragAreaComp(source='chart'),
                          promotes=['*'])
    p.setup()
    p.set_val('X_chart', total / A_b + sigma * CT ** 2 / mu ** 4)
    p.set_val('CT_sigma', CT)
    p.set_val('mu', mu)
    p.set_val('sigma', sigma)
    p.set_val('A_b', A_b)
    p.run_model()

    assert p.get_val('f_M')[0] == pytest.approx(-total, rel=1e-6)


def test_propulsive_balance_matches_cases_7_and_8():
    """p. 244 and p. 246: the same 2,764 lb, once as gravity and once as
    propeller thrust."""
    inputs = dict(q=45.2, f=19.5, f_M=41.3, H_T=16.0)

    dive = om.Problem()
    dive.model.add_subsystem('b', PropulsiveBalanceComp(mode='dive'),
                             promotes=['*'])
    dive.setup()
    for name, value in {**inputs, 'GW': 20000.0, 'mu': 0.3,
                        'V_tip': 650.0}.items():
        dive.set_val(name, value)
    dive.run_model()

    auxiliary = om.Problem()
    auxiliary.model.add_subsystem('b', PropulsiveBalanceComp(mode='auxiliary'),
                                  promotes=['*'])
    auxiliary.setup()
    for name, value in inputs.items():
        auxiliary.set_val(name, value)
    auxiliary.run_model()

    assert dive.get_val('F_prop')[0] == pytest.approx(2764.0, abs=2.0)
    assert dive.get_val('gamma_D', units='deg')[0] == pytest.approx(7.9,
                                                                   abs=0.1)
    assert auxiliary.get_val('T_aux')[0] == pytest.approx(
        dive.get_val('F_prop')[0], rel=1e-8)


def test_climb_drag_area_matches_case_6():
    """p. 242 step dd."""
    p = om.Problem()
    p.model.add_subsystem('c', ClimbDragAreaComp(), promotes=['*'])
    p.setup()
    for name, value in (('f', 20.6), ('H_T', 139.0), ('q', 45.2),
                        ('gamma_fp', np.deg2rad(4.9)), ('GW', 20000.0)):
        p.set_val(name, value)
    p.run_model()

    assert p.get_val('f_climb')[0] == pytest.approx(61.6, abs=0.1)


# -------------------------------------------------------- Table 3.5 case 9
def test_wall_correction_matches_case_9():
    """p. 247-248: chi = 86 deg, delta alpha_TPP = 0.6 deg."""
    p = om.Problem()
    p.model.add_subsystem('w', TunnelWallCorrectionComp(), promotes=['*'])
    p.setup()
    p.set_val('alpha_TPP_uncorr', np.deg2rad(5.3))
    p.set_val('CT_sigma', 0.105)
    p.set_val('mu', 0.3)
    p.set_val('sigma', 0.062)
    p.set_val('area_ratio', DISC_AREA / 3200.0)
    p.set_val('delta_WL', -0.54)
    p.run_model()

    assert p.get_val('chi', units='deg')[0] == pytest.approx(86.0, abs=0.2)
    assert p.get_val('delta_alpha_TPP', units='deg')[0] == pytest.approx(
        0.6, abs=0.05)
    assert p.get_val('alpha_TPP_corr', units='deg')[0] == pytest.approx(
        5.9, abs=0.05)


def test_wake_skew_is_regular_at_zero_thrust():
    """chi -> 90 deg as the wake lies in the disc plane. arctan2 would have
    been the natural spelling and is not complex-step safe."""
    p = om.Problem()
    p.model.add_subsystem('w', TunnelWallCorrectionComp(num_nodes=3),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('alpha_TPP_uncorr', np.zeros(3))
    p.set_val('CT_sigma', [0.105, 0.001, 0.0])
    p.set_val('mu', np.full(3, 0.3))
    p.run_model()

    chi = p.get_val('chi', units='deg')
    assert chi[2] == pytest.approx(90.0)
    assert chi[0] < chi[1] < chi[2]


def test_side_force_uses_the_uncorrected_angle():
    """p. 248 step n resolves forces along the TUNNEL axes, fixed by how the
    model was mounted, while the thrust belongs to the corrected condition.
    Using the corrected angle instead is a 17 % error."""
    def side_force(alpha_deg):
        p = om.Problem()
        p.model.add_subsystem('x', RotorSideForceComp(), promotes=['*'])
        p.setup()
        p.set_val('CT_sigma', 0.106)
        p.set_val('CH_sigma', -0.0033)
        p.set_val('alpha_TPP_uncorr', np.deg2rad(alpha_deg))
        p.run_model()
        return p.get_val('CXR_sigma')[0]

    assert side_force(5.3) == pytest.approx(-0.0065, abs=0.0001)
    assert abs(side_force(5.9) / side_force(5.3) - 1.0) > 0.15


def test_side_force_changes_sign_with_the_disc_tilt():
    """Tilted forward the rotor propels, tilted back it is dragged."""
    values = []
    for alpha_deg in (-5.0, 0.0, 5.3):
        p = om.Problem()
        p.model.add_subsystem('x', RotorSideForceComp(), promotes=['*'])
        p.setup()
        p.set_val('CT_sigma', 0.106)
        p.set_val('CH_sigma', -0.0033)
        p.set_val('alpha_TPP_uncorr', np.deg2rad(alpha_deg))
        p.run_model()
        values.append(p.get_val('CXR_sigma')[0])

    assert values[0] > values[1] > 0.0 > values[2]


@pytest.mark.slow
def test_tunnel_group_coning_is_connected():
    """Regression. G2 takes a_0 as an input and ClosedFormRotorGroup does not,
    so swapping the rotor left a_0 unconnected at its default of 1 RADIAN. The
    rotor ran with 57.3 deg of coning, A_1 came out at -20.9 deg instead of
    -2.6, and nothing failed: residuals read 1e-13 throughout."""
    p = build_tunnel('numerical')
    p.run_model()

    assert p.get_val('a0', units='deg')[0] == pytest.approx(5.96, abs=0.5)


@pytest.mark.parametrize('name, tolerance', [('CT_sigma', 0.10),
                                             ('CXR_sigma', 0.35),
                                             ('A_1', 0.30), ('B_1', 0.15)])
@pytest.mark.slow
def test_case_9_against_the_measurement(name, tolerance):
    """p. 249, test 276 run 3 of reference 3.27. The only measured comparison
    in the chapter. C_Q/sigma is excluded: it sits a factor of 1.7 high for the
    documented Mach drag rise reason, and the induced part is negative here so
    the torque is a small difference of cancelling terms."""
    p = build_tunnel('numerical')
    p.run_model()

    units = 'deg' if name in ('A_1', 'B_1') else None
    assert p.get_val(name, units=units)[0] == pytest.approx(
        MEASURED[name], rel=tolerance)


@pytest.mark.slow
def test_the_numerical_rotor_beats_the_closed_form_on_the_tunnel_case():
    """A mean c_d cannot describe this run. At positive shaft angle the flow
    is upward through the disc, the induced torque is NEGATIVE, and the closed
    form returns C_Q/sigma = -0.0014 against +0.0040 measured -- it has the
    rotor extracting energy. G2 computes the drag element by element."""
    closed = build_tunnel('closed_form')
    closed.run_model()
    numerical = build_tunnel('numerical')
    numerical.run_model()

    assert closed.get_val('CQ_sigma')[0] < 0.0 < numerical.get_val(
        'CQ_sigma')[0]
    for name in ('CT_sigma', 'CXR_sigma', 'B_1'):
        units = 'deg' if name == 'B_1' else None
        measured = MEASURED[name]
        assert abs(numerical.get_val(name, units=units)[0] - measured) < \
            abs(closed.get_val(name, units=units)[0] - measured), name


# ----------------------------------------------------------------- gradients
def test_chart_and_drag_area_partials():
    rng = np.random.default_rng(11)
    problems = [
        (ChartParameterComp(num_nodes=3, mode='from_inflow'),
         dict(lambda_p=[-0.03, 0.016, -0.05], CT_sigma=[0.085, 0.083, 0.097],
              mu=[0.3, 0.3, 0.45])),
        (ChartParameterComp(num_nodes=3, mode='to_inflow'),
         dict(f_Ab=0.081, sigma=0.0849, CT_sigma=[0.085, 0.083, 0.097],
              mu=[0.3, 0.3, 0.45])),
        (RotorDragAreaComp(num_nodes=3, source='inflow'),
         dict(lambda_p=[-0.03, 0.016, -0.05], CT_sigma=[0.085, 0.083, 0.097],
              mu=[0.3, 0.3, 0.45], sigma=0.085, A_b=240.0)),
        (PropulsiveBalanceComp(num_nodes=3, mode='dive'),
         dict(q=[45.2, 60.0, 80.0], f=[19.5, 20.0, 21.0],
              f_M=[41.3, 10.0, -20.0], H_T=[16.0, 20.0, 25.0], GW=20000.0,
              mu=[0.3, 0.35, 0.4], V_tip=650.0)),
        (ClimbDragAreaComp(num_nodes=3),
         dict(f=[20.6, 19.5, 21.0], H_T=[139.0, 16.0, 60.0],
              q=[45.2, 45.2, 60.0], gamma_fp=[0.0855, -0.14, 0.02],
              GW=20000.0)),
        (TunnelWallCorrectionComp(num_nodes=3),
         dict(alpha_TPP_uncorr=[0.09, 0.05, -0.02],
              CT_sigma=[0.105, 0.08, 0.06], mu=[0.3, 0.35, 0.25],
              sigma=0.062, area_ratio=0.57, delta_WL=-0.54)),
        (RotorSideForceComp(num_nodes=3),
         dict(CT_sigma=[0.106, 0.08, 0.12], CH_sigma=[-0.0033, 0.001, -0.005],
              alpha_TPP_uncorr=[0.0925, -0.05, 0.15])),
    ]

    for component, values in problems:
        p = om.Problem()
        p.model.add_subsystem('c', component, promotes=['*'])
        p.setup(force_alloc_complex=True)
        for name, value in values.items():
            p.set_val(name, value)
        p.run_model()

        data = p.check_partials(method='cs', compact_print=True,
                                out_stream=None)
        checked = 0
        for key, entry in data['c'].items():
            for error in errors(entry):
                assert error < 1e-10, f'{type(component).__name__} {key}'
                checked += 1
        assert checked > 0, type(component).__name__


# ---------------------------------------------------------- vectorisation
def build_fixed_collective(num_nodes, mode, mu, theta_0_deg):
    from prouty.forward_flight.fixed_collective_trim_group import (
        FixedCollectiveTrimGroup)

    p = om.Problem()
    p.model.add_subsystem('g', FixedCollectiveTrimGroup(num_nodes=num_nodes,
                                                        mode=mode),
                          promotes=['*'])
    p.setup()
    p.final_setup()
    for name, value in dict(
            mu=mu, V_tip=650.0, rho=0.002377, A_b=240.0, sigma=0.084883,
            theta_0=np.deg2rad(theta_0_deg), theta_1=np.deg2rad(-10.0), a=6.0,
            gamma=8.05033, R=30.0, GW=20000.0, i_s=0.0, a1s=0.0, l_T_R=1.23,
            cd_bar=0.0100, delta_3=np.deg2rad(-30.0), B=0.97,
            x_0=0.15).items():
        p.set_val(name, value)
    return p


@pytest.mark.slow
def test_fixed_collective_vectorises_without_coupling_the_nodes():
    """Three conditions in one solve must give exactly what three separate
    solves give. A group can converge and still be wrong here, by leaking one
    node's state into another through a scalar promoted where a vector was
    meant."""
    mu = [0.28, 0.30, 0.32]
    theta_0 = [12.0, 13.0, 14.0]

    together = build_fixed_collective(3, 'dive', mu, theta_0)
    together.run_model()
    assert np.abs(together.get_val('res_lift')).max() < 1e-10

    values = {}
    for i in range(3):
        p = build_fixed_collective(1, 'dive', mu[i], theta_0[i])
        p.run_model()
        for name in ('CT_sigma', 'alpha_F', 'f_M', 'gamma_D', 'R_D', 'hp_M'):
            values.setdefault(name, []).append(p.get_val(name)[0])

    for name, scalar in values.items():
        assert np.abs(together.get_val(name) - np.array(scalar)).max() < 1e-10, \
            name


@pytest.mark.slow
def test_wind_tunnel_vectorises_with_the_numerical_rotor():
    """Same check on the heavier group, where each node carries its own G2
    disc and its own nested Newton."""
    mu = [0.28, 0.30, 0.32]
    theta_75 = [8.0, 9.0, 10.0]
    alpha_s = [4.0, 5.0, 6.0]

    def build(num_nodes, index=None):
        pick = lambda seq: seq if index is None else seq[index]
        p = om.Problem()
        p.model.add_subsystem('g', WindTunnelRotorGroup(
            num_nodes=num_nodes, rotor='numerical', grid=(12, 15)),
            promotes=['*'])
        p.setup()
        p.final_setup()
        for name, value in {**TUNNEL,
                            'mu': pick(mu),
                            'theta_75': np.deg2rad(pick(theta_75)),
                            'alpha_s': np.deg2rad(pick(alpha_s))}.items():
            try:
                p.set_val(name, value)
            except KeyError:
                pass
        p.run_model()
        return p

    together = build(3)
    assert np.abs(together.get_val('res_CT')).max() < 1e-10

    values = {}
    for i in range(3):
        p = build(1, i)
        for name in ('CT_sigma', 'CQ_sigma', 'CXR_sigma', 'chi'):
            values.setdefault(name, []).append(p.get_val(name)[0])

    for name, scalar in values.items():
        assert np.abs(together.get_val(name) - np.array(scalar)).max() < 1e-12, \
            name


def test_fixed_collective_totals():
    """Analytic totals through the vertical-balance Newton, twenty pairs.

    This group's rotor is the closed form, whose derivatives are smooth
    everywhere, so the agreement is machine level -- unlike the numerical
    rotor, see validation_forward_flight section 5.
    """
    p = build_fixed_collective(1, 'dive', 0.3, 13.0)
    p.run_model()

    data = p.check_totals(
        of=['CT_sigma', 'gamma_D', 'R_D', 'hp_M', 'f_M'],
        wrt=['theta_0', 'mu', 'GW', 'cd_bar'], method='fd', step=1e-6,
        form='central', compact_print=True, out_stream=None)

    for key, entry in data.items():
        analytic = np.atleast_2d(entry['J_fwd'])[0, 0]
        difference = np.atleast_2d(entry['J_fd'])[0, 0]
        assert abs(analytic - difference) <= 1e-5 * max(abs(difference),
                                                        1e-4), key
    assert len(data) == 20


def test_wind_tunnel_totals_with_the_closed_form_rotor():
    """The wall correction loop differentiates cleanly; what does not is the
    airfoil model inside G2, not the nesting."""
    p = build_tunnel('closed_form')
    p.run_model()

    data = p.check_totals(of=['CT_sigma', 'CQ_sigma', 'CXR_sigma', 'chi'],
                          wrt=['theta_75', 'alpha_s', 'mu', 'delta_WL'],
                          method='fd', step=1e-6, form='central',
                          compact_print=True, out_stream=None)

    for key, entry in data.items():
        analytic = np.atleast_2d(entry['J_fwd'])[0, 0]
        difference = np.atleast_2d(entry['J_fd'])[0, 0]
        assert abs(analytic - difference) <= 1e-6 * max(abs(difference),
                                                        1e-4), key
    assert len(data) == 16
