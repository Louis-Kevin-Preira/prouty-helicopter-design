"""Chapter 7 -- flapping in hover including the effect of hinge offset,
p. 455-462."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.assert_utils import assert_near_equal

from prouty.flapping import HoverFlappingGroup
from prouty.flapping.accel_coupling_comp import AccelCouplingComp
from prouty.flapping.blade_inertia_comp import BladeInertiaComp
from prouty.flapping.blade_time_constant_comp import BladeTimeConstantComp
from prouty.flapping.flap_damping_comp import FlapDampingComp
from prouty.flapping.flap_frequency_comp import FlapFrequencyComp
from prouty.flapping.lock_number_comp import LockNumberComp
from prouty.flapping.phase_angle_comp import PhaseAngleComp


# ------------------------------------------------------------------------
# blade_inertia_comp
# ------------------------------------------------------------------------

R_EX_blade_inertia, E_OVER_R_EX_blade_inertia, I_B_EX_blade_inertia = 30.0, 0.05, 2870.0


def _run_blade_inertia(mode, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', BladeInertiaComp(input_mode=mode),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_uniform_mass_closed_form():
    """I_b and M_b/g follow the p. 457 integrals for uniform mass."""
    m, R, x = 0.32, 30.0, 0.05
    p = _run_blade_inertia('m', m=m, R=R, e_over_R=x)

    assert_near_equal(p.get_val('I_b')[0], m * R ** 3 * (1 - x) ** 3 / 3, 1e-12)
    assert_near_equal(p.get_val('M_b_over_g')[0],
                      m * R ** 2 * (1 - x) ** 2 / 2, 1e-12)
    assert_near_equal(p.get_val('e')[0], x * R, 1e-12)


def test_numerical_integration_of_definitions():
    """Closed forms match a direct quadrature of the p. 456 integrals."""
    m, R, x = 0.32, 30.0, 0.05
    rp = np.linspace(0.0, R * (1 - x), 200001)
    p = _run_blade_inertia('m', m=m, R=R, e_over_R=x)

    assert_near_equal(p.get_val('I_b')[0], np.trapezoid(m * rp ** 2, rp), 1e-8)
    assert_near_equal(p.get_val('M_b_over_g')[0],
                      np.trapezoid(m * rp, rp), 1e-8)


def test_two_modes_are_consistent():
    """Mode 'I_b' inverts mode 'm' exactly."""
    m, R, x = 0.32, 30.0, 0.05
    ref = _run_blade_inertia('m', m=m, R=R, e_over_R=x)
    inv = _run_blade_inertia('I_b', I_b_ref=ref.get_val('I_b'), R=R, e_over_R=x)

    assert_near_equal(inv.get_val('M_b_over_g')[0],
                      ref.get_val('M_b_over_g')[0], 1e-12)


def test_frequency_ratio_relation():
    """e*(M_b/g)/I_b reproduces the p. 457 grouping (3/2)(e/R)/(1-e/R)."""
    p = _run_blade_inertia('I_b', I_b_ref=I_B_EX_blade_inertia, R=R_EX_blade_inertia, e_over_R=E_OVER_R_EX_blade_inertia)
    ratio = p.get_val('e')[0] * p.get_val('M_b_over_g')[0] / p.get_val('I_b')[0]

    expected = 1.5 * E_OVER_R_EX_blade_inertia / (1 - E_OVER_R_EX_blade_inertia)
    assert_near_equal(ratio, expected, 1e-12)
    # p. 457: the resulting first flapping frequency ratio is 1.04
    assert_near_equal(np.sqrt(1 + ratio), 1.0387, 1e-3)


def test_weight_moment_units():
    """M_b = g * (M_b/g), in ft-lbf."""
    p = _run_blade_inertia('m', m=0.32, R=30.0, e_over_R=0.05)
    assert_near_equal(p.get_val('M_b')[0],
                      32.174 * p.get_val('M_b_over_g')[0], 1e-12)


def test_zero_offset_limit():
    """e/R = 0 recovers the classical mR^3/3 and mR^2/2."""
    p = _run_blade_inertia('m', m=0.32, R=30.0, e_over_R=0.0)
    assert_near_equal(p.get_val('I_b')[0], 0.32 * 30.0 ** 3 / 3, 1e-12)
    assert_near_equal(p.get_val('M_b_over_g')[0], 0.32 * 30.0 ** 2 / 2, 1e-12)
    assert_near_equal(p.get_val('e')[0], 0.0, 1e-12)



@pytest.mark.parametrize('mode', ['m', 'I_b'])
def test_partials_blade_inertia(mode):
    src = {'m': 0.32} if mode == 'm' else {'I_b_ref': I_B_EX_blade_inertia}
    p = _run_blade_inertia(mode, R=R_EX_blade_inertia, e_over_R=E_OVER_R_EX_blade_inertia, **src)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------------
# lock_number_comp
# ------------------------------------------------------------------------

C_EX_lock_number, R_EX_lock_number, I_B_EX_lock_number, RHO_SL_lock_number, A_EX_lock_number = 2.0, 30.0, 2870.0, 0.002378, 6.0


def _run_lock_number(mode='gamma', nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', LockNumberComp(num_nodes=nn, mode=mode),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_definition_lock_number():
    p = _run_lock_number(c=C_EX_lock_number, R=R_EX_lock_number, rho=RHO_SL_lock_number, a=A_EX_lock_number, I_b=I_B_EX_lock_number)
    expected = C_EX_lock_number * RHO_SL_lock_number * A_EX_lock_number * R_EX_lock_number ** 4 / I_B_EX_lock_number
    assert_near_equal(p.get_val('gamma')[0], expected, 1e-12)


def test_example_helicopter_anchor_lock_number():
    """Appendix A quotes gamma = 8.1 alongside I_b = 2870 slug.ft2.

    The pair is internally consistent only to ~0.6 %: gamma = 8.1 implies
    I_b = 2853.6. Both values are rounded in the data sheet.
    """
    p = _run_lock_number(c=C_EX_lock_number, R=R_EX_lock_number, rho=RHO_SL_lock_number, a=A_EX_lock_number, I_b=I_B_EX_lock_number)
    assert_near_equal(p.get_val('gamma')[0], 8.1, 1e-2)


def test_inverse_mode_round_trip():
    fwd = _run_lock_number(c=C_EX_lock_number, R=R_EX_lock_number, rho=RHO_SL_lock_number, a=A_EX_lock_number, I_b=I_B_EX_lock_number)
    inv = _run_lock_number('I_b', c=C_EX_lock_number, R=R_EX_lock_number, rho=RHO_SL_lock_number, a=A_EX_lock_number,
               gamma=fwd.get_val('gamma'))
    assert_near_equal(inv.get_val('I_b')[0], I_B_EX_lock_number, 1e-12)


def test_lift_curve_slope_units_are_converted():
    """a supplied in 1/deg must give the same gamma as 6.0 1/rad."""
    ref = _run_lock_number(c=C_EX_lock_number, R=R_EX_lock_number, rho=RHO_SL_lock_number, a=A_EX_lock_number, I_b=I_B_EX_lock_number)

    p = om.Problem()
    p.model.add_subsystem('comp', LockNumberComp(), promotes=['*'])
    p.setup()
    p.set_val('c', C_EX_lock_number)
    p.set_val('R', R_EX_lock_number)
    p.set_val('rho', RHO_SL_lock_number)
    p.set_val('a', np.degrees(1.0) ** -1 * A_EX_lock_number, units='1/deg')
    p.set_val('I_b', I_B_EX_lock_number)
    p.run_model()

    assert_near_equal(p.get_val('gamma')[0], ref.get_val('gamma')[0], 1e-12)


def test_vectorized_over_density():
    nn = 3
    rho = np.array([0.002378, 0.002048, 0.001756])
    p = _run_lock_number(nn=nn, c=C_EX_lock_number, R=R_EX_lock_number, rho=rho, a=np.full(nn, A_EX_lock_number), I_b=I_B_EX_lock_number)

    expected = C_EX_lock_number * rho * A_EX_lock_number * R_EX_lock_number ** 4 / I_B_EX_lock_number
    assert_near_equal(p.get_val('gamma'), expected, 1e-12)


def test_inverse_mode_rejects_multiple_nodes():
    with pytest.raises(ValueError, match='num_nodes=1'):
        _run_lock_number('I_b', nn=4)



@pytest.mark.parametrize('mode,nn', [('gamma', 1), ('gamma', 4), ('I_b', 1)])
def test_partials_lock_number(mode, nn):
    extra = {'gamma': 8.1} if mode == 'I_b' else {'I_b': I_B_EX_lock_number}
    p = _run_lock_number(mode, nn=nn, c=C_EX_lock_number, R=R_EX_lock_number,
             rho=np.linspace(0.0018, 0.00238, nn),
             a=np.linspace(5.7, 6.1, nn), **extra)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------------
# flap_frequency_comp
# ------------------------------------------------------------------------

R_EX_flap_frequency, E_OVER_R_EX_flap_frequency, I_B_EX_flap_frequency = 30.0, 0.05, 2870.0


def _run_flap_frequency(mode='frequency', **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', FlapFrequencyComp(mode=mode), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _chained_flap_frequency(e_over_R, R=R_EX_flap_frequency, I_b=I_B_EX_flap_frequency):
    """BladeInertiaComp -> FlapFrequencyComp, as wired in HoverFlappingGroup."""
    p = om.Problem()
    p.model.add_subsystem('inertia', BladeInertiaComp(input_mode='I_b'),
                          promotes=['*'])
    p.model.add_subsystem('freq', FlapFrequencyComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('I_b_ref', I_b)
    p.set_val('R', R)
    p.set_val('e_over_R', e_over_R)
    p.run_model()
    return p


def test_definition_flap_frequency():
    e, M, I = 1.5, 151.05, 2870.0
    p = _run_flap_frequency(e=e, M_b_over_g=M, I_b=I)
    assert_near_equal(p.get_val('omega_n_ratio')[0], np.sqrt(1 + e * M / I), 1e-12)


def test_matches_uniform_mass_closed_form():
    """Chained with BladeInertiaComp, recovers sqrt(1 + 1.5x/(1-x)) (p. 457)."""
    for x in (0.0, 0.02, 0.05, 0.13):
        p = _chained_flap_frequency(x)
        expected = np.sqrt(1.0 + 1.5 * x / (1.0 - x))
        assert_near_equal(p.get_val('omega_n_ratio')[0], expected, 1e-12)


def test_example_helicopter_anchor_flap_frequency():
    """p. 457: e/R = 0.05 gives a first flapping frequency ratio of 1.04.

    The exact value is 1.03872; the book quotes two decimals.
    """
    p = _chained_flap_frequency(E_OVER_R_EX_flap_frequency)
    assert_near_equal(p.get_val('omega_n_ratio')[0], 1.04, 2e-3)


def test_zero_offset_is_resonance():
    p = _chained_flap_frequency(0.0)
    assert_near_equal(p.get_val('omega_n_ratio')[0], 1.0, 1e-12)


def test_inverse_matches_book_form():
    nu = 1.04
    p = _run_flap_frequency('e_over_R', omega_n_ratio=nu)
    expected = 2 * (nu ** 2 - 1) / (1 + 2 * nu ** 2)
    assert_near_equal(p.get_val('e_over_R_eff')[0], expected, 1e-12)


def test_round_trip():
    """(e/R) -> omega_n/Omega -> (e/R)_eff returns the original offset."""
    for x in (0.02, 0.05, 0.13):
        nu = _chained_flap_frequency(x).get_val('omega_n_ratio')
        p = _run_flap_frequency('e_over_R', omega_n_ratio=nu)
        assert_near_equal(p.get_val('e_over_R_eff')[0], x, 1e-12)


def test_hingeless_rotor_order_of_magnitude():
    """p. 477 quotes ~13 % effective offset for the AH-56A hingeless rotor."""
    p = _run_flap_frequency('e_over_R', omega_n_ratio=1.11)
    assert 0.10 < p.get_val('e_over_R_eff')[0] < 0.16



@pytest.mark.parametrize('mode', ['frequency', 'e_over_R'])
def test_partials_flap_frequency(mode):
    ivc = ({'omega_n_ratio': 1.04} if mode == 'e_over_R'
           else {'e': 1.5, 'M_b_over_g': 151.05, 'I_b': I_B_EX_flap_frequency})
    p = _run_flap_frequency(mode, **ivc)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_chained_totals_flap_frequency():
    """Analytic totals through both components against complex step.

    d(omega_n/Omega)/d(I_b_ref) is exactly zero: with M_b/g taken from the
    uniform-mass relation, e (M_b/g) / I_b = (3/2)(e/R)/(1 - e/R) and the
    inertia cancels out.
    """
    p = _chained_flap_frequency(E_OVER_R_EX_flap_frequency)
    data = p.check_totals(of=['omega_n_ratio'], wrt=['e_over_R', 'R', 'I_b_ref'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-12)

    assert data[('omega_n_ratio', 'I_b_ref')]['J_fd'] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------------
# flap_damping_comp
# ------------------------------------------------------------------------

GAMMA_EX_flap_damping, E_OVER_R_EX_flap_damping, NU_EX, I_B_EX_flap_damping = 8.1, 0.05, 1.04, 2870.0


OMEGA_EX_flap_damping = 650.0 / 30.0  # rad/s, from tip speed and radius


def _run_flap_damping(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', FlapDampingComp(num_nodes=nn), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _default(nn=1):
    return _run_flap_damping(nn, I_b=I_B_EX_flap_damping, e_over_R=E_OVER_R_EX_flap_damping, omega_n_ratio=NU_EX,
                gamma=np.full(nn, GAMMA_EX_flap_damping), Omega=np.full(nn, OMEGA_EX_flap_damping))


def test_damping_ratio_anchor():
    """p. 459: gamma = 8.1, e/R = 0.05, nu = 1.04 gives c/c_crit = 0.42.

    The exact value is 0.42431; the book quotes two decimals.
    """
    p = _default()
    assert_near_equal(p.get_val('zeta')[0], 0.42, 1.2e-2)


def test_zeta_equals_c_over_c_crit():
    """The p. 459 closed form agrees with the ratio of the p. 458 quantities."""
    p = _default()
    ratio = p.get_val('c_damp')[0] / p.get_val('c_crit')[0]
    assert_near_equal(p.get_val('zeta')[0], ratio, 1e-12)


def test_damping_matches_quadrature_of_hinge_moment():
    """c_damp reproduces the p. 458 integral for arbitrary blade data.

        M_A = int_0^(R-e) r' (rho/2) a c r' (r' + e) Omega dr'
    """
    rho, a, chord, R, x = 0.002378, 6.0, 2.0, 30.0, 0.05
    I_b = 2870.0
    gamma = chord * rho * a * R ** 4 / I_b

    p = _run_flap_damping(I_b=I_b, e_over_R=x, omega_n_ratio=1.0,
             gamma=gamma, Omega=OMEGA_EX_flap_damping)

    rp = np.linspace(0.0, R * (1 - x), 400001)
    integral = np.trapezoid(rp ** 2 * (rp + x * R), rp)
    expected = 0.5 * rho * a * chord * OMEGA_EX_flap_damping * integral

    assert_near_equal(p.get_val('c_damp')[0], expected, 1e-8)


def test_collapsed_shape_factor():
    """(1-x)^4 (1+x/3)/(1-x) as printed equals the collapsed (1-x)^3 (1+x/3)."""
    for x in (0.0, 0.05, 0.13):
        printed = (1 - x) ** 4 * (1 + x / 3) / (1 - x)
        p = _run_flap_damping(I_b=1.0, e_over_R=x, omega_n_ratio=1.0, gamma=8.0, Omega=1.0)
        assert_near_equal(p.get_val('c_damp')[0], printed, 1e-12)


def test_critical_damping_definition():
    """p. 458: c_crit = 2 I_b omega_n."""
    p = _default()
    assert_near_equal(p.get_val('omega_n')[0], NU_EX * OMEGA_EX_flap_damping, 1e-12)
    assert_near_equal(p.get_val('c_crit')[0],
                      2 * I_B_EX_flap_damping * p.get_val('omega_n')[0], 1e-12)


def test_zeta_independent_of_omega_and_inertia():
    """Omega and I_b cancel between c_damp and c_crit."""
    ref = _default().get_val('zeta')[0]
    for I_b, Om in ((I_B_EX_flap_damping, OMEGA_EX_flap_damping), (500.0, OMEGA_EX_flap_damping), (I_B_EX_flap_damping, 12.0)):
        p = _run_flap_damping(I_b=I_b, e_over_R=E_OVER_R_EX_flap_damping, omega_n_ratio=NU_EX,
                 gamma=GAMMA_EX_flap_damping, Omega=Om)
        assert_near_equal(p.get_val('zeta')[0], ref, 1e-12)


def test_zero_offset_reduces_to_gamma_over_16():
    """With e/R = 0 and nu = 1, the damping ratio is the universal gamma/16."""
    p = _run_flap_damping(I_b=I_B_EX_flap_damping, e_over_R=0.0, omega_n_ratio=1.0,
             gamma=GAMMA_EX_flap_damping, Omega=OMEGA_EX_flap_damping)
    assert_near_equal(p.get_val('zeta')[0], GAMMA_EX_flap_damping / 16.0, 1e-12)


def test_vectorized_flap_damping():
    nn = 3
    gamma = np.array([8.1, 6.9, 5.9])
    Omega = np.array([21.67, 21.67, 20.0])
    p = _run_flap_damping(nn, I_b=I_B_EX_flap_damping, e_over_R=E_OVER_R_EX_flap_damping, omega_n_ratio=NU_EX,
             gamma=gamma, Omega=Omega)

    K = (1 - E_OVER_R_EX_flap_damping) ** 3 * (1 + E_OVER_R_EX_flap_damping / 3)
    assert_near_equal(p.get_val('zeta'), gamma * K / (16 * NU_EX), 1e-12)
    assert_near_equal(p.get_val('c_damp'), I_B_EX_flap_damping * gamma * Omega * K / 8, 1e-12)



@pytest.mark.parametrize('nn', [1, 4])
def test_partials_flap_damping(nn):
    p = _run_flap_damping(nn, I_b=I_B_EX_flap_damping, e_over_R=E_OVER_R_EX_flap_damping, omega_n_ratio=NU_EX,
             gamma=np.linspace(5.9, 8.1, nn), Omega=np.linspace(19.0, 21.7, nn))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------------
# phase_angle_comp
# ------------------------------------------------------------------------

NU_BOOK, ZETA_BOOK = 1.04, 0.42


NU_EXACT, ZETA_EXACT = 1.038724, 0.424308


def _run_phase_angle(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', PhaseAngleComp(num_nodes=nn), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _phi_deg(nu, zeta):
    return _run_phase_angle(zeta=zeta, omega_n_ratio=nu).get_val('phi', units='deg')[0]


def test_matches_printed_arccos_form():
    """atan2 formulation is identical to the arccos written on p. 459."""
    for nu, zeta in ((1.02, 0.30), (1.04, 0.42), (1.15, 0.55), (1.30, 0.20)):
        f = nu ** 2 - 1.0
        printed = np.arccos(f / np.sqrt(f ** 2 + 4 * zeta ** 2 * nu ** 2))
        assert_near_equal(_run_phase_angle(zeta=zeta, omega_n_ratio=nu).get_val('phi')[0],
                          printed, 1e-13)


def test_example_helicopter_anchor_phase_angle():
    """p. 459 reads 84.8 deg off Figure 7.3 for the example helicopter.

    The closed form gives 84.66 deg with the rounded inputs (1.04, 0.42) and
    84.88 deg with the unrounded ones, so the chart reading sits between them.
    """
    assert_near_equal(_phi_deg(NU_BOOK, ZETA_BOOK), 84.8, 2e-3)
    assert_near_equal(_phi_deg(NU_EXACT, ZETA_EXACT), 84.8, 2e-3)


def test_no_offset_is_quarter_cycle():
    """nu = 1 means resonance: the response lags exactly 90 deg (p. 447)."""
    assert_near_equal(_phi_deg(1.0, 0.42), 90.0, 1e-12)


def test_offset_reduces_lag_below_90():
    """p. 459: with hinge offset the phase angle is lower than 90 deg."""
    for nu in (1.02, 1.05, 1.10, 1.20):
        assert _phi_deg(nu, 0.42) < 90.0


def test_monotonic_in_frequency_ratio():
    angles = [_phi_deg(nu, 0.42) for nu in (1.0, 1.05, 1.10, 1.20, 1.35)]
    assert all(b < a for a, b in zip(angles, angles[1:]))


def test_limits():
    """Vanishing damping drives phi to 0; heavy damping restores 90 deg."""
    assert _phi_deg(1.10, 1e-6) == pytest.approx(0.0, abs=1e-3)
    assert _phi_deg(1.10, 1e6) == pytest.approx(90.0, abs=1e-3)


def test_stays_in_first_quadrant_range():
    """atan2 with a positive sine argument keeps phi in (0, 180) deg."""
    for nu in (0.8, 1.0, 1.5):
        for zeta in (0.01, 0.4, 3.0):
            assert 0.0 < _phi_deg(nu, zeta) < 180.0


def test_soft_inplane_case_exceeds_90():
    """nu < 1 (no offset, added flapping spring absent) gives a lag above 90."""
    assert _phi_deg(0.9, 0.42) > 90.0


def test_vectorized_phase_angle():
    nn = 3
    zeta = np.array([0.4243, 0.3638, 0.3119])
    p = _run_phase_angle(nn, zeta=zeta, omega_n_ratio=NU_EXACT)
    expected = 0.5 * np.pi - np.arctan((NU_EXACT ** 2 - 1) / (2 * zeta * NU_EXACT))
    assert_near_equal(p.get_val('phi'), expected, 1e-13)



@pytest.mark.parametrize('nn', [1, 4])
def test_partials_phase_angle(nn):
    p = _run_phase_angle(nn, zeta=np.linspace(0.31, 0.43, nn), omega_n_ratio=NU_EXACT)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_partials_near_resonance():
    """nu = 1 makes the arccos argument zero; derivatives must stay finite."""
    p = _run_phase_angle(zeta=0.42, omega_n_ratio=1.0)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------------
# accel_coupling_comp
# ------------------------------------------------------------------------

GAMMA_EX_accel_coupling, E_OVER_R_EX_accel_coupling, I_B_EX_accel_coupling, R_EX_accel_coupling = 8.1, 0.05, 2870.0, 30.0


def _uniform_state(gamma, x):
    """nu and zeta of a uniform blade, from pp. 457 and 459."""
    nu = np.sqrt(1.0 + 1.5 * x / (1.0 - x))
    zeta = gamma / 16.0 * (1 - x) ** 3 * (1 + x / 3) / nu
    return nu, zeta


def _run_accel_coupling(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', AccelCouplingComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _for(gamma=GAMMA_EX_accel_coupling, x=E_OVER_R_EX_accel_coupling, r_over_R=0.75):
    nu, zeta = _uniform_state(gamma, x)
    return _run_accel_coupling(omega_n_ratio=nu, zeta=zeta, e_over_R=x, r_over_R=r_over_R)


def test_matches_closed_form_of_page_460():
    """-cot(phi) collapses to -(12/gamma)(e/R)/{[1+e/3R][1-e/R]^4}."""
    for gamma, x in ((8.1, 0.05), (6.0, 0.02), (10.0, 0.13)):
        closed = -(12.0 / gamma) * x / ((1 + x / 3) * (1 - x) ** 4)
        assert_near_equal(_for(gamma, x).get_val('b1s_over_a1s')[0],
                          closed, 1e-13)


def test_matches_negative_cotangent_of_phase_angle():
    """Cross-check against PhaseAngleComp rather than reusing its output."""
    nu, zeta = _uniform_state(GAMMA_EX_accel_coupling, E_OVER_R_EX_accel_coupling)

    p = om.Problem()
    p.model.add_subsystem('phase', PhaseAngleComp(), promotes=['*'])
    p.setup()
    p.set_val('omega_n_ratio', nu)
    p.set_val('zeta', zeta)
    p.run_model()

    expected = -1.0 / np.tan(p.get_val('phi')[0])
    assert_near_equal(_for().get_val('b1s_over_a1s')[0], expected, 1e-12)


def test_complete_form_anchor():
    """Complete form for the example helicopter, pp. 459-461."""
    p = _for()
    assert_near_equal(p.get_val('b1s_over_a1s')[0], -0.08945, 1e-3)
    assert_near_equal(p.get_val('dalpha_over_a1s')[0], 0.08349, 1e-3)


def test_printed_values_come_from_the_reduced_form():
    """Documents the pp. 460-461 inconsistency rather than hiding it.

    Dropping (1-e/R)^4, as p. 461 does, reproduces the printed -0.07 and 0.07;
    keeping it, as p. 460's own equation requires, gives -0.089 and 0.083.
    """
    x, gamma = E_OVER_R_EX_accel_coupling, GAMMA_EX_accel_coupling
    reduced = -(12.0 / gamma) * x / (1 + x / 3)
    station = 1 - x / 0.75

    assert reduced == pytest.approx(-0.07, abs=5e-3)
    assert -reduced * station == pytest.approx(0.07, abs=5e-3)

    complete = _for().get_val('b1s_over_a1s')[0]
    assert complete / reduced == pytest.approx((1 - x) ** -4, rel=1e-12)


def test_station_factor():
    """Delta_alpha uses r'/(r'+e) = 1 - (e/R)/(r/R), p. 461."""
    p = _for(r_over_R=0.75)
    ratio = -p.get_val('dalpha_over_a1s')[0] / p.get_val('b1s_over_a1s')[0]
    assert_near_equal(ratio, (0.75 - 0.05) / 0.75, 1e-13)


def test_no_offset_means_no_coupling():
    """p. 461: with zero hinge offset the flapping needs no cyclic alpha."""
    p = _for(x=0.0)
    assert_near_equal(p.get_val('b1s_over_a1s')[0], 0.0, 1e-14)
    assert_near_equal(p.get_val('dalpha_over_a1s')[0], 0.0, 1e-14)


def test_coupling_grows_with_offset_and_falls_with_lock_number():
    offsets = [abs(_for(x=x).get_val('b1s_over_a1s')[0])
               for x in (0.01, 0.05, 0.10, 0.15)]
    assert all(b > a for a, b in zip(offsets, offsets[1:]))

    lock = [abs(_for(gamma=g).get_val('b1s_over_a1s')[0])
            for g in (4.0, 6.0, 8.1, 12.0)]
    assert all(b < a for a, b in zip(lock, lock[1:]))


def test_chained_from_blade_properties():
    """inertia -> frequency -> damping -> coupling, as wired in G1."""
    nn = 2
    p = om.Problem()
    m = p.model
    m.add_subsystem('inertia', BladeInertiaComp(input_mode='I_b'), promotes=['*'])
    m.add_subsystem('freq', FlapFrequencyComp(), promotes=['*'])
    m.add_subsystem('damp', FlapDampingComp(num_nodes=nn), promotes=['*'])
    m.add_subsystem('coupling', AccelCouplingComp(num_nodes=nn), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('I_b_ref', I_B_EX_accel_coupling)
    p.set_val('R', R_EX_accel_coupling)
    p.set_val('e_over_R', E_OVER_R_EX_accel_coupling)
    p.set_val('gamma', np.full(nn, GAMMA_EX_accel_coupling))
    p.set_val('Omega', np.full(nn, 650.0 / 30.0))
    p.run_model()

    closed = -(12.0 / GAMMA_EX_accel_coupling) * E_OVER_R_EX_accel_coupling \
        / ((1 + E_OVER_R_EX_accel_coupling / 3) * (1 - E_OVER_R_EX_accel_coupling) ** 4)
    assert_near_equal(p.get_val('b1s_over_a1s'), np.full(nn, closed), 1e-12)

    data = p.check_totals(of=['b1s_over_a1s', 'dalpha_over_a1s'],
                          wrt=['e_over_R', 'gamma', 'r_over_R'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-12)


def test_vectorized_accel_coupling():
    nn = 3
    nu, _ = _uniform_state(GAMMA_EX_accel_coupling, E_OVER_R_EX_accel_coupling)
    zeta = np.array([0.4248, 0.3638, 0.3119])
    p = _run_accel_coupling(nn, omega_n_ratio=nu, zeta=zeta, e_over_R=E_OVER_R_EX_accel_coupling)
    expected = -(nu ** 2 - 1) / (2 * zeta * nu)
    assert_near_equal(p.get_val('b1s_over_a1s'), expected, 1e-13)



@pytest.mark.parametrize('nn', [1, 4])
def test_partials_accel_coupling(nn):
    nu, _ = _uniform_state(GAMMA_EX_accel_coupling, E_OVER_R_EX_accel_coupling)
    p = _run_accel_coupling(nn, omega_n_ratio=nu, zeta=np.linspace(0.31, 0.43, nn),
             e_over_R=E_OVER_R_EX_accel_coupling, r_over_R=0.75)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------------
# blade_time_constant_comp
# ------------------------------------------------------------------------

GAMMA_EX_blade_time_constant, E_OVER_R_EX_blade_time_constant, I_B_EX_blade_time_constant, R_EX_blade_time_constant = 8.1, 0.05, 2870.0, 30.0


OMEGA_EX_blade_time_constant = 650.0 / 30.0


def _shape(x):
    return (1.0 - x) ** 3 * (1.0 + x / 3.0)


def _run_blade_time_constant(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', BladeTimeConstantComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _chained_blade_time_constant(nn=1, gamma=GAMMA_EX_blade_time_constant, x=E_OVER_R_EX_blade_time_constant, Omega=OMEGA_EX_blade_time_constant):
    """inertia -> frequency -> damping -> time constant, as wired in G1."""
    p = om.Problem()
    m = p.model
    m.add_subsystem('inertia', BladeInertiaComp(input_mode='I_b'), promotes=['*'])
    m.add_subsystem('freq', FlapFrequencyComp(), promotes=['*'])
    m.add_subsystem('damp', FlapDampingComp(num_nodes=nn), promotes=['*'])
    m.add_subsystem('tconst', BladeTimeConstantComp(num_nodes=nn), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('I_b_ref', I_B_EX_blade_time_constant)
    p.set_val('R', R_EX_blade_time_constant)
    p.set_val('e_over_R', x)
    p.set_val('gamma', np.full(nn, gamma))
    p.set_val('Omega', np.full(nn, Omega))
    p.run_model()
    return p


def test_definition_blade_time_constant():
    """p. 462: t_63 = I_b / (c/2)."""
    p = _run_blade_time_constant(I_b=I_B_EX_blade_time_constant, c_damp=54567.0, Omega=OMEGA_EX_blade_time_constant)
    assert_near_equal(p.get_val('t_63')[0], I_B_EX_blade_time_constant / (54567.0 / 2), 1e-13)
    assert_near_equal(p.get_val('psi_63')[0],
                      OMEGA_EX_blade_time_constant * p.get_val('t_63')[0], 1e-13)


def test_time_constant_closed_form():
    """t_63 = (16/(gamma Omega)) / [(1-e/R)^3 (1+e/3R)], p. 462."""
    for gamma, x in ((8.1, 0.05), (6.0, 0.0), (10.0, 0.13)):
        p = _chained_blade_time_constant(gamma=gamma, x=x)
        expected = 16.0 / (gamma * OMEGA_EX_blade_time_constant * _shape(x))
        assert_near_equal(p.get_val('t_63')[0], expected, 1e-12)


def test_azimuth_constant_closed_form():
    """psi_63 = 917 / [gamma (1-e/R)^3 (1+e/3R)] deg, p. 462.

    The printed 917 is 16 * 180/pi = 916.73 rounded.
    """
    for gamma, x in ((8.1, 0.05), (6.0, 0.0), (10.0, 0.13)):
        p = _chained_blade_time_constant(gamma=gamma, x=x)
        expected = np.degrees(16.0) / (gamma * _shape(x))
        assert_near_equal(p.get_val('psi_63', units='deg')[0], expected, 1e-12)


def test_example_helicopter_anchor_blade_time_constant():
    """p. 462: the example helicopter has an azimuth constant of 130 deg."""
    p = _chained_blade_time_constant()
    assert_near_equal(p.get_val('psi_63', units='deg')[0], 130.0, 2e-3)


def test_azimuth_constant_independent_of_omega_and_inertia():
    """Omega and I_b both cancel against c_damp."""
    ref = _chained_blade_time_constant().get_val('psi_63')[0]

    slow = _chained_blade_time_constant(Omega=15.0)
    assert_near_equal(slow.get_val('psi_63')[0], ref, 1e-12)
    assert slow.get_val('t_63')[0] > _chained_blade_time_constant().get_val('t_63')[0]

    p = _run_blade_time_constant(I_b=2 * I_B_EX_blade_time_constant, c_damp=2 * 54567.0, Omega=OMEGA_EX_blade_time_constant)
    q = _run_blade_time_constant(I_b=I_B_EX_blade_time_constant, c_damp=54567.0, Omega=OMEGA_EX_blade_time_constant)
    assert_near_equal(p.get_val('psi_63')[0], q.get_val('psi_63')[0], 1e-13)


def test_step_response_reaches_63_percent():
    """The definition is consistent with the p. 462 exponential."""
    p = _chained_blade_time_constant()
    t = p.get_val('t_63')[0]
    c_over_2I = 1.0 / t
    assert_near_equal(1.0 - np.exp(-c_over_2I * t), 1.0 - 1.0 / np.e, 1e-13)


def test_faster_response_with_higher_lock_number():
    times = [_chained_blade_time_constant(gamma=g).get_val('t_63')[0] for g in (4.0, 6.0, 8.1, 12.0)]
    assert all(b < a for a, b in zip(times, times[1:]))


def test_vectorized_blade_time_constant():
    nn = 3
    c_damp = np.array([54567.0, 46995.0, 40294.0])
    Omega = np.full(nn, OMEGA_EX_blade_time_constant)
    p = _run_blade_time_constant(nn, I_b=I_B_EX_blade_time_constant, c_damp=c_damp, Omega=Omega)

    assert_near_equal(p.get_val('t_63'), 2 * I_B_EX_blade_time_constant / c_damp, 1e-13)
    assert_near_equal(p.get_val('psi_63'), Omega * 2 * I_B_EX_blade_time_constant / c_damp, 1e-13)



@pytest.mark.parametrize('nn', [1, 4])
def test_partials_blade_time_constant(nn):
    p = _run_blade_time_constant(nn, I_b=I_B_EX_blade_time_constant, c_damp=np.linspace(40000.0, 55000.0, nn),
             Omega=np.linspace(19.0, 21.7, nn))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_chained_totals_blade_time_constant():
    p = _chained_blade_time_constant(nn=2)
    data = p.check_totals(of=['t_63', 'psi_63'],
                          wrt=['e_over_R', 'gamma', 'Omega', 'I_b_ref'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------------
# hover_flapping_group
# ------------------------------------------------------------------------

R_EX_hover_flapping_grp, C_EX_hover_flapping_grp, I_B_EX_hover_flapping_grp, E_OVER_R_EX_hover_flapping_grp = 30.0, 2.0, 2870.0, 0.05


A_EX_hover_flapping_grp, RHO_SL_hover_flapping_grp, OMEGA_EX_hover_flapping_grp = 6.0, 0.002378, 650.0 / 30.0


RHO_ALT = np.array([0.002378, 0.002048, 0.001756])


def _build(nn=1, blade_input='I_b', hinge_input='e_over_R', **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'hover', HoverFlappingGroup(num_nodes=nn, blade_input=blade_input,
                                    hinge_input=hinge_input),
        promotes=['*'])
    p.setup(force_alloc_complex=True)

    p.set_val('R', R_EX_hover_flapping_grp)
    p.set_val('c', C_EX_hover_flapping_grp)
    p.set_val('rho', np.full(nn, RHO_SL_hover_flapping_grp))
    p.set_val('a', np.full(nn, A_EX_hover_flapping_grp))
    p.set_val('Omega', np.full(nn, OMEGA_EX_hover_flapping_grp))
    if blade_input == 'I_b':
        p.set_val('I_b_ref', I_B_EX_hover_flapping_grp)
    if hinge_input == 'e_over_R':
        p.set_val('e_over_R', E_OVER_R_EX_hover_flapping_grp)
    for k, v in ivc.items():
        p.set_val(k, v)

    p.run_model()
    return p


def test_example_helicopter_book_anchors():
    """All the hover results quoted in pp. 457-462, in one pass.

    gamma comes out at 8.054 rather than the 8.1 of the data sheet, because
    I_b = 2870 and gamma = 8.1 are each rounded and are not quite consistent.
    Tolerances below absorb that plus the book's two-decimal printing.
    """
    p = _build()

    assert_near_equal(p.get_val('omega_n_ratio')[0], 1.04, 2e-3)     # p. 457
    assert_near_equal(p.get_val('zeta')[0], 0.42, 1.2e-2)            # p. 459
    assert_near_equal(p.get_val('phi', units='deg')[0], 84.8, 2e-3)  # p. 459
    assert_near_equal(p.get_val('psi_63', units='deg')[0], 130.0, 6e-3)  # p. 462


def test_cross_coupling_uses_the_complete_form():
    """pp. 460-461 print -0.07 and 0.07 from a form missing (1-e/R)^4.

    The group reproduces the complete form instead; the ratio between the two
    is exactly (1-e/R)^-4.
    """
    p = _build()
    gamma = p.get_val('gamma')[0]

    reduced = -(12.0 / gamma) * E_OVER_R_EX_hover_flapping_grp / (1 + E_OVER_R_EX_hover_flapping_grp / 3)
    complete = p.get_val('b1s_over_a1s')[0]

    assert_near_equal(complete / reduced, (1 - E_OVER_R_EX_hover_flapping_grp) ** -4, 1e-12)
    assert_near_equal(complete, -0.0900, 1e-3)
    assert_near_equal(p.get_val('dalpha_over_a1s')[0], 0.0840, 1e-3)


def test_blade_input_modes_agree():
    """Feeding m instead of I_b gives the same rotor when they match."""
    ref = _build()
    m_equiv = 3 * I_B_EX_hover_flapping_grp / (R_EX_hover_flapping_grp ** 3 * (1 - E_OVER_R_EX_hover_flapping_grp) ** 3)
    alt = _build(blade_input='m', m=m_equiv)

    for name in ('I_b', 'M_b_over_g', 'gamma', 'omega_n_ratio', 'zeta',
                 'phi', 'b1s_over_a1s', 't_63', 'psi_63'):
        assert_near_equal(alt.get_val(name)[0], ref.get_val(name)[0], 1e-11)


def test_hingeless_mode_is_the_inverse_of_the_articulated_one():
    """Feeding back omega_n/Omega recovers the same offset and results."""
    ref = _build()
    nu = ref.get_val('omega_n_ratio')[0]

    hingeless = _build(hinge_input='omega_n_ratio', omega_n_ratio=nu)

    assert_near_equal(hingeless.get_val('e_over_R')[0], E_OVER_R_EX_hover_flapping_grp, 1e-12)
    for name in ('gamma', 'zeta', 'phi', 'b1s_over_a1s', 'dalpha_over_a1s',
                 't_63', 'psi_63'):
        assert_near_equal(hingeless.get_val(name)[0],
                          ref.get_val(name)[0], 1e-11)


def test_hingeless_rotor_case():
    """A stiff hingeless blade: nu = 1.11 gives roughly 13 % offset, p. 477."""
    p = _build(hinge_input='omega_n_ratio', omega_n_ratio=1.11)
    assert 0.10 < p.get_val('e_over_R')[0] < 0.16
    assert p.get_val('phi', units='deg')[0] < 84.8


def test_zero_offset_recovers_the_articulated_reference():
    """e/R = 0: resonance, 90 deg lag, no cross-coupling, zeta = gamma/16."""
    p = _build(e_over_R=0.0)

    assert_near_equal(p.get_val('omega_n_ratio')[0], 1.0, 1e-13)
    assert_near_equal(p.get_val('phi', units='deg')[0], 90.0, 1e-12)
    assert_near_equal(p.get_val('b1s_over_a1s')[0], 0.0, 1e-13)
    assert_near_equal(p.get_val('dalpha_over_a1s')[0], 0.0, 1e-13)
    assert_near_equal(p.get_val('zeta')[0], p.get_val('gamma')[0] / 16, 1e-13)


def test_altitude_sweep_is_node_independent_where_it_should_be():
    """Only gamma-dependent quantities move with density."""
    p = _build(nn=3, rho=RHO_ALT)

    gamma = p.get_val('gamma')
    assert np.all(np.diff(gamma) < 0)

    # Node-independent by construction
    assert p.get_val('omega_n_ratio').size == 1
    assert_near_equal(p.get_val('c_crit'), np.full(3, p.get_val('c_crit')[0]),
                      1e-13)

    # Damping falls with density, the phase lag with it
    assert np.all(np.diff(p.get_val('zeta')) < 0)
    assert np.all(np.diff(p.get_val('phi')) < 0)
    assert np.all(np.diff(np.abs(p.get_val('b1s_over_a1s'))) > 0)
    assert np.all(np.diff(p.get_val('psi_63')) > 0)


def test_group_is_feed_forward():
    """No implicit states and no cycles: run_model alone must converge."""
    p = _build(nn=3, rho=RHO_ALT)
    before = p.get_val('phi').copy()
    p.run_model()
    assert_near_equal(p.get_val('phi'), before, 1e-14)



@pytest.mark.parametrize('hinge_input', ['e_over_R', 'omega_n_ratio'])
def test_totals(hinge_input):
    extra = {'omega_n_ratio': 1.0387} if hinge_input == 'omega_n_ratio' else {}
    p = _build(nn=3, hinge_input=hinge_input, rho=RHO_ALT, **extra)

    wrt = ['R', 'c', 'rho', 'a', 'Omega', 'I_b_ref', 'r_over_R']
    wrt.append('omega_n_ratio' if hinge_input == 'omega_n_ratio' else 'e_over_R')

    data = p.check_totals(
        of=['gamma', 'zeta', 'phi', 'b1s_over_a1s', 'dalpha_over_a1s',
            't_63', 'psi_63'],
        wrt=wrt, method='cs', compact_print=True, out_stream=None)

    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-12)
