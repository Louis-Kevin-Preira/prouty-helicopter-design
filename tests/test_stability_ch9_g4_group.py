"""G4 group tests -- every hover stability result of pp. 596-605 at once."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_near_equal

from prouty.stability import describe_modes, routh_tests
from prouty.stability.hover_derivatives_group import HoverDerivativesGroup
from prouty.stability.hover_stability_group import (
    EXAMPLE_DEFAULTS,
    SCALAR_DEFAULTS,
    HoverStabilityGroup,
)

from test_stability_ch9_g1_group import EXAMPLE as ROTOR_STATE
from test_stability_ch9_g1_group import SCALARS as ROTOR_SCALARS

# Everything pp. 596-605 print for the example helicopter.
BOOK = {
    'char_coeffs_long': [1.0, 1.02, 0.215, 0.12, 0.034],          # p. 597
    'char_coeffs_long_2dof': [1.0, 0.724, 0.0, 0.115],            # p. 598
}
BOOK_SCALARS = {
    'routh_discriminant_long_2dof': -0.115,                       # p. 602
    'period': 15.7,                                               # p. 600
    'root_yaw': -0.38,                                            # p. 605
    'time_to_half_yaw': 1.82,                                     # p. 605
}


def run(nn=1, **overrides):
    prob = om.Problem()
    prob.model.add_subsystem('s', HoverStabilityGroup(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in overrides.items():
        scalar = name in SCALAR_DEFAULTS
        prob.set_val(name, value if scalar else np.full(nn, value))
    prob.run_model()
    return prob


def test_it_runs_standalone_on_the_book_defaults():
    """No inputs set: the group already holds the example helicopter."""
    prob = run()
    for name, expected in BOOK.items():
        got = prob.get_val(name)[0][::-1]
        assert np.allclose(got, expected, atol=5e-3), (name, got)
    for name, expected in BOOK_SCALARS.items():
        assert_near_equal(prob.get_val(name)[0], expected, 5e-3)


def test_lateral_cubic_p604():
    got = run().get_val('char_coeffs_lateral')[0][::-1]
    assert np.allclose(got, [1.0, 5.8544, 0.14609, 0.41860], rtol=1e-4), got


def test_the_quartic_factors_exactly_into_the_cubic_and_the_plunge_root():
    """p. 600 calls the 2-DOF roots "almost the same". They are exactly the same.

    In hover dZ/dxdot and dZ/dq are zero, so the heave equation decouples
    completely and the 3-DOF determinant is the 2-DOF cubic times
    (s - dZ/dzdot/m). Prouty's printed roots differ only because he solved two
    separately rounded polynomials.
    """
    prob = run()
    quartic = prob.get_val('char_coeffs_long')[0]
    cubic = prob.get_val('char_coeffs_long_2dof')[0]
    plunge = EXAMPLE_DEFAULTS['dZ_dzdot'][0] / (EXAMPLE_DEFAULTS['G_W'][0]
                                                / EXAMPLE_DEFAULTS['g'][0])

    assert np.allclose(quartic, np.convolve(cubic, [-plunge, 1.0]), atol=1e-14)
    assert_near_equal(plunge, -0.293, 1e-2)


def test_the_three_longitudinal_analyses_agree_on_the_oscillation():
    """3 DOF and 2 DOF give the same period; Hohenemser drops 13 %."""
    prob = run()
    periods = [
        [m for m in describe_modes(prob.get_val(name)[0]) if m.oscillatory][0].period
        for name in ('char_coeffs_long', 'char_coeffs_long_2dof')
    ] + [prob.get_val('period')[0]]

    assert_near_equal(periods[0], periods[1], 1e-12)
    assert_near_equal(periods[1], 17.7, 2e-2)
    assert_near_equal(periods[2], 15.7, 5e-3)
    assert periods[2] < periods[0]


def test_the_factorisation_needs_the_hover_zeros():
    """Give dZ/dxdot or dZ/dq a value and the heave mode couples back in."""
    prob = run(dZ_dq=500.0)
    quartic = prob.get_val('char_coeffs_long')[0]
    cubic = prob.get_val('char_coeffs_long_2dof')[0]
    plunge = EXAMPLE_DEFAULTS['dZ_dzdot'][0] / (EXAMPLE_DEFAULTS['G_W'][0]
                                                / EXAMPLE_DEFAULTS['g'][0])
    assert not np.allclose(quartic, np.convolve(cubic, [-plunge, 1.0]),
                           atol=1e-6)


def test_the_quartic_also_gets_a_routh_discriminant():
    """R.D.(4) is not in the book but the quartic supports it, and it agrees."""
    prob = run()
    assert prob.get_val('routh_discriminant_long')[0] < 0.0
    assert max(m.root.real for m
               in describe_modes(prob.get_val('char_coeffs_long')[0])) > 0.0


def test_table_9_17_on_the_two_cubics():
    """Longitudinal unstable, lateral marginally stable on the printed values."""
    prob = run()
    longitudinal = routh_tests(prob.get_val('char_coeffs_long_2dof')[0],
                               prob.get_val('routh_discriminant_long_2dof')[0])
    lateral = routh_tests(prob.get_val('char_coeffs_lateral')[0],
                          prob.get_val('routh_discriminant_lateral')[0])

    assert longitudinal['unstable']
    assert not longitudinal['no_unstable_oscillation']
    assert lateral['no_unstable_oscillation']


def test_c9_8_flips_the_lateral_verdict_through_the_group():
    prob = run(dR_dydot=-222.0)
    assert prob.get_val('routh_discriminant_lateral')[0] < 0.0
    pair = [m for m in describe_modes(prob.get_val('char_coeffs_lateral')[0])
            if m.oscillatory][0]
    assert not pair.stable


def test_shared_inputs_reach_every_branch_that_needs_them():
    """dM/dq moves the quartic, the cubic and the Hohenemser period together."""
    base, moved = run(), run(dM_dq=-20000.0)
    for name in ('char_coeffs_long', 'char_coeffs_long_2dof'):
        assert not np.allclose(base.get_val(name), moved.get_val(name))
    assert not np.isclose(base.get_val('period')[0],
                          moved.get_val('period')[0])
    # ... and leaves the lateral and yaw branches alone
    assert np.allclose(base.get_val('char_coeffs_lateral'),
                       moved.get_val('char_coeffs_lateral'))
    assert np.isclose(base.get_val('root_yaw')[0], moved.get_val('root_yaw')[0])


def test_hover_zeros_are_inputs_not_hard_coded():
    """The five derivatives p. 597 calls zero can be given values."""
    base, moved = run(), run(dZ_dq=500.0, dZ_dxdot=3.0)
    assert not np.allclose(base.get_val('char_coeffs_long'),
                           moved.get_val('char_coeffs_long'))


def test_ct_sigma_bar_is_kept_apart_from_ct_sigma():
    """p. 601 uses .086 with the vertical drag penalty; Table 9.1 implies .0849."""
    names = set(EXAMPLE_DEFAULTS) | set(SCALAR_DEFAULTS)
    assert 'CT_sigma_bar_M' in names
    assert 'CT_sigma_M' not in names


# ------------------------------------------------- G1 feeding G4 end to end

def chained():
    """HoverDerivativesGroup -> HoverStabilityGroup, by promotion alone."""
    prob = om.Problem()
    prob.model.add_subsystem('derivatives', HoverDerivativesGroup(),
                             promotes=['*'])
    prob.model.add_subsystem('stability', HoverStabilityGroup(),
                             promotes=['*'])
    prob.model.set_input_defaults('G_W', val=np.full(1, 20000.0), units='lbf')
    prob.setup()
    for name, value in ROTOR_STATE.items():
        prob.set_val(name, value if name in ROTOR_SCALARS else np.full(1, value))
    prob.set_val('I_yy', np.full(1, 40000.0))
    prob.set_val('I_xx', np.full(1, 5000.0))
    prob.set_val('I_zz', np.full(1, 35000.0))
    prob.run_model()
    return prob


def test_the_two_groups_chain_by_promotion():
    """Rotor geometry in one end, hover stability out the other."""
    prob = chained()
    got = prob.get_val('char_coeffs_long')[0][::-1]
    assert np.allclose(got, BOOK['char_coeffs_long'], atol=2e-2), got


def test_chained_yaw_and_period():
    prob = chained()
    assert_near_equal(prob.get_val('root_yaw')[0], -0.38, 3e-2)
    assert_near_equal(prob.get_val('period')[0], 15.7, 5e-2)


def test_chained_flapping_derivatives_come_from_table_9_1():
    """The Hohenemser inputs are promoted out of the Table 9.1 component."""
    prob = chained()
    assert_near_equal(prob.get_val('d_a1s_d_mu_M')[0], 0.34, 2e-2)
    assert_near_equal(prob.get_val('d_a1s_dq_M')[0], -0.105, 2e-2)


def test_chained_lateral_inherits_the_c9_8_value():
    """Fed by Table 9.3's equation rather than its printed value, so -222."""
    prob = chained()
    assert_near_equal(prob.get_val('dR_dydot')[0], -222.0, 5e-2)
    assert prob.get_val('routh_discriminant_lateral')[0] < 0.0


@pytest.mark.parametrize('nn', (1, 3))
def test_vectorized(nn):
    prob = run(nn=nn)
    for name in ('char_coeffs_long', 'char_coeffs_lateral'):
        assert prob.get_val(name).shape[0] == nn
    assert prob.get_val('root_yaw').shape == (nn,)


def test_totals():
    prob = run()
    data = prob.check_totals(
        of=['char_coeffs_long', 'routh_discriminant_lateral', 'root_yaw'],
        wrt=['dM_dq', 'dR_dydot', 'I_zz'], method='cs', out_stream=None)
    for key, entry in data.items():
        scale = max(1.0, np.max(np.abs(entry['J_fwd'])))
        assert entry['abs error'].forward < 1e-6 * scale, key
