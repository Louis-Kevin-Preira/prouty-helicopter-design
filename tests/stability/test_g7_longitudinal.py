"""G7 tests -- longitudinal equations of motion in forward flight (pp. 614-618)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import (
    PolyDeterminantComp,
    RouthDiscriminantComp,
    describe_modes,
)
from prouty.stability.long_matrix_ff_comp import LongMatrixFFComp

# The Table 9.16 totals (pp. 591-595) plus the three quantities Table 9.20
# implies: G.W. = 20,000 lb from its -621 s^2, I_yy = 40,000 from its
# -40,000 s^2, and Theta_bar = -.0165 rad from its 3927 s.
EXAMPLE = dict(
    dX_dxdot=-20.0, dX_dzdot=-8.0, dX_dq=1937.0,
    dZ_dxdot=49.0, dZ_dzdot=-287.0, dZ_dzddot=0.0, dZ_dq=-217.0,
    dM_dxdot=144.0, dM_dzdot=650.0, dM_dzddot=9.0, dM_dq=-43752.0,
    G_W=20000.0, I_yy=40000.0, g=32.2, V=194.1, Theta_bar=-0.016507,
)

# p. 617, descending.
QUARTIC = [1.0, 1.545, -2.618, 0.0228, 0.0949]
ROUTH = -0.32
ROOTS = [-2.564, -0.1782, 0.2106, 0.9867]

# p. 616: the coupled eighth-order system, for the misprint test.
COUPLED = [1.0, 10.02, 28.88, 48.98, 26.28, -137.88, -4.627, 4.315, 0.1675]
COUPLED_ROOTS = [-6.602, -2.907, -0.7822 + 2.4432j, -0.7822 - 2.4432j,
                 -0.1710, -0.0391, 0.1828, 1.085]


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('matrix', LongMatrixFFComp(num_nodes=nn),
                             promotes=['*'])
    prob.model.add_subsystem(
        'det', PolyDeterminantComp(n=3, degree=2, n_zero_roots=2, num_nodes=nn),
        promotes=['*'])
    prob.model.add_subsystem('rd', RouthDiscriminantComp(degree=4, num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


def test_characteristic_equation_p617():
    """s^4 + 1.545 s^3 - 2.618 s^2 + .0228 s + .0949 = 0."""
    got = run().get_val('char_coeffs')[0][::-1]
    assert np.allclose(got, QUARTIC, atol=3e-3), got


def test_routh_discriminant_p617():
    """R.D. = -.32, so the helicopter is longitudinally unstable at 115 knots."""
    assert_near_equal(run().get_val('routh_discriminant')[0], ROUTH, 2e-2)
    assert run().get_val('routh_discriminant')[0] < 0.0


def test_roots_p617():
    """Four real roots: -2.564, -.1782, .2106, .9867."""
    modes = describe_modes(run().get_val('char_coeffs')[0])
    assert not any(m.oscillatory for m in modes)
    real = sorted(m.root.real for m in modes)
    assert np.allclose(real, sorted(ROOTS), atol=2e-2), real


def test_the_divergence_doubles_in_under_a_second():
    """p. 617: the largest positive root governs. .9867 gives 0.70 seconds."""
    fastest = max(describe_modes(run().get_val('char_coeffs')[0]),
                  key=lambda m: m.root.real)
    assert_near_equal(fastest.time_to_double, 0.70, 3e-2)
    assert fastest.time_to_double < 1.0


def test_the_matrix_reproduces_table_9_20():
    """Every entry of the longitudinal block, p. 614."""
    mat = run().get_val('matrix_coeffs')[0]
    expected = {
        (0, 0): (0.0, -20.0, -621.1), (0, 1): (0.0, -8.0, 0.0),
        (0, 2): (-20000.0, 3927.0, 0.0),
        (1, 0): (0.0, 49.0, 0.0), (1, 1): (0.0, -287.0, -621.1),
        (1, 2): (0.0, 120338.0, 0.0),
        (2, 0): (0.0, 144.0, 0.0), (2, 1): (0.0, 650.0, 9.0),
        (2, 2): (0.0, -43752.0, -40000.0),
    }
    for (row, col), values in expected.items():
        for power, value in enumerate(values):
            assert abs(mat[row, col, power] - value) <= max(0.5,
                                                            1e-3 * abs(value)), \
                (row, col, power, mat[row, col, power], value)


def test_the_two_terms_forward_flight_adds():
    """Both come from the trim velocity and neither exists in hover."""
    prob = run()
    mass = EXAMPLE['G_W'] / EXAMPLE['g']
    momentum = mass * EXAMPLE['V']

    # pitch rate rotating the velocity vector: 1990 against 1937 aerodynamic
    kinematic = -momentum * EXAMPLE['Theta_bar']
    assert_near_equal(kinematic, 1990.0, 1e-2)
    assert_near_equal(prob.get_val('matrix_coeffs')[0][0, 2, 1],
                      EXAMPLE['dX_dq'] + kinematic, 1e-10)

    # the centrifugal term swamps the aerodynamics by 500 to 1
    assert_near_equal(momentum, 120555.0, 1e-3)
    assert abs(EXAMPLE['dZ_dq'] / momentum) < 0.003


def test_theta_bar_is_what_table_9_20_implies():
    """-0.95 degrees, slightly nose down, from the 3927 s entry."""
    mass = EXAMPLE['G_W'] / EXAMPLE['g']
    theta = (EXAMPLE['dX_dq'] - 3927.0) / (mass * EXAMPLE['V'])
    assert_near_equal(np.degrees(theta), -0.95, 2e-2)
    assert_near_equal(theta, EXAMPLE['Theta_bar'], 1e-2)


def test_the_determinant_carries_two_zero_roots():
    """Displacement form, so the degree-six determinant has an s^2 factor."""
    prob = om.Problem()
    prob.model.add_subsystem('matrix', LongMatrixFFComp(), promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=3, degree=2,
                                                        normalize=False),
                             promotes=['*'])
    prob.setup()
    for name, value in EXAMPLE.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()

    full = prob.get_val('char_coeffs')[0]
    assert np.allclose(full[:2], 0.0, atol=1e-8 * np.max(np.abs(full)))
    assert abs(full[6]) > 0.0                       # genuinely degree six


def test_the_uncoupled_subset_tracks_the_coupled_system():
    """p. 617 compares the two root sets; they match to about 13 %.

    Uncoupled: -2.564, -.1782, .2106, .9867
    Coupled:   -2.907, -.1710, .1828, 1.085
    """
    modes = sorted(m.root.real
                   for m in describe_modes(run().get_val('char_coeffs')[0]))
    coupled = sorted([-2.907, -0.1710, 0.1828, 1.085])
    for uncoupled, full in zip(modes, coupled):
        assert abs(uncoupled / full - 1.0) < 0.16


def test_the_coupled_equation_has_two_s_squared_terms_printed():
    """C9-18: p. 616 prints `- 4.627 s^2 + 4.315 s^2`, a degree-8 polynomial
    with nine coefficients and two of them on the same power.

    Rebuilding the polynomial from the eight roots printed on the same page
    puts 4.3154 in the `s` position, so the second `s^2` is a misprint.
    """
    rebuilt = np.real(np.poly(COUPLED_ROOTS))
    assert len(COUPLED) == 9
    assert np.allclose(rebuilt, COUPLED, rtol=2e-3, atol=2e-2), rebuilt
    assert_near_equal(rebuilt[7], 4.315, 1e-3)      # the s coefficient


def test_partials():
    prob = run(nn=3, dZ_dzddot=2.0)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    assert np.allclose(triple.get_val('char_coeffs'),
                       np.tile(single.get_val('char_coeffs'), (3, 1)))
