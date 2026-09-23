"""G8 tests -- lateral-directional equations in forward flight (pp. 614, 628)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import (
    PolyDeterminantComp,
    RouthDiscriminantComp,
    describe_modes,
)
from prouty.stability.lateral_matrix_ff_comp import LateralMatrixFFComp

# Table 9.16 totals (pp. 591-595) plus the inertias Table 9.20 implies:
# I_xx = 5,000 from its -5000 s^2 and I_zz = 35,000 from its -35,000 s^2.
# Theta_bar is the same -.0165 rad the longitudinal matrix needs.
EXAMPLE = dict(
    dY_dydot=-107.0, dY_dp=-1974.0, dY_dr=1397.0,
    dR_dydot=-382.0, dR_dp=-33738.0, dR_dr=6912.0,
    dN_dydot=1207.0, dN_dp=6909.0, dN_dr=-53913.0,
    G_W=20000.0, I_xx=5000.0, I_zz=35000.0, g=32.2, V=194.1,
    Theta_bar=-0.016507,
)

# p. 628, descending.
QUARTIC = [1.0, 8.460, 17.68, 45.54, 2.2548]
ROOTS_REAL = [-6.842, -0.05058]
DUTCH_ROLL = (-0.7841, 2.4317)
COUPLED = {'roll': -6.602, 'dutch': (-0.7822, 2.4432), 'spiral': -0.03910}


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('matrix', LateralMatrixFFComp(num_nodes=nn),
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


def test_characteristic_equation_p628():
    """s^4 + 8.460 s^3 + 17.68 s^2 + 45.54 s + 2.2548 = 0."""
    got = run().get_val('char_coeffs')[0][::-1]
    assert np.allclose(got, QUARTIC, rtol=5e-3), got


def test_the_lateral_directional_motion_is_stable():
    """All four roots in the left half plane, where the longitudinal diverges."""
    modes = describe_modes(run().get_val('char_coeffs')[0])
    assert all(m.root.real < 0.0 for m in modes)
    assert run().get_val('routh_discriminant')[0] > 0.0


def test_the_three_modes_p628():
    """Roll convergence, Dutch roll and spiral."""
    modes = describe_modes(run().get_val('char_coeffs')[0])
    real = sorted(m.root.real for m in modes if not m.oscillatory)
    pair = [m for m in modes if m.oscillatory]

    assert_near_equal(real[0], ROOTS_REAL[0], 1e-2)        # roll, -6.842
    assert_near_equal(real[1], ROOTS_REAL[1], 3e-2)        # spiral, -.05058
    assert_near_equal(pair[0].root.real, DUTCH_ROLL[0], 1e-2)
    assert_near_equal(abs(pair[0].root.imag), DUTCH_ROLL[1], 1e-2)


def test_the_dutch_roll_period_is_two_and_a_half_seconds():
    """p. 628: "a period of 2.6 seconds and is well damped"."""
    pair = [m for m in describe_modes(run().get_val('char_coeffs')[0])
            if m.oscillatory][0]
    assert_near_equal(pair.period, 2.58, 2e-2)
    assert pair.stable and pair.time_to_half < 1.0


def test_the_spiral_halves_in_about_fourteen_seconds():
    """p. 632-633: "a time to half in amplitude of about 14 seconds"."""
    spiral = min((m for m in describe_modes(run().get_val('char_coeffs')[0])
                  if not m.oscillatory), key=lambda m: abs(m.root.real))
    assert_near_equal(spiral.time_to_half, 13.7, 5e-2)


def test_the_matrix_reproduces_table_9_20():
    """Every entry of the lateral-directional block, p. 614."""
    mat = run().get_val('matrix_coeffs')[0]
    expected = {
        (0, 0): (0.0, -107.0, -621.1), (0, 1): (20000.0, -3964.0, 0.0),
        (0, 2): (0.0, -119158.0, 0.0),
        (1, 0): (0.0, -382.0, 0.0), (1, 1): (0.0, -33738.0, -5000.0),
        (1, 2): (0.0, 6912.0, 0.0),
        (2, 0): (0.0, 1207.0, 0.0), (2, 1): (0.0, 6909.0, 0.0),
        (2, 2): (0.0, -53913.0, -35000.0),
    }
    for (row, col), values in expected.items():
        for power, value in enumerate(values):
            assert abs(mat[row, col, power] - value) <= max(0.5,
                                                            1e-3 * abs(value)), \
                (row, col, power, mat[row, col, power], value)


def test_the_gravity_term_is_positive():
    """+G.W. here against -G.W. in the X row, as Table 9.20 prints."""
    assert_near_equal(run().get_val('matrix_coeffs')[0][0, 1, 0],
                      EXAMPLE['G_W'], 1e-12)


def test_the_kinematic_term_mirrors_the_longitudinal_one():
    """(dY/dp + m V Theta) carries the same 1,990 with the opposite sign."""
    mass = EXAMPLE['G_W'] / EXAMPLE['g']
    kinematic = mass * EXAMPLE['V'] * EXAMPLE['Theta_bar']
    assert_near_equal(kinematic, -1990.0, 1e-2)
    assert_near_equal(run().get_val('matrix_coeffs')[0][0, 1, 1],
                      EXAMPLE['dY_dp'] + kinematic, 1e-10)
    assert_near_equal(EXAMPLE['dY_dp'] + kinematic, -3964.0, 1e-3)


def test_the_centrifugal_term_swamps_the_aerodynamics():
    """(dY/dr - m V): -120,555 against +1,397."""
    mass = EXAMPLE['G_W'] / EXAMPLE['g']
    assert abs(EXAMPLE['dY_dr'] / (mass * EXAMPLE['V'])) < 0.02


def test_there_is_no_product_of_inertia():
    """Table 9.19 carries no I_xz; roll and yaw couple aerodynamically."""
    names = set(EXAMPLE)
    assert not any('xz' in name for name in names)
    assert abs(EXAMPLE['dR_dr'] / EXAMPLE['dN_dp'] - 1.0) < 1e-3


def test_the_uncoupled_subset_tracks_the_coupled_system():
    """p. 628 compares the two root sets."""
    modes = describe_modes(run().get_val('char_coeffs')[0])
    real = sorted(m.root.real for m in modes if not m.oscillatory)
    pair = [m for m in modes if m.oscillatory][0]

    assert abs(real[0] / COUPLED['roll'] - 1.0) < 0.05
    assert abs(pair.root.real / COUPLED['dutch'][0] - 1.0) < 0.01
    assert abs(abs(pair.root.imag) / COUPLED['dutch'][1] - 1.0) < 0.01
    assert abs(real[1] / COUPLED['spiral'] - 1.0) < 0.30      # the loose one


def test_the_determinant_carries_two_zero_roots():
    prob = om.Problem()
    prob.model.add_subsystem('matrix', LateralMatrixFFComp(), promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=3, degree=2,
                                                        normalize=False),
                             promotes=['*'])
    prob.setup()
    for name, value in EXAMPLE.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()

    full = prob.get_val('char_coeffs')[0]
    assert np.allclose(full[:2], 0.0, atol=1e-8 * np.max(np.abs(full)))
    assert abs(full[6]) > 0.0


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    assert np.allclose(triple.get_val('char_coeffs'),
                       np.tile(single.get_val('char_coeffs'), (3, 1)))
