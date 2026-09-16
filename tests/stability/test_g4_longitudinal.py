"""G4 tests -- longitudinal equations of motion in hover (pp. 596-598)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import PolyDeterminantComp, describe_modes
from prouty.stability.hover_longitudinal_matrix_comp import (
    ZERO_IN_HOVER,
    HoverLongitudinalMatrixComp,
)

# Table 9.4 totals (pp. 571-573), plus the two inertias the chapter implies:
# G.W. = 20,000 lb makes m = 621 slug, which is the -621 s^2 of Table 9.20,
# and I_yy = 40,000 slug ft^2 is the -40,000 s^2 of the same table.
EXAMPLE = dict(
    dX_dxdot=-5.0, dX_dq=1008.0,
    dZ_dzdot=-182.0,
    dM_dxdot=143.0, dM_dzdot=91.0, dM_dq=-28659.0,
    G_W=20000.0, I_yy=40000.0, g=32.2,
)

# p. 597, example helicopter. Prouty prints two significant figures, so .12 is
# a rounding of the .1151 the derivatives actually give.
QUARTIC = [1.0, 1.02, 0.215, 0.12, 0.034]        # descending
# p. 598.
ROOTS = [-0.89, -0.28, 0.076]                     # real parts, third is a pair


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('matrix', HoverLongitudinalMatrixComp(num_nodes=nn),
                             promotes=['*'])
    prob.model.add_subsystem(
        'det', PolyDeterminantComp(n=3, degree=2, degree_out=4, num_nodes=nn),
        promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


def test_characteristic_equation_p597():
    """s^4 + 1.02 s^3 + .215 s^2 + .12 s + .034 = 0."""
    got = run().get_val('char_coeffs')[0][::-1]
    assert np.allclose(got, QUARTIC, atol=5e-3), got


def test_roots_p598():
    """-.89, -.28 and an unstable pair at .076 +/- .360i."""
    modes = describe_modes(run().get_val('char_coeffs')[0])
    real = sorted(m.root.real for m in modes)

    assert_near_equal(real[0], -0.89, 2e-2)
    assert_near_equal(real[1], -0.28, 5e-2)      # -.293 unrounded
    oscillatory = [m for m in modes if m.oscillatory]
    assert len(oscillatory) == 2
    assert_near_equal(oscillatory[0].root.real, 0.076, 2e-2)
    assert_near_equal(abs(oscillatory[0].root.imag), 0.360, 2e-2)


def test_period_and_time_to_double_p598():
    """P = 17.5 s, t_double = 9.1 s."""
    modes = [m for m in describe_modes(run().get_val('char_coeffs')[0])
             if m.oscillatory]
    assert_near_equal(modes[0].period, 17.5, 2e-2)
    assert_near_equal(modes[0].time_to_double, 9.1, 2e-2)


def test_the_plunge_root_is_the_mass_sign_check():
    """-dZ/dzdot/m = -.293. C9-11: the p. 596 plus sign would give +.293.

    The matrix of p. 597 carries the mass with a minus in the Z row, like the
    X and M rows. The Z-force equation printed on p. 596 has a plus, which
    would turn the plunge convergence p. 598 lists as -.28 into a divergence.
    """
    prob = run()
    plunge = EXAMPLE['dZ_dzdot'] / (EXAMPLE['G_W'] / EXAMPLE['g'])
    assert_near_equal(plunge, -0.293, 1e-2)

    real = sorted(m.root.real for m in describe_modes(prob.get_val('char_coeffs')[0]))
    assert min(abs(r - plunge) for r in real) < 0.02


def test_dM_dzdot_drops_out_of_the_determinant():
    """It is 91 in Table 9.4 but the middle row being [0, ., 0] removes it."""
    base = run().get_val('char_coeffs')
    moved = run(dM_dzdot=1000.0).get_val('char_coeffs')
    assert np.allclose(base, moved, atol=1e-12)


def test_the_two_top_coefficients_vanish_identically():
    """The degree bound says six; only the I_yy entry is quadratic."""
    prob = om.Problem()
    prob.model.add_subsystem('matrix', HoverLongitudinalMatrixComp(),
                             promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=3, degree=2,
                                                        normalize=False),
                             promotes=['*'])
    prob.setup()
    for name, value in EXAMPLE.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    full = prob.get_val('char_coeffs')[0]
    assert np.allclose(full[5:], 0.0, atol=1e-9 * np.max(np.abs(full)))


def test_cross_terms_cancel_as_p597_claims():
    """(dM/dq)(dX/dxdot) - (dM/dxdot)(dX/dq) = 0 for Table 9.2's structure.

    Not enforced anywhere: it falls out of Table 9.2 because both dX rows carry
    dCH/sigma/da1s and both dM rows carry dM/da1s and -h_M. Rebuilt here from
    the Table 9.1 pieces, it closes to a part in 10^3, which is the rounding of
    the printed Table 9.2 values.
    """
    scale, dCH, h_M = 241131.0, 0.0398, 7.5
    dM_da1s, a1s_mu, a1s_q, mu_x = 200940.0, 0.34, -0.105, 1.0 / 650.0

    dX_dxdot = -scale * dCH * a1s_mu * mu_x
    dX_dq = -scale * dCH * a1s_q
    dM_dxdot = dM_da1s * a1s_mu * mu_x - dX_dxdot * h_M
    dM_dq = dM_da1s * a1s_q - dX_dq * h_M

    cross = dM_dq * dX_dxdot - dM_dxdot * dX_dq
    assert abs(cross) < 1e-9 * abs(dM_dq * dX_dxdot)


def test_printed_quartic_s_coefficient_is_only_the_gravity_term():
    """(g/I_yy)(dM/dxdot) = .115, and the printed .12 is that alone."""
    got = run().get_val('char_coeffs')[0][1]
    assert_near_equal(got, EXAMPLE['g'] / EXAMPLE['I_yy'] * EXAMPLE['dM_dxdot'],
                      2e-3)


@pytest.mark.parametrize('name', ZERO_IN_HOVER)
def test_hover_zeros_default_to_zero(name):
    """The five derivatives p. 597 calls zero are inputs, defaulting to zero."""
    prob = om.Problem()
    prob.model.add_subsystem('m', HoverLongitudinalMatrixComp(), promotes=['*'])
    prob.setup()
    prob.final_setup()
    assert_near_equal(prob.get_val(name)[0], 0.0, 1e-13)


def test_a_nonzero_hover_zero_changes_the_answer():
    """They are inputs, not hard-coded, so a hingeless case can use them."""
    base = run().get_val('char_coeffs')
    moved = run(dZ_dq=500.0, dZ_dxdot=3.0).get_val('char_coeffs')
    assert not np.allclose(base, moved, atol=1e-6)


def test_partials():
    prob = run(nn=3, dX_dzdot=0.7, dZ_dxdot=1.3, dZ_dzddot=2.1, dZ_dq=40.0,
               dM_dzddot=9.0)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    assert np.allclose(triple.get_val('char_coeffs'),
                       np.tile(single.get_val('char_coeffs'), (3, 1)))
