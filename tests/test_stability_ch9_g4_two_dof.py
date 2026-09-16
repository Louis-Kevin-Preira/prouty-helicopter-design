"""G4 tests -- the two-degree-of-freedom hover system (pp. 598-602)."""

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import (
    PolyDeterminantComp,
    RouthDiscriminantComp,
    describe_modes,
    routh_tests,
)
from prouty.stability.hover_long_two_dof_matrix_comp import (
    HoverLongTwoDofMatrixComp,
)

from test_stability_ch9_g4_longitudinal import EXAMPLE as THREE_DOF

EXAMPLE = {name: THREE_DOF[name] for name in
           ('dX_dxdot', 'dX_dq', 'dM_dxdot', 'dM_dq', 'G_W', 'I_yy', 'g')}

CUBIC = [1.0, 0.724, 0.0, 0.115]                  # p. 598, descending
ROOTS = (-0.87, 0.075, 0.355)                     # p. 598


def run(nn=1, routh=True, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('matrix', HoverLongTwoDofMatrixComp(num_nodes=nn),
                             promotes=['*'])
    prob.model.add_subsystem(
        'det', PolyDeterminantComp(n=2, degree=2, degree_out=3, num_nodes=nn),
        promotes=['*'])
    if routh:
        prob.model.add_subsystem('rd', RouthDiscriminantComp(degree=3,
                                                             num_nodes=nn),
                                 promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


def test_characteristic_equation_p598():
    """s^3 + .724 s^2 + .115 = 0."""
    got = run().get_val('char_coeffs')[0][::-1]
    assert np.allclose(got, CUBIC, atol=1e-3), got


def test_roots_p598():
    """-.87 and an unstable pair at .075 +/- .355i."""
    modes = describe_modes(run().get_val('char_coeffs')[0])
    real = [m for m in modes if not m.oscillatory]
    pair = [m for m in modes if m.oscillatory]

    assert len(real) == 1 and len(pair) == 2
    assert_near_equal(real[0].root.real, ROOTS[0], 2e-2)
    assert_near_equal(pair[0].root.real, ROOTS[1], 3e-2)
    assert_near_equal(abs(pair[0].root.imag), ROOTS[2], 2e-2)


def test_period_and_time_to_double_p600():
    """P = 17.7 s and t_double = 9.2 s, against the 3-DOF 17.5 and 9.1."""
    pair = [m for m in describe_modes(run().get_val('char_coeffs')[0])
            if m.oscillatory]
    assert_near_equal(pair[0].period, 17.7, 2e-2)
    assert_near_equal(pair[0].time_to_double, 9.2, 3e-2)


def test_only_the_plunge_root_is_lost():
    """p. 600: the 2-DOF oscillation is within 2 % of the 3-DOF one."""
    from test_stability_ch9_g4_longitudinal import run as run_three

    two = [m for m in describe_modes(run().get_val('char_coeffs')[0])
           if m.oscillatory][0]
    three = [m for m in describe_modes(run_three().get_val('char_coeffs')[0])
             if m.oscillatory][0]

    assert abs(two.period / three.period - 1.0) < 0.02
    assert abs(two.root.real / three.root.real - 1.0) < 0.03


def test_the_s_coefficient_cancels_to_rounding():
    """(dX/dxdot)(dM/dq) - (dX/dq)(dM/dxdot) = 0 for a single rotor, p. 597.

    Not enforced. With Table 9.4's two-figure values it lands at -3.4e-5 next
    to an s^2 coefficient of .72.
    """
    coeffs = run().get_val('char_coeffs')[0]
    assert abs(coeffs[1]) < 1e-4
    assert abs(coeffs[1]) < 1e-3 * abs(coeffs[2])


def test_the_top_coefficient_vanishes_identically():
    """Only the I_yy entry is quadratic, so the cubic bound is degree four."""
    prob = om.Problem()
    prob.model.add_subsystem('matrix', HoverLongTwoDofMatrixComp(),
                             promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=2, degree=2,
                                                        normalize=False),
                             promotes=['*'])
    prob.setup()
    for name, value in EXAMPLE.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    full = prob.get_val('char_coeffs')[0]
    assert np.allclose(full[4:], 0.0, atol=1e-9 * np.max(np.abs(full)))


def test_routh_discriminant_p602():
    """R.D.(3) = -(g/I_yy)(dM/dxdot) = -.115, so the hover is unstable."""
    prob = run()
    expected = -EXAMPLE['g'] / EXAMPLE['I_yy'] * EXAMPLE['dM_dxdot']
    assert_near_equal(prob.get_val('routh_discriminant')[0], expected, 2e-3)
    assert prob.get_val('routh_discriminant')[0] < 0.0


def test_table_9_17_verdicts():
    """The six tests of Table 9.17 applied to the example helicopter."""
    prob = run()
    verdicts = routh_tests(prob.get_val('char_coeffs')[0],
                           prob.get_val('routh_discriminant')[0])

    assert verdicts['unstable']                      # test 4
    assert not verdicts['no_unstable_oscillation']   # test 2 fails
    assert not verdicts['neutrally_stable']          # test 3


def test_a_teetering_rotor_reverses_the_verdict():
    """p. 603: with dM/dxdot negative, R.D. turns positive and test 2 passes.

    Prouty's figure 9.10 argument: a teetering rotor with no hub stiffness
    reverses the sign of the speed stability derivative, which removes the
    unstable oscillation but leaves a pure divergence -- the constant term goes
    negative, which is test 6.
    """
    prob = run(dM_dxdot=-143.0)
    coeffs = prob.get_val('char_coeffs')[0]
    verdicts = routh_tests(coeffs, prob.get_val('routh_discriminant')[0])

    assert prob.get_val('routh_discriminant')[0] > 0.0
    assert verdicts['no_unstable_oscillation']       # test 2 now passes
    assert not verdicts['all_coefficients_positive']  # test 6: pure divergence
    assert max(m.root.real for m in describe_modes(coeffs)) > 0.0


def test_speed_stability_sign_drives_everything():
    """Tests 3 and 5 of Table 9.17 both hinge on dM/dxdot alone.

    Zeroing dM/dxdot on its own is not a physical case: Table 9.2 builds
    dM/dxdot and dM/dq from the same dM/da1s and the same -h_M, so a rotor
    with no hub spring sitting on the c.g. zeroes both, and only then does the
    s coefficient stay cancelled. Do that, and R.D. and the constant term go to
    zero together -- neutral stability, no oscillation.
    """
    prob = run(dM_dxdot=0.0, dM_dq=0.0)
    assert_near_equal(prob.get_val('routh_discriminant')[0], 0.0, 1e-12)
    assert_near_equal(prob.get_val('char_coeffs')[0][0], 0.0, 1e-12)
    assert max(m.root.real for m
               in describe_modes(prob.get_val('char_coeffs')[0])) < 1e-12


def test_zeroing_dM_dxdot_alone_breaks_the_cancellation():
    """The reason the test above moves dM/dq too, stated as an assertion.

    With dM/dxdot alone at zero the s coefficient becomes
    (dX/dxdot)(dM/dq)/(m I_yy), which is no longer negligible, and R.D. picks
    it up as BC rather than -AD.
    """
    prob = run(dM_dxdot=0.0)
    coeffs = prob.get_val('char_coeffs')[0]
    assert abs(coeffs[1]) > 1e-3
    assert_near_equal(prob.get_val('routh_discriminant')[0],
                      coeffs[2] * coeffs[1], 1e-10)


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    assert np.allclose(triple.get_val('char_coeffs'),
                       np.tile(single.get_val('char_coeffs'), (3, 1)))
