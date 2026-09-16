"""G7 tests -- phugoid and short-period approximations (pp. 623-626)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import (
    HohenemserPeriodComp,
    PolyDeterminantComp,
    describe_modes,
)
from prouty.stability.long_matrix_ff_comp import LongMatrixFFComp
from prouty.stability.long_mode_approximations import (
    PhugoidMatrixComp,
    ShortPeriodMatrixComp,
)

from test_stability_ch9_g7_longitudinal import EXAMPLE as FULL

#: p. 624 works the three approximations on the 54 sq ft stabilizer, which is
#: the Table 9.16 horizontal stabilizer column counted three times.
HS = dict(dM_dxdot=42.0, dM_dq=-7161.0, dM_dzdot=-219.0, dM_dzddot=9.0,
          dZ_dxdot=1.0, dZ_dzdot=-7.0, dZ_dq=-217.0, dX_dzdot=-1.0)


def tripled():
    """The example helicopter with 54 sq ft of horizontal stabilizer."""
    values = dict(FULL)
    for name, contribution in HS.items():
        values[name] = FULL[name] + 2.0 * contribution
    return values


# p. 624, calculated natural frequency and period.
P624 = {'full 3 dof': (0.365, 17.2), 'full 2 dof': (0.342, 18.4),
        'approximate 2 dof': (0.356, 17.6)}


def run(comp, n_zero_roots, values, nn=1):
    prob = om.Problem()
    prob.model.add_subsystem('matrix', comp(num_nodes=nn), promotes=['*'])
    prob.model.add_subsystem(
        'det', PolyDeterminantComp(n=2, degree=2, num_nodes=nn,
                                   n_zero_roots=n_zero_roots),
        promotes=['*'])
    prob.setup(force_alloc_complex=True)
    inputs = {name for name, _ in
              prob.model.list_inputs(out_stream=None, prom_name=False,
                                     val=False)}
    for name, value in values.items():
        short = name.split('.')[-1]
        if any(entry.endswith(short) for entry in inputs):
            prob.set_val(short, np.full(nn, value))
    prob.run_model()
    return prob


def phugoid(values=None, nn=1):
    return run(PhugoidMatrixComp, 1, values or FULL, nn)


def short_period(values=None, nn=1):
    return run(ShortPeriodMatrixComp, 2, values or FULL, nn)


def full_quartic(values=None):
    values = values or FULL
    prob = om.Problem()
    prob.model.add_subsystem('matrix', LongMatrixFFComp(), promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=3, degree=2,
                                                        n_zero_roots=2),
                             promotes=['*'])
    prob.setup()
    for name, value in values.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    return prob


def damped_frequency(prob):
    """p. 624's "calculated natural frequency" is 2 pi / P, the imaginary part.

    Not the modulus of the root. The distinction is invisible on the phugoid
    reduction, whose real part is .035, and unmissable on the full quartic,
    whose real part is .226 -- there the modulus is .431 against a printed
    .365.
    """
    modes = [m for m in describe_modes(prob.get_val('char_coeffs')[0])
             if m.oscillatory]
    return abs(modes[0].root.imag) if modes else None


# ------------------------------------------------------------- p. 624 table

def test_approximate_two_dof_is_the_hover_formula():
    """p. 624: "the same equation as that derived for hover". .356 rad/sec."""
    values = tripled()
    prob = om.Problem()
    prob.model.add_subsystem('h', HohenemserPeriodComp(), promotes=['*'])
    prob.setup()
    prob.set_val('dM_dxdot', np.full(1, values['dM_dxdot']))
    prob.set_val('dM_dq', np.full(1, values['dM_dq']))
    prob.run_model()

    omega = np.sqrt(prob.get_val('omega_N_squared')[0])
    assert_near_equal(omega, P624['approximate 2 dof'][0], 5e-3)
    assert_near_equal(prob.get_val('period')[0], P624['approximate 2 dof'][1],
                      5e-3)


def test_full_two_dof_phugoid_p624():
    """omega = .342 rad/sec, P = 18.4 s for the 54 sq ft stabilizer."""
    prob = phugoid(tripled())
    modes = [m for m in describe_modes(prob.get_val('char_coeffs')[0])
             if m.oscillatory]
    assert len(modes) == 2
    assert_near_equal(damped_frequency(prob), P624['full 2 dof'][0], 5e-3)
    assert_near_equal(modes[0].period, P624['full 2 dof'][1], 5e-3)


def test_full_three_dof_p624():
    """omega = .365 rad/sec, P = 17.2 s from the complete quartic."""
    prob = full_quartic(tripled())
    modes = [m for m in describe_modes(prob.get_val('char_coeffs')[0])
             if m.oscillatory]
    assert len(modes) == 2
    assert_near_equal(damped_frequency(prob), P624['full 3 dof'][0], 5e-3)
    assert_near_equal(modes[0].period, P624['full 3 dof'][1], 5e-3)

    # the modulus would be .431: p. 624 means the damped frequency
    assert abs(abs(modes[0].root) / P624['full 3 dof'][0] - 1.0) > 0.15


def test_the_three_approximations_bracket_each_other():
    """18.4, 17.6 and 17.2 seconds: under 7 % apart, as p. 624 claims."""
    periods = [P624[key][1] for key in
               ('full 3 dof', 'approximate 2 dof', 'full 2 dof')]
    assert periods == sorted(periods)
    assert max(periods) / min(periods) - 1.0 < 0.07


# --------------------------------------------------------------- C9-19 sign

def test_the_phugoid_cubic_sign_is_minus():
    """C9-19: p. 623 prints a plus where the determinant gives a minus.

    The book's own p. 624 table settles it. With the minus the 54 sq ft
    phugoid comes out at 18.4 seconds, which is what p. 624 prints; with the
    printed plus it is 18.0.
    """
    values = tripled()
    mass = values['G_W'] / values['g']
    kinematic = (values['dX_dq']
                 - mass * values['V'] * values['Theta_bar'])
    I_yy = values['I_yy']

    def period(sign):
        coeffs = np.array([
            I_yy,
            -(I_yy / mass * values['dX_dxdot'] + values['dM_dq']),
            (values['dX_dxdot'] * values['dM_dq']
             + sign * kinematic * values['dM_dxdot']) / mass,
            values['g'] * values['dM_dxdot']]) / I_yy
        roots = [r for r in np.roots(coeffs) if abs(r.imag) > 1e-9]
        return 2.0 * np.pi / abs(roots[0].imag)

    assert_near_equal(period(-1.0), 18.4, 5e-3)
    assert abs(period(1.0) - 18.4) > 0.3

    # and the component, built from the matrix, gives the minus
    modes = [m for m in describe_modes(phugoid(values).get_val('char_coeffs')[0])
             if m.oscillatory]
    assert_near_equal(modes[0].period, period(-1.0), 1e-6)


# -------------------------------------------------------- short period, 625

def test_short_period_tracks_the_full_system():
    """p. 625: "very similar to the corresponding roots" of the 3-DOF system."""
    approx = sorted(m.root.real for m
                    in describe_modes(short_period().get_val('char_coeffs')[0]))
    full = sorted(m.root.real for m
                  in describe_modes(full_quartic().get_val('char_coeffs')[0]))

    assert_near_equal(approx[0], -2.549, 1e-2)
    assert_near_equal(approx[1], 1.037, 1e-2)
    assert abs(approx[0] / full[0] - 1.0) < 0.01        # against -2.564
    assert abs(approx[1] / full[3] - 1.0) < 0.06        # against .9867


def test_the_short_period_quadratic_is_printed_correctly():
    """Unlike the phugoid cubic, p. 625's equation matches the determinant."""
    values = FULL
    mass = values['G_W'] / values['g']
    I_yy = values['I_yy']
    denominator = I_yy * (mass - values['dZ_dzddot'])
    lift = values['dZ_dq'] + mass * values['V']

    damping = (-I_yy * values['dZ_dzdot']
               + values['dM_dq'] * (values['dZ_dzddot'] - mass)
               - lift * values['dM_dzddot']) / denominator
    stiffness = (values['dZ_dzdot'] * values['dM_dq']
                 - lift * values['dM_dzdot']) / denominator

    got = short_period().get_val('char_coeffs')[0]
    assert_near_equal(got[1], damping, 1e-10)
    assert_near_equal(got[0], stiffness, 1e-10)
    assert_near_equal(damping, 1.5123, 1e-3)
    assert_near_equal(stiffness, -2.6427, 1e-3)


def test_the_phugoid_reduction_is_the_worse_one():
    """p. 624: it "sacrificed reasonableness for the phugoid damping".

    On the 18 sq ft aircraft the full system has four real roots and the
    phugoid reduction returns a complex pair, which is a change of character
    and not just of damping.
    """
    assert not any(m.oscillatory for m
                   in describe_modes(full_quartic().get_val('char_coeffs')[0]))
    assert any(m.oscillatory for m
               in describe_modes(phugoid().get_val('char_coeffs')[0]))
    assert any(not m.oscillatory for m
               in describe_modes(short_period().get_val('char_coeffs')[0]))


def test_each_reduction_strips_the_right_number_of_zero_roots():
    """Phugoid keeps one rigid-body root, short period two."""
    for comp, n_zero, degree in ((PhugoidMatrixComp, 1, 3),
                                 (ShortPeriodMatrixComp, 2, 2)):
        prob = om.Problem()
        prob.model.add_subsystem('m', comp(), promotes=['*'])
        prob.model.add_subsystem('det', PolyDeterminantComp(n=2, degree=2,
                                                            normalize=False),
                                 promotes=['*'])
        prob.setup()
        for name, value in FULL.items():
            names = {n for n, _ in prob.model.list_inputs(out_stream=None,
                                                          val=False)}
            if any(entry.endswith(name) for entry in names):
                prob.set_val(name, np.full(1, value))
        prob.run_model()
        full = prob.get_val('char_coeffs')[0]
        assert np.allclose(full[:n_zero], 0.0,
                           atol=1e-8 * np.max(np.abs(full))), comp.__name__
        assert abs(full[n_zero]) > 0.0
        assert len(full) - n_zero == degree + 1


@pytest.mark.parametrize('comp, n_zero', [(PhugoidMatrixComp, 1),
                                          (ShortPeriodMatrixComp, 2)])
def test_partials(comp, n_zero):
    prob = run(comp, n_zero, dict(FULL, dZ_dzddot=2.0), nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    for builder in (phugoid, short_period):
        single, triple = builder(), builder(nn=3)
        assert np.allclose(triple.get_val('char_coeffs'),
                           np.tile(single.get_val('char_coeffs'), (3, 1)))
