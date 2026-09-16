"""G5 tests -- control response in hover (pp. 605-609)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import HoverLongTwoDofMatrixComp, TransferFunctionGroup
from prouty.stability.control_response_hover import (
    HoverControlColumnComp,
    heaviside_step_response,
)

from test_stability_ch9_g4_two_dof import EXAMPLE as MATRIX

# Recovered from Table 9.1 and Table 9.2 in hover:
#   dX/dB1 = -rho A_b (Omega R)^2 (dCH/sigma/da1s)(da1s/dB1)
#   dM/dB1 = (dM/da1s)(da1s/dB1) - (dX/dB1) h_M
# with da1s/dB1 = -1/(1 + kappa^2) = -.99138.
CONTROL = dict(dX_dB1=9516.0, dM_dB1=-270578.0)

#: p. 607: the pitch rate numerator coefficient, (1/I_yy)(dM/dB1).
NUMERATOR_GAIN = -6.78

#: p. 608, with the angles in degrees.
CLOSED_FORM = (5.78, -0.874, -6.85, 0.075, 20.34, 57.54)


def transfer_function(nn=1, **overrides):
    """Theta(s)/B1(s) for the two-degree-of-freedom hover system."""
    values = dict(MATRIX, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('matrix', HoverLongTwoDofMatrixComp(num_nodes=nn),
                             promotes=['*'])
    prob.model.add_subsystem('column', HoverControlColumnComp(num_nodes=nn),
                             promotes=['*'])
    prob.model.add_subsystem(
        'tf', TransferFunctionGroup(n=2, degree=2, column=1, num_nodes=nn),
        promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, np.full(nn, value))
    for name, value in CONTROL.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


def closed_form(times):
    a, s1, b, s2, omega, phase = CLOSED_FORM
    return a * np.exp(s1 * times) + b * np.exp(s2 * times) * np.sin(
        np.deg2rad(omega * times + phase))


def test_the_control_column_is_minus_the_right_hand_side():
    """p. 606: the equations carry -(dX/dB1)B1 and -(dM/dB1)B1."""
    column = transfer_function().get_val('control_coeffs')[0]
    assert_near_equal(column[0, 0], -CONTROL['dX_dB1'], 1e-12)
    assert_near_equal(column[1, 0], -CONTROL['dM_dB1'], 1e-12)
    assert np.allclose(column[:, 1:], 0.0)


def test_the_denominator_is_the_p598_cubic():
    """The transfer function's denominator is the characteristic equation."""
    denominator = transfer_function().get_val('denominator_coeffs')[0]
    monic = denominator[:4] / denominator[3]
    assert np.allclose(monic[::-1], [1.0, 0.724, 0.0, 0.115], atol=1e-3), monic


def test_the_numerator_collapses_to_one_term():
    """p. 606: "some terms in the numerator cancel themselves out".

    (dM/dB1)(dX/dxdot) = (dX/dB1)(dM/dxdot) identically, from Table 9.2's
    shared factors, leaving (dM/dB1) m s. On the rounded Table 9.4 values the
    residual constant term is 5e-5 of the surviving one.
    """
    numerator = transfer_function().get_val('numerator_coeffs')[0]
    assert abs(numerator[0]) < 1e-4 * abs(numerator[1])

    residual = (-CONTROL['dM_dB1'] * MATRIX['dX_dxdot']
                + CONTROL['dX_dB1'] * MATRIX['dM_dxdot'])
    assert_near_equal(numerator[0], residual, 1e-9)


def test_the_attitude_gain_is_dM_dB1_over_I_yy():
    """Theta/B1 = (1/I_yy)(dM/dB1) s / cubic, so the ratio is -6.78."""
    prob = transfer_function()
    numerator = prob.get_val('numerator_coeffs')[0]
    denominator = prob.get_val('denominator_coeffs')[0]

    gain = numerator[1] / denominator[3]
    assert_near_equal(gain, CONTROL['dM_dB1'] / MATRIX['I_yy'], 1e-3)
    assert_near_equal(gain, NUMERATOR_GAIN, 3e-3)


def test_pitch_rate_costs_one_more_s():
    """q = s Theta, so the numerator gains a power."""
    prob = transfer_function()
    attitude = prob.get_val('numerator_coeffs')[0]
    rate = np.concatenate([[0.0], attitude])[:len(attitude)]

    assert abs(rate[0]) < 1e-12
    assert_near_equal(rate[2], attitude[1], 1e-12)


def rate_polynomials():
    """N and D of q(s)/B1(s), ascending, for the example helicopter."""
    prob = transfer_function()
    attitude = prob.get_val('numerator_coeffs')[0]
    denominator = prob.get_val('denominator_coeffs')[0]
    numerator = np.zeros_like(attitude)
    numerator[2] = attitude[1]                      # multiply by s
    return numerator / denominator[3], denominator / denominator[3]


@pytest.mark.parametrize('time', [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0])
def test_heaviside_reproduces_the_p608_closed_form(time):
    """q/B1 = 5.78 e^-.874t - 6.85 e^.075t sin(20.34t + 57.54), p. 608.

    Within 1 %, which is the gap between the book's rounded gain of -6.78 and
    the -6.764 the recovered derivatives give.
    """
    numerator, denominator = rate_polynomials()
    got = heaviside_step_response(numerator, denominator, np.array([time]))[0]
    expected = closed_form(np.array([time]))[0]
    assert abs(got - expected) <= max(0.03, 0.01 * abs(expected)), (time, got)


def test_the_response_starts_from_zero():
    """N(0) = 0, so there is no steady term and the step starts at rest."""
    numerator, denominator = rate_polynomials()
    assert_near_equal(numerator[0], 0.0, 1e-9)
    assert abs(heaviside_step_response(numerator, denominator,
                                       np.array([0.0]))[0]) < 1e-6


def test_the_response_turns_round_within_the_first_period():
    """Figure 9.12: the rate builds to about -7 deg/sec then reverses.

    The unstable oscillation of p. 598 has a 17.7 second period, so the
    reversal near 5 seconds is the first quarter cycle.
    """
    numerator, denominator = rate_polynomials()
    times = np.linspace(0.0, 10.0, 201)
    response = heaviside_step_response(numerator, denominator, times)

    trough = times[np.argmin(response)]
    assert 2.0 < trough < 5.0
    assert -7.5 < response.min() < -6.0
    assert response[-1] > response.min()


def test_heaviside_matches_a_numerical_inverse_transform():
    """A second check on the expansion, independent of the book's algebra."""
    from scipy import signal

    numerator, denominator = rate_polynomials()
    system = signal.lti(np.trim_zeros(numerator[::-1], 'f'),
                        denominator[::-1])
    times, direct = signal.step(system, T=np.linspace(0.0, 6.0, 61))

    expansion = heaviside_step_response(numerator, denominator, times)
    assert np.allclose(expansion, direct, atol=1e-6)


def test_partials():
    prob = transfer_function(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = transfer_function(), transfer_function(nn=3)
    for name in ('numerator_coeffs', 'denominator_coeffs'):
        assert np.allclose(triple.get_val(name),
                           np.tile(single.get_val(name), (3, 1)))
