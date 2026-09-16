"""G5/G6 group tests -- Figure 9.14 and the wired control response (pp. 598-613)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_near_equal

from prouty.stability.control_response_hover import heaviside_step_response
from prouty.stability.control_response_hover_group import (
    DEG_PER_INCH,
    ControlResponseHoverGroup,
)
from prouty.stability.mil_response_requirements_comp import (
    FIGURE_9_14,
    figure_9_14_violations,
)


def run(nn=1, **overrides):
    prob = om.Problem()
    prob.model.add_subsystem('g', ControlResponseHoverGroup(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in overrides.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


# ------------------------------------------------------- Figure 9.14, p. 613

def test_the_three_boxes_of_figure_9_14():
    """Pitch, roll and yaw, the last with two roles."""
    assert FIGURE_9_14['longitudinal']['any'] == (5.0, 12.3, 1.0)
    assert FIGURE_9_14['lateral']['any'] == (9.0, 20.5, 0.5)
    assert set(FIGURE_9_14['directional']) == {'utility', 'armed'}
    assert FIGURE_9_14['directional']['armed'][2] < \
        FIGURE_9_14['directional']['utility'][2]


@pytest.mark.parametrize('rate, tau, expected', [
    (8.0, 0.5, ()),                                   # inside the pitch box
    (3.0, 0.5, ('sluggish',)),
    (15.0, 0.5, ('oversensitive',)),
    (8.0, 1.5, ('takes too long to respond',)),
    (15.0, 1.5, ('oversensitive', 'takes too long to respond')),
])
def test_the_pitch_box_boundaries(rate, tau, expected):
    assert figure_9_14_violations(rate, tau, 'longitudinal') == expected


def test_the_armed_helicopter_box_is_the_tighter_one():
    """p. 613: two yaw boxes, 15-24 at .5 s and 30-50 at .24 s."""
    assert figure_9_14_violations(20.0, 0.4, 'directional', 'utility') == ()

    # the same point fails the armed box on both axes at once
    assert figure_9_14_violations(20.0, 0.4, 'directional', 'armed') == \
        ('sluggish', 'takes too long to respond')
    assert figure_9_14_violations(40.0, 0.4, 'directional', 'armed') == \
        ('takes too long to respond',)
    assert figure_9_14_violations(40.0, 0.2, 'directional', 'armed') == ()


def test_the_plane_is_the_one_p612_arrives_at():
    """Time constant is the inverse of damping over inertia."""
    prob = run()
    assert_near_equal(prob.get_val('time_constant')[0],
                      1.0 / prob.get_val('damping_over_inertia')[0], 1e-12)
    assert_near_equal(prob.get_val('time_constant')[0], 1.3957, 1e-3)


def test_p612s_two_asymptotic_limits_of_the_displacement():
    """As damping vanishes the displacement is 2 Theta/t^2; as it grows
    without bound it is Theta (Damping/Inertia). Both recover CP/I.
    """
    power_ratio = run().get_val('control_power_over_inertia')[0]

    slack = run(dM_dq=-1e-2)
    assert_near_equal(2.0 * np.radians(slack.get_val('displacement')[0]),
                      power_ratio, 1e-3)

    stiff = run(dM_dq=-4.0e8)
    assert_near_equal(np.radians(stiff.get_val('displacement')[0])
                      * stiff.get_val('damping_over_inertia')[0],
                      power_ratio, 1e-3)


# -------------------------------------------------- the group, end to end

def test_it_runs_standalone_on_the_book_defaults():
    prob = run()
    assert prob.get_val('denominator_coeffs').shape == (1, 5)


def test_the_denominator_is_the_p598_cubic():
    prob = run()
    denominator = prob.get_val('denominator_coeffs')[0]
    monic = denominator[:4] / denominator[3]
    assert np.allclose(monic[::-1], [1.0, 0.724, 0.0, 0.115], atol=1e-3), monic


def test_the_attitude_gain_p607():
    prob = run()
    denominator = prob.get_val('denominator_coeffs')[0]
    gain = prob.get_val('numerator_coeffs')[0][1] / denominator[3]
    assert_near_equal(gain, -6.78, 3e-3)


def test_the_two_sign_conversions_are_visible_in_the_model():
    """Table 9.18 wants magnitudes; the derivatives are negative."""
    prob = run()
    assert prob.get_val('dM_dq')[0] < 0.0
    assert prob.get_val('dM_dB1')[0] < 0.0
    assert_near_equal(prob.get_val('damping')[0], -prob.get_val('dM_dq')[0],
                      1e-12)
    assert_near_equal(prob.get_val('control_power')[0],
                      -prob.get_val('dM_dB1')[0]
                      * np.deg2rad(DEG_PER_INCH), 1e-10)


def test_the_gearing_puts_it_on_figure_9_13():
    """2.71 deg/in is what gives 0.32 on the pitch abscissa."""
    assert_near_equal(run().get_val('control_power_over_inertia')[0], 0.32,
                      1e-2)


def test_it_passes_table_9_18_and_fails_figure_9_14():
    """Both at once, which is p. 613's point in printing the second figure.

    MIL-H-8501A: 7.32 deg/in against a required 2.65, and 28,659 of damping
    against a required 24,977. Figure 9.14: 25.6 deg/sec/in at a 1.40 second
    time constant, outside the pitch box on both axes.
    """
    prob = run()
    assert prob.get_val('response_margin')[0] > 0.0
    assert prob.get_val('damping_margin')[0] > 0.0
    assert_near_equal(prob.get_val('displacement')[0], 7.32, 5e-3)

    assert_near_equal(prob.get_val('steady_rate')[0], 25.59, 1e-3)
    assert figure_9_14_violations(prob.get_val('steady_rate')[0],
                                  prob.get_val('time_constant')[0],
                                  'longitudinal') == (
        'oversensitive', 'takes too long to respond')


def test_a_lighter_gearing_brings_it_into_the_box():
    """Halving the cyclic per inch halves the steady rate; the time constant
    does not move, because it depends on damping alone."""
    softer = run(deg_per_inch=1.2)          # the promoted input is in degrees
    assert_near_equal(softer.get_val('steady_rate')[0], 25.59 * 1.2 / 2.71,
                      1e-2)
    assert_near_equal(softer.get_val('time_constant')[0], 1.3957, 1e-3)
    assert figure_9_14_violations(softer.get_val('steady_rate')[0],
                                  softer.get_val('time_constant')[0],
                                  'longitudinal') == (
        'takes too long to respond',)


def test_the_time_history_comes_off_the_transfer_function():
    """Figure 9.12, through heaviside_step_response as post-processing."""
    prob = run()
    denominator = prob.get_val('denominator_coeffs')[0]
    attitude = prob.get_val('numerator_coeffs')[0]

    rate = np.zeros_like(attitude)
    rate[2] = attitude[1]                                # q = s Theta
    response = heaviside_step_response(rate / denominator[3],
                                        denominator / denominator[3],
                                        np.array([0.0, 2.0, 8.0]))
    assert abs(response[0]) < 1e-6
    assert_near_equal(response[1], -6.87, 2e-2)
    assert_near_equal(response[2], 8.07, 2e-2)


def test_totals():
    prob = run()
    data = prob.check_totals(
        of=['displacement', 'damping_margin', 'steady_rate'],
        wrt=['dM_dq', 'dM_dB1', 'I_yy'], method='cs', out_stream=None)
    for key, entry in data.items():
        scale = max(1.0, np.max(np.abs(entry['J_fwd'])))
        assert entry['abs error'].forward < 1e-6 * scale, key


@pytest.mark.parametrize('nn', (1, 3))
def test_vectorized(nn):
    prob = run(nn=nn)
    for name in ('displacement', 'steady_rate', 'time_constant'):
        assert prob.get_val(name).shape == (nn,)
