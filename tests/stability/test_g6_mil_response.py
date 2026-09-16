"""G6 tests -- Table 9.18, MIL-H-8501A response requirements (pp. 610-612)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.mil_response_requirements_comp import (
    FIGURE_9_13_CONTROL_POWER,
    PREFERENCE_ONLY,
    TABLE_9_18,
    MilResponseRequirementsComp,
)

# Table 9.4 hover damping and the three inertias Table 9.20 implies.
AXES = {
    'longitudinal': dict(inertia=40000.0, damping=28659.0),
    'lateral': dict(inertia=5000.0, damping=29127.0),
    'directional': dict(inertia=13326.0 / 0.38074, damping=13326.0),
}
G_W = 20000.0

#: Cube root of G.W. + 1,000 for the example helicopter.
CUBE_ROOT = 21000.0 ** (1.0 / 3.0)


def run(axis='longitudinal', instrument=True, nn=1, **overrides):
    values = dict(AXES[axis], G_W=G_W,
                  control_power=FIGURE_9_13_CONTROL_POWER[axis])
    values.update(overrides)
    prob = om.Problem()
    prob.model.add_subsystem(
        'mil', MilResponseRequirementsComp(axis=axis, instrument=instrument,
                                           num_nodes=nn), promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


def test_the_cube_root_of_gross_weight_plus_a_thousand():
    """27.59 for 20,000 lb; the radical in Table 9.18 carries an index of 3."""
    assert_near_equal(CUBE_ROOT, 27.589, 1e-4)


@pytest.mark.parametrize('axis, instrument, expected', [
    ('longitudinal', False, 45.0), ('longitudinal', True, 73.0),
    ('lateral', False, 27.0), ('lateral', True, 32.0),
    ('directional', False, 110.0), ('directional', True, 110.0),
])
def test_the_minimum_response_column(axis, instrument, expected):
    """Every cell of Table 9.18's two response columns."""
    got = run(axis, instrument).get_val('response_required')[0]
    assert_near_equal(got, expected / CUBE_ROOT, 1e-10)


@pytest.mark.parametrize('axis, instrument, factor', [
    ('longitudinal', False, 8.0), ('longitudinal', True, 15.0),
    ('lateral', False, 18.0), ('lateral', True, 25.0),
    ('directional', False, 27.0), ('directional', True, 27.0),
])
def test_the_damping_column(axis, instrument, factor):
    """Every cell of Table 9.18's two damping columns, k times inertia^0.7."""
    got = run(axis, instrument).get_val('damping_required')[0]
    assert_near_equal(got, factor * AXES[axis]['inertia'] ** 0.7, 1e-10)


def test_the_three_specified_times():
    """One second on pitch and yaw, half a second on roll."""
    assert TABLE_9_18['longitudinal']['time'] == 1.0
    assert TABLE_9_18['lateral']['time'] == 0.5
    assert TABLE_9_18['directional']['time'] == 1.0


def test_only_the_lateral_axis_has_a_maximum_rate():
    """20 deg/sec; the other two carry a dash."""
    assert TABLE_9_18['lateral']['max_rate'] == 20.0
    assert TABLE_9_18['longitudinal']['max_rate'] is None
    assert TABLE_9_18['directional']['max_rate'] is None


def test_the_footnote_is_carried():
    """Visual directional damping is "not a requirement, only a preference"."""
    assert PREFERENCE_ONLY == (('directional', 'visual'),)

    prob = om.Problem()
    prob.model.add_subsystem('mil', MilResponseRequirementsComp(
        axis='directional', instrument=False), promotes=['*'])
    prob.setup()
    assert prob.model.mil.is_preference_only

    prob = om.Problem()
    prob.model.add_subsystem('mil', MilResponseRequirementsComp(
        axis='directional', instrument=True), promotes=['*'])
    prob.setup()
    assert not prob.model.mil.is_preference_only


def test_the_single_degree_of_freedom_solution_of_p611():
    """57.3 (CP/I)/(D/I) [t + (1/(D/I))(exp(-(D/I)t) - 1)] deg per inch."""
    prob = run()
    ratio = AXES['longitudinal']['damping'] / AXES['longitudinal']['inertia']
    power_ratio = (FIGURE_9_13_CONTROL_POWER['longitudinal']
                   / AXES['longitudinal']['inertia'])

    printed = np.degrees(1.0) * (power_ratio / ratio) * (
        1.0 + (np.exp(-ratio) - 1.0) / ratio)
    assert_near_equal(prob.get_val('displacement')[0], printed, 1e-10)
    assert_near_equal(printed, 7.32, 5e-3)


def test_the_solution_integrates_the_printed_equation():
    """I theta_ddot - (dM/dq) theta_dot = (dM/dB1) B1, checked numerically."""
    from scipy.integrate import solve_ivp

    inertia = AXES['longitudinal']['inertia']
    damping = AXES['longitudinal']['damping']
    power = FIGURE_9_13_CONTROL_POWER['longitudinal']

    def rates(_, state):
        return [state[1], (power - damping * state[1]) / inertia]

    solution = solve_ivp(rates, (0.0, 1.0), [0.0, 0.0], rtol=1e-10,
                         atol=1e-12)
    assert_near_equal(np.degrees(solution.y[0, -1]),
                      run().get_val('displacement')[0], 1e-6)


def test_the_steady_rate_is_control_power_over_damping():
    """p. 608's asymptote, -(dM/dB1)/(dM/dq) deg/sec per inch."""
    prob = run()
    assert_near_equal(prob.get_val('steady_rate')[0],
                      np.degrees(FIGURE_9_13_CONTROL_POWER['longitudinal']
                                 / AXES['longitudinal']['damping']), 1e-10)


def test_the_figure_9_13_coordinates():
    """Both axes of the figure are outputs, so a point can be plotted."""
    prob = run()
    assert_near_equal(prob.get_val('damping_over_inertia')[0], 0.7165, 1e-3)
    assert_near_equal(prob.get_val('control_power_over_inertia')[0], 0.32,
                      1e-3)


def test_pitch_and_roll_clear_the_instrument_damping_requirement():
    """But not by the same margin: pitch by 15 %, roll by three to one.

    28,659 against a required 24,977 in pitch, and 29,127 against 9,707 in
    roll. The pitch axis is the tight one, and doubling the stabilizer would
    be the way to widen it.
    """
    pitch = run('longitudinal', instrument=True)
    assert pitch.get_val('damping_margin')[0] > 0.0
    assert_near_equal(pitch.get_val('damping_required')[0], 24977.0, 1e-3)
    assert 1.1 < AXES['longitudinal']['damping'] / pitch.get_val(
        'damping_required')[0] < 1.2

    roll = run('lateral', instrument=True)
    assert roll.get_val('damping_margin')[0] > 0.0
    assert_near_equal(roll.get_val('damping_required')[0], 9707.0, 1e-3)
    assert AXES['lateral']['damping'] > 2.9 * roll.get_val(
        'damping_required')[0]


def test_yaw_fails_the_damping_requirement_by_three():
    """13,326 against a required 40,948 ft lb/rad/sec.

    p. 612 states the example helicopter "would satisfy the instrument flight
    requirements", which is hard to reconcile with the yaw axis on these
    numbers. The hover yaw damping is the tail rotor's -17,797 working against
    a main rotor term of the opposite sign (entry C9-15), and even correcting
    that sign leaves it short.
    """
    prob = run('directional', instrument=True)
    assert_near_equal(prob.get_val('damping_required')[0], 40948.0, 5e-3)
    assert prob.get_val('damping_margin')[0] < 0.0
    assert AXES['directional']['damping'] < 0.4 * prob.get_val(
        'damping_required')[0]

    corrected = 4471.0 + 17797.0          # C9-15: both terms damping
    assert corrected < prob.get_val('damping_required')[0]


def test_every_axis_clears_the_response_requirement():
    """The displacements are several times the Table 9.18 minima."""
    for axis in TABLE_9_18:
        prob = run(axis, instrument=True)
        assert prob.get_val('response_margin')[0] > 0.0


def test_more_damping_cuts_the_response():
    """The two requirements pull against each other, which is the point of
    plotting them on the same axes."""
    base = run().get_val('displacement')[0]
    stiffer = run(damping=2.0 * AXES['longitudinal']['damping'])
    assert stiffer.get_val('displacement')[0] < base
    assert stiffer.get_val('damping_margin')[0] > run().get_val(
        'damping_margin')[0]


@pytest.mark.parametrize('axis', sorted(TABLE_9_18))
def test_partials(axis):
    prob = run(axis, nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in ('response_required', 'damping_required', 'displacement',
                 'response_margin', 'damping_margin'):
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
