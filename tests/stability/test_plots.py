"""Tests -- the two presentation figures (pp. 609, 619).

The drawing itself gets a smoke test: a file is written and it is not blank.
What is tested properly is the grid machinery underneath, because that is
where a wrong figure would come from.
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_near_equal

from prouty.stability import PolyDeterminantComp, describe_modes
from prouty.stability.control_response_hover import heaviside_step_response
from prouty.stability.long_matrix_ff_comp import LongMatrixFFComp
from prouty.stability.long_stability_map_comp import classify
from prouty.stability.mil_response_requirements_comp import FIGURE_9_14
from prouty.stability.plots import (
    LATERAL_REGIONS,
    REGIONS,
    classify_grid,
    plot_lateral_stability_map,
    plot_longitudinal_stability_map,
    plot_response_box,
    plot_step_response,
    roots_on_grid,
)
from prouty.stability.stability_map import affine_coefficients

from prouty.stability.lateral_matrix_ff_comp import LateralMatrixFFComp
from test_g5_control_response import rate_polynomials
from test_g7_longitudinal import EXAMPLE as LONGITUDINAL
from test_g8_lateral import EXAMPLE as LATERAL

#: The five stabilizer areas of Figure 9.15, as (dM/dzdot, dM/dxdot, label).
AREAS = ((650.0, 144.0, '18'), (431.0, 186.0, '36'), (212.0, 228.0, '54'),
         (-7.0, 270.0, '72'), (-226.0, 312.0, '90'))


def evaluate(first, second):
    """Characteristic coefficients, descending, for a pair of derivatives."""
    prob = om.Problem()
    prob.model.add_subsystem('m', LongMatrixFFComp(), promotes=['*'])
    prob.model.add_subsystem('d', PolyDeterminantComp(n=3, degree=2,
                                                      n_zero_roots=2),
                             promotes=['*'])
    prob.setup()
    values = dict(LONGITUDINAL, dM_dzdot=first, dM_dxdot=second)
    for name, value in values.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    return prob.get_val('char_coeffs')[0][::-1]


# ------------------------------------------------------ the grid machinery

def test_batched_roots_match_numpy():
    rng = np.random.default_rng(0)
    coefficients = np.concatenate(
        [np.ones((4, 5, 1)), rng.normal(size=(4, 5, 4))], axis=-1)
    batched = roots_on_grid(coefficients)
    for i in range(4):
        for j in range(5):
            assert np.allclose(np.sort_complex(batched[i, j]),
                               np.sort_complex(np.roots(coefficients[i, j])))


def test_the_grid_classifier_agrees_with_classify():
    """Point for point, against the function the map components already use."""
    for first, second, _ in AREAS:
        descending = evaluate(first, second)
        grid = classify_grid(descending[None, :])[0]
        expected = {'stable': 0, 'unstable oscillation': 1,
                    'unstable divergence': 2}[classify(descending[::-1])]
        assert grid == expected, (first, second, REGIONS[grid])


def test_the_whole_plane_costs_four_model_evaluations():
    """The affine expansion fixes it; the grid is arithmetic after that."""
    calls = []

    def counted(first, second):
        calls.append((first, second))
        return evaluate(first, second)

    constant, d_first, d_second = affine_coefficients(counted)
    assert len(calls) == 4

    # and the expansion reproduces the model anywhere
    for first, second, _ in AREAS:
        predicted = constant + first * d_first + second * d_second
        assert np.allclose(evaluate(first, second), predicted, atol=1e-10)


def test_the_five_areas_land_in_the_regions_p619_describes():
    """p. 619: doubling moves it from pure divergences to unstable
    oscillations, and further growth stabilises it."""
    codes = [classify_grid(evaluate(first, second)[None, :])[0]
             for first, second, _ in AREAS]
    assert codes[0] == 2                      # 18 sq ft, pure divergence
    assert codes[1] == 1                      # 36 sq ft, unstable oscillation
    assert codes[-1] == 0                     # 90 sq ft, stable
    assert codes == sorted(codes, reverse=True)


def test_the_region_boundary_is_where_the_roots_turn_real():
    """Walking the 18 to 36 segment crosses from divergence to oscillation."""
    first = np.linspace(650.0, 431.0, 40)
    second = np.linspace(144.0, 186.0, 40)
    codes = [classify_grid(evaluate(a, b)[None, :])[0]
             for a, b in zip(first, second)]
    assert codes[0] == 2 and codes[-1] == 1
    assert len(set(codes)) == 2


# -------------------------------------------------------------- the figures

def test_the_map_is_drawn(tmp_path):
    path = tmp_path / 'longitudinal_map.png'
    plot_longitudinal_stability_map(evaluate, path, points=AREAS,
                                    resolution=60,
                                    title='Example helicopter at 115 knots')
    assert path.exists() and path.stat().st_size > 10000


def test_the_map_accepts_explicit_ranges(tmp_path):
    path = tmp_path / 'ranged.png'
    plot_longitudinal_stability_map(evaluate, path, points=AREAS[:1],
                                    first_range=(-400.0, 800.0),
                                    second_range=(0.0, 400.0),
                                    resolution=40)
    assert path.exists()


def test_the_response_is_drawn(tmp_path):
    numerator, denominator = rate_polynomials()
    times = np.linspace(0.0, 10.0, 300)
    response = heaviside_step_response(numerator, denominator, times)

    path = tmp_path / 'response.png'
    plot_step_response(times, response, path, single_dof=(-9.44, 1.396))
    assert path.exists() and path.stat().st_size > 10000


def test_how_long_the_two_response_curves_actually_agree():
    """p. 608 says "essentially identical during the first quarter cycle".
    Measured, that is generous.

        t = 1.0 s   -4.788 against -4.828   0.04
        t = 2.0 s   -6.843 against -7.187   0.34
        t = 3.0 s   -7.088 against -8.339   1.25
        t = 4.4 s   -5.046 against -9.036   3.99

    The oscillation's period is 17.7 seconds, so a true quarter cycle is
    4.4 seconds, where the two differ by **44 %** of the smaller. They are
    within 5 % through two seconds and part company steadily after. The plot
    carries both curves so a reader can see where the translation takes over,
    which is the useful part of Figure 9.12.
    """
    numerator, denominator = rate_polynomials()
    times = np.linspace(0.0, 10.0, 2001)
    free = heaviside_step_response(numerator, denominator, times)
    trunnions = -9.4411 * (1.0 - np.exp(-times / 1.3957))
    gap = np.abs(free - trunnions)

    assert gap[times <= 2.0].max() < 0.05 * 7.0
    assert_near_equal(times[np.argmax(gap > 1.0)], 2.79, 2e-2)

    quarter_cycle = np.argmin(np.abs(times - 4.4))
    assert gap[quarter_cycle] / abs(free[quarter_cycle]) > 0.4


def test_the_response_shows_the_reversal():
    """The trace turns round inside the first quarter cycle, which is the
    thing a plot says and four roots do not."""
    numerator, denominator = rate_polynomials()
    times = np.linspace(0.0, 10.0, 401)
    response = heaviside_step_response(numerator, denominator, times)

    trough = times[np.argmin(response)]
    assert 2.0 < trough < 5.0
    assert response[-1] > 0.0 > response.min()


def evaluate_lateral(first, second):
    """The lateral plane of Figure 9.23."""
    prob = om.Problem()
    prob.model.add_subsystem('m', LateralMatrixFFComp(), promotes=['*'])
    prob.model.add_subsystem('d', PolyDeterminantComp(n=3, degree=2,
                                                      n_zero_roots=2),
                             promotes=['*'])
    prob.setup()
    values = dict(LATERAL, dR_dydot=first, dN_dydot=second)
    for name, value in values.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    return prob.get_val('char_coeffs')[0][::-1]


def test_the_lateral_regions_are_named_after_the_modes():
    """Figure 9.23 calls the same three codes spiral dive and Dutch roll."""
    assert LATERAL_REGIONS[0] == REGIONS[0] == 'stable'
    assert LATERAL_REGIONS[1] == 'unstable Dutch roll'
    assert LATERAL_REGIONS[2] == 'spiral dive'


def test_the_lateral_map_puts_the_aircraft_in_the_stable_region():
    assert classify_grid(evaluate_lateral(-382.0, 1207.0)[None, :])[0] == 0


def test_the_lateral_boundaries_are_crossed_at_opposite_ends():
    """Losing directional stability gives the Dutch roll; gaining too much
    gives the spiral dive. Both from the same plane, as Figure 9.23 shows."""
    assert classify_grid(evaluate_lateral(-382.0, 50.0)[None, :])[0] == 1
    assert classify_grid(evaluate_lateral(-382.0, 3500.0)[None, :])[0] == 2


def test_the_lateral_map_is_drawn(tmp_path):
    path = tmp_path / 'lateral_map.png'
    plot_lateral_stability_map(evaluate_lateral, path,
                               points=((-382.0, 1207.0, 'example'),),
                               first_range=(-500.0, 0.0),
                               second_range=(-200.0, 2000.0), resolution=60)
    assert path.exists() and path.stat().st_size > 10000


# ------------------------------------------------------- Figure 9.14's box

def test_the_box_is_drawn_and_reports_the_verdict(tmp_path):
    path = tmp_path / 'box.png'
    plot_response_box(25.59, 1.3957, path, label='example')
    assert path.exists() and path.stat().st_size > 10000


def test_the_box_handles_the_two_yaw_roles(tmp_path):
    """Both are drawn so the tighter one is visible."""
    assert len(FIGURE_9_14['directional']) == 2
    for role in FIGURE_9_14['directional']:
        path = tmp_path / f'yaw_{role}.png'
        plot_response_box(20.0, 0.4, path, axis='directional', role=role)
        assert path.exists()


def test_the_box_needs_no_grid():
    """Its boundaries are the box; there is nothing to classify."""
    import inspect

    import prouty.stability.plots as plots

    source = inspect.getsource(plots.plot_response_box)
    assert 'meshgrid' not in source and 'classify_grid' not in source


def test_nothing_here_is_a_component():
    """The figures are post-processing; the model never draws."""
    import prouty.stability.plots as plots

    assert not any(isinstance(getattr(plots, name, None), type)
                   and issubclass(getattr(plots, name), om.ExplicitComponent)
                   for name in dir(plots))
