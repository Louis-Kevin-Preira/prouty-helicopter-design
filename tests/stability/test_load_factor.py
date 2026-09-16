"""Tests -- angle-of-attack stability against load factor (pp. 622-623)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.load_factor_margin_comp import (
    FIGURE_9_18,
    LoadFactorMarginComp,
)
from prouty.stability.plots import plot_load_factor_margin

#: Table 9.16's three columns for dM/dzdot, with the stabilizer counted five
#: times for the 90 sq ft configuration p. 622 discusses.
TABLE_9_16 = {'main': 495.0, 'horiz': -219.0, 'fuse': 374.0}


def run(nn=1, **overrides):
    prob = om.Problem()
    prob.model.add_subsystem('m', LoadFactorMarginComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in overrides.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


def test_figure_9_18_endpoints():
    """The line runs from (1.0, -214) to (1.82, 0)."""
    prob = run()
    assert_near_equal(prob.get_val('dM_dzdot')[0], -214.0, 1e-10)
    assert_near_equal(prob.get_val('load_factor_at_neutral')[0],
                      FIGURE_9_18['neutral'], 2e-3)


def test_the_margin_is_gone_at_1_82_g():
    """p. 622: "the margin would be gone in a 1.82-g turn or pull-up"."""
    assert_near_equal(run().get_val('load_factor_margin')[0], 0.82, 3e-3)
    assert run(load_factor=1.82).get_val('dM_dzdot')[0] > -1.0
    assert run(load_factor=2.0).get_val('dM_dzdot')[0] > 0.0


def test_the_total_at_one_g_agrees_with_table_9_16():
    """495 - 5 x 219 + 374 = -226 against the figure's -214, 5 %."""
    total = (TABLE_9_16['main'] + 5.0 * TABLE_9_16['horiz']
             + TABLE_9_16['fuse'])
    assert_near_equal(total, -226.0, 1e-10)
    assert abs(total / run().get_val('dM_dzdot')[0] - 1.0) < 0.06


def test_the_split_cannot_be_recovered_from_table_9_8():
    """Its three dM/dzdot terms give 176 where the figure needs 261.

    Chapter 9 never prints the split, which is why FIGURE_9_18 holds values
    read back off the line. Recording the gap here so nobody later mistakes
    them for derived quantities.
    """
    hub_spring = 200940.0 * 1.2 * 0.00138
    h_M_arm = -(-6.0) * 7.5
    l_M_arm = (-261.0) * (-0.5)

    assert_near_equal(hub_spring + h_M_arm + l_M_arm, 508.3, 2e-3)
    thrust_dependent = h_M_arm + l_M_arm
    assert_near_equal(thrust_dependent, 175.5, 2e-3)
    assert abs(thrust_dependent / FIGURE_9_18['rotor'] - 1.0) > 0.25


def test_only_the_rotor_part_moves():
    """p. 622: "the airframe ... maintains a constant stabilizing influence"."""
    base = run(load_factor=1.0).get_val('dM_dzdot')[0]
    doubled = run(load_factor=2.0).get_val('dM_dzdot')[0]
    assert_near_equal(doubled - base, FIGURE_9_18['rotor'], 1e-10)

    stiffer = run(dM_dzdot_airframe=FIGURE_9_18['airframe'] - 100.0)
    assert_near_equal(stiffer.get_val('load_factor_at_neutral')[0],
                      575.0 / 261.0, 1e-9)


def test_a_bigger_stabilizer_buys_load_factor():
    """Which is the design use: it moves the crossing to the right."""
    margins = [run(dM_dzdot_airframe=FIGURE_9_18['airframe'] - extra
                   ).get_val('load_factor_margin')[0]
               for extra in (0.0, 100.0, 200.0)]
    assert margins == sorted(margins)


def test_the_limit_load_factor_is_the_thing_to_constrain():
    """Finite everywhere, where the neutral point runs to infinity."""
    prob = run(limit_load_factor=2.0)
    assert_near_equal(prob.get_val('dM_dzdot_at_limit')[0], 47.0, 1e-10)
    assert prob.get_val('dM_dzdot_at_limit')[0] > 0.0      # unstable at 2 g

    enough = run(dM_dzdot_airframe=-600.0, limit_load_factor=2.0)
    assert enough.get_val('dM_dzdot_at_limit')[0] < 0.0


def test_a_rotor_that_never_destabilises():
    """Then stability does not vanish at any load factor, and it says so."""
    with pytest.warns(UserWarning, match='never vanishes'):
        prob = run(dM_dzdot_rotor=-50.0)
    assert_near_equal(prob.get_val('load_factor_at_neutral')[0], 0.0, 1e-13)
    assert prob.get_val('dM_dzdot_at_limit')[0] < 0.0


def test_the_airplane_contrast_of_p622():
    """An aeroplane's angle-of-attack stability is "nearly invariant with wing
    angle of attack" -- a zero rotor contribution, a horizontal line."""
    flat = run(dM_dzdot_rotor=1e-12, load_factor=3.0)
    assert_near_equal(flat.get_val('dM_dzdot')[0],
                      FIGURE_9_18['airframe'], 1e-6)


def test_the_figure_is_drawn(tmp_path):
    path = tmp_path / 'load_factor.png'
    plot_load_factor_margin(FIGURE_9_18['rotor'], FIGURE_9_18['airframe'],
                            path, limit=2.0)
    assert path.exists() and path.stat().st_size > 10000


def test_partials():
    prob = run(nn=3, load_factor=1.4, limit_load_factor=2.5)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-9, rtol=1e-9)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in ('dM_dzdot', 'load_factor_at_neutral', 'load_factor_margin'):
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
