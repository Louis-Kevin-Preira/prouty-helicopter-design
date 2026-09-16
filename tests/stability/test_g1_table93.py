"""Table 9.3 tests -- tail rotor derivatives near hover (pp. 569-570).

Values cross-checked against Table 9.4, pp. 571-573.
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.tail_rotor_derivatives_hover_comp import (
    SCALAR_INPUTS,
    TailRotorDerivativesHoverComp,
)

# Tail rotor column of Table 9.1 (pp. 564-565) plus the geometry Table 9.3
# implies. rho A_b (Omega R)^2 = 19,492 comes from dY/dtheta0 = 9,746 and
# R = 6.5 ft from dM/dtheta0 = 11,276.
INPUTS = dict(
    rho=0.002378, A_b=19.397, Omega_R=650.0, R=6.5, h_T=6.0, l_T=37.0,
    d_lambda_d_ydot=-1.0 / 650.0,
    dCT_sigma_dlambda=0.44, dCT_sigma_dtheta0=0.50,
    dCQ_sigma_dlambda=-0.038, dCQ_sigma_dtheta0=0.089,
)

# Table 9.3, example helicopter column, pp. 569-570.
TABLE_9_3 = {
    'dY_dydot': -13.0, 'dY_dp': -78.0, 'dY_dr': 481.0, 'dY_dtheta0': 9746.0,
    'dR_dydot': 78.0, 'dR_dp': -468.0, 'dR_dr': 2886.0, 'dR_dtheta0': 58476.0,
    'dM_dydot': -7.0, 'dM_dr': 274.0, 'dM_dtheta0': 11276.0,
    'dN_dydot': 481.0, 'dN_dp': 2886.0, 'dN_dr': -17797.0,
    'dN_dtheta0': -360602.0,
}

# The two rows the book prints against its own equations. See C9-8 and C9-9.
CONTRADICTED_BY_THE_BOOK = {'dR_dydot': -79.13, 'dM_dtheta0': -11276.0}


def run(nn=1, blade_closest='up', **overrides):
    values = dict(INPUTS, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem(
        'tr', TailRotorDerivativesHoverComp(num_nodes=nn,
                                            blade_closest=blade_closest),
        promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS
                     else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_3.items())
def test_table_9_3(name, expected):
    """Every printed value, apart from the two the book contradicts itself on.

    Rows downstream of dY/dydot inherit Prouty rounding it from -13.19 to -13
    before multiplying by an arm, hence the 2 % tolerance.
    """
    if name in CONTRADICTED_BY_THE_BOOK:
        pytest.skip('covered by test_rows_the_book_contradicts')
    got = run().get_val(name)[0]
    assert abs(got - expected) <= max(0.5, 2e-2 * abs(expected)), \
        f'{name}: {got} vs {expected}'


@pytest.mark.parametrize('name, expected', CONTRADICTED_BY_THE_BOOK.items())
def test_rows_the_book_contradicts(name, expected):
    assert_near_equal(run().get_val(name)[0], expected, 1e-2)


def test_roll_from_sideslip_follows_its_own_printed_equation():
    """C9-8: dR/dydot = (dY/dydot) h_T, so it is negative, not the printed +78.

    The three sibling rows built the same way -- dR/dp, dR/dr, dR/dtheta0 --
    are all exact against the book, which is what makes the value rather than
    the equation the odd one out.
    """
    prob = run()
    assert_near_equal(prob.get_val('dR_dydot')[0],
                      prob.get_val('dY_dydot')[0] * INPUTS['h_T'], 1e-13)
    assert prob.get_val('dR_dydot')[0] < 0.0

    for name, expected in (('dR_dp', -468.0), ('dR_dr', 2886.0),
                           ('dR_dtheta0', 58476.0)):
        assert abs(prob.get_val(name)[0] - expected) <= 2e-2 * abs(expected)


def test_the_two_torque_rows_cannot_have_opposite_signs():
    """C9-9: dM/dydot and dM/dtheta0 both come from tail rotor torque."""
    for blade_closest in ('up', 'down'):
        prob = run(blade_closest=blade_closest)
        assert (np.sign(prob.get_val('dM_dydot')[0])
                == np.sign(prob.get_val('dM_dtheta0')[0]))


def test_blade_closest_flips_only_the_torque_rows():
    """s reaches dM/dydot, dM/dr and dM/dtheta0, and nothing else."""
    up, down = run(), run(blade_closest='down')
    torque_rows = ('dM_dydot', 'dM_dr', 'dM_dtheta0')

    for name in TABLE_9_3:
        got_up, got_down = up.get_val(name)[0], down.get_val(name)[0]
        if name in torque_rows:
            assert_near_equal(got_down, -got_up, 1e-13)
        else:
            assert_near_equal(got_down, got_up, 1e-13)


def test_default_reproduces_the_printed_torque_rows():
    """'up' is the default because it matches dM/dydot = -7 and dM/dr = 274."""
    prob = run()
    assert prob.get_val('dM_dydot')[0] < 0.0
    assert prob.get_val('dM_dr')[0] > 0.0


def test_arms_are_recoverable_three_ways_each():
    """h_T = 6 from the R rows, l_T = 37 from the N rows."""
    prob = run()
    get = lambda name: prob.get_val(name)[0]

    for moment, force in (('dR_dp', 'dY_dp'), ('dR_dr', 'dY_dr'),
                          ('dR_dtheta0', 'dY_dtheta0')):
        assert_near_equal(get(moment) / get(force), INPUTS['h_T'], 1e-13)

    for moment, force in (('dN_dydot', 'dY_dydot'), ('dN_dp', 'dY_dp'),
                          ('dN_dr', 'dY_dr'), ('dN_dtheta0', 'dY_dtheta0')):
        assert_near_equal(get(moment) / get(force), -INPUTS['l_T'], 1e-13)


def test_rows_that_coincide_in_the_book():
    """dY/dr = dN/dydot = 481 and dR/dr = dN/dp = 2,886."""
    prob = run()
    assert_near_equal(prob.get_val('dY_dr')[0], prob.get_val('dN_dydot')[0],
                      1e-13)
    assert_near_equal(prob.get_val('dR_dr')[0], prob.get_val('dN_dp')[0],
                      1e-13)


def test_sideslip_and_yaw_damping_signs():
    """dY/dydot < 0 and dN/dr < 0: the tail rotor is what makes hover stable."""
    prob = run()
    assert prob.get_val('dY_dydot')[0] < 0.0
    assert prob.get_val('dN_dr')[0] < 0.0


def test_yaw_damping_beats_the_main_rotor_governed_engine_term():
    """Table 9.4, p. 573: -17,797 against +4,471 leaves -13,326."""
    total = run().get_val('dN_dr')[0] + 4471.0
    assert abs(total - (-13326.0)) <= 2e-2 * 13326.0


@pytest.mark.parametrize('blade_closest', ('up', 'down'))
def test_partials(blade_closest):
    prob = run(nn=3, blade_closest=blade_closest)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_3:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-13)
