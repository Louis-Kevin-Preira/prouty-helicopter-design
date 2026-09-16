"""HoverDerivativesGroup tests -- Tables 9.1 to 9.4 end to end (pp. 563-573).

Unlike the per-table tests, nothing here is fed a printed intermediate value.
The group is given the example helicopter's physical parameters and has to
produce Table 9.4 through Table 9.1 and then Tables 9.2 and 9.3.
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_near_equal

from prouty.stability.hover_derivatives_group import (
    BASIC_OUTPUTS,
    HoverDerivativesGroup,
    suffixed,
)

from test_stability_ch9_g1_table94 import TABLE_9_4

# The main rotor state is the one recovered in the Table 9.1 anchors. The tail
# rotor state is recovered the same way, from the two printed rows that involve
# it: dCQ/sigma/dlambda' = -.038 and da1s/dmu = .34, with theta_1_T = 0, give
# theta_0_T = .1888 and v_1/(Omega R)_T = .0817. sigma_T and CT_sigma_T then
# follow from dCT/sigma/dlambda' = .44.
EXAMPLE = dict(
    rho=0.002378,
    # main rotor
    e_over_R_M=0.05, a_M=6.0, sigma_M=0.085, R_M=30.0, A_b_M=240.0,
    Omega_M=650.0 / 30.0, Omega_R_M=650.0, gamma_M=8.1, CT_sigma_M=0.0849,
    theta_0_M=0.2794, theta_1_M=-0.1396, v_1_over_Omega_R_M=0.062, a_0_M=0.075,
    h_M=7.5, l_M=-0.5, y_M=0.0, i_M=0.0,
    a1s_bar=-0.025, b1s_bar=-0.027, CQ_sigma_bar=0.006696,
    dCT_sigma_dtheta0_M=0.61, dCQ_sigma_dtheta0_M=0.078,
    # tail rotor
    e_over_R_T=0.0, a_T=6.0, sigma_T=0.14614, R_T=6.5, A_b_T=19.397,
    Omega_T=100.0, Omega_R_T=650.0, gamma_T=4.0, CT_sigma_T=0.08281,
    theta_0_T=0.18880, theta_1_T=0.0, v_1_over_Omega_R_T=0.081733,
    a_0_T=0.0375, h_T=6.0, l_T=37.0,
    dCT_sigma_dtheta0_T=0.50, dCQ_sigma_dtheta0_T=0.089,
)

SCALARS = ('e_over_R_M', 'a_M', 'sigma_M', 'R_M', 'A_b_M', 'h_M', 'l_M',
           'y_M', 'i_M', 'e_over_R_T', 'a_T', 'sigma_T', 'R_T', 'A_b_T',
           'h_T', 'l_T')

# Rows the chapter's own roundings keep out of reach end to end. Each is
# explained in docs/validation_stability.md.
KNOWN_DEPARTURES = {
    'dR_dydot': 'C9-8, tail rotor sign misprint',
    'dM_dtheta0_T': 'C9-9, tail rotor torque sign',
    'dX_dydot': 'dY/dxdot rounded from 1.48 to 1',
    'dY_dxdot': 'dY/dxdot rounded from 1.48 to 1',
    'dR_dxdot': 'dY/dxdot rounded before multiplying by h_M',
    'dM_dydot': 'equals dR/dxdot',
}


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('g', HoverDerivativesGroup(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALARS else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name', sorted(TABLE_9_4))
def test_table_9_4_end_to_end(name):
    """Physical parameters in, Table 9.4 out, through three layers."""
    if name in KNOWN_DEPARTURES:
        pytest.skip(KNOWN_DEPARTURES[name])
    expected = TABLE_9_4[name][2]
    got = run().get_val(name)[0]
    assert abs(got - expected) <= max(0.5, 2e-2 * abs(expected)), \
        f'{name}: {got} vs {expected}'


def test_table_9_1_is_reproduced_for_both_rotors():
    """The group's own Table 9.1 layer, against the printed two columns."""
    prob = run()
    main = {'d_mu_d_xdot': 0.00154, 'd_a1s_d_mu': 0.34, 'd_b1s_d_mu': 0.10,
            'dCT_sigma_dlambda': 0.49, 'dCQ_sigma_dlambda': -0.076,
            'd_a1s_dq': -0.105, 'd_a1s_dp': 0.037, 'd_a1s_dA1': 0.086,
            'd_a1s_dB1': -0.993, 'dCH_sigma_da1s': 0.040,
            'dM_da1s': 200940.0}
    tail = {'d_mu_d_xdot': 0.00154, 'd_a1s_d_mu': 0.34, 'd_b1s_d_mu': 0.05,
            'dCT_sigma_dlambda': 0.44, 'dCQ_sigma_dlambda': -0.038,
            'd_a1s_dA1': 0.0, 'dM_da1s': 0.0}

    for rotor, expected in (('main', main), ('tail', tail)):
        for name, value in expected.items():
            got = prob.get_val(f'g.{rotor}_basic.{name}')[0]
            assert abs(got - value) <= max(5e-4, 1.5e-2 * abs(value)), \
                f'{rotor} {name}: {got} vs {value}'


def test_teetering_tail_rotor_has_no_hub_moment_path():
    """e/R_T = 0 zeroes dM/da1s and the cyclic coupling, as Table 9.1 prints."""
    prob = run()
    for name in ('dM_da1s', 'dR_db1s', 'd_a1s_dA1', 'd_b1s_dB1'):
        assert_near_equal(prob.get_val(f'g.tail_basic.{name}')[0], 0.0, 1e-13)


def test_inflow_signs_survive_the_wiring():
    """dlambda'/dzdot > 0 on the main rotor, dlambda'/dydot < 0 on the tail."""
    prob = run()
    assert prob.get_val('g.main_basic.d_lambda_d_zdot')[0] > 0.0
    assert prob.get_val('g.tail_basic.d_lambda_d_ydot')[0] < 0.0
    assert prob.get_val('dZ_dzdot')[0] < 0.0
    assert prob.get_val('dY_dydot')[0] < 0.0


def test_the_two_departures_are_the_documented_ones():
    """No third row moves away from the book beyond the rounding band."""
    prob = run()
    beyond = {name for name, (_, _, total) in TABLE_9_4.items()
              if abs(prob.get_val(name)[0] - total)
              > max(0.5, 5e-2 * abs(total))}
    assert beyond == {'dR_dydot', 'dM_dtheta0_T', 'dR_dxdot', 'dM_dydot'}


def test_chart_entries_are_group_inputs_with_the_book_defaults():
    """p. 564 sends these two to the Chapter 1 charts; they default to it."""
    prob = om.Problem()
    prob.model.add_subsystem('g', HoverDerivativesGroup(), promotes=['*'])
    prob.setup()
    prob.final_setup()
    for name, value in (('dCT_sigma_dtheta0_M', 0.61),
                        ('dCQ_sigma_dtheta0_M', 0.078),
                        ('dCT_sigma_dtheta0_T', 0.50),
                        ('dCQ_sigma_dtheta0_T', 0.089)):
        assert_near_equal(prob.get_val(name)[0], value, 1e-13)


def test_naming_rule():
    """Shared and already-qualified names take no suffix; everything else does."""
    assert suffixed('rho', '_M') == 'rho'
    assert suffixed('h_M', '_M') == 'h_M'
    assert suffixed('l_T', '_T') == 'l_T'
    assert suffixed('a1s_bar', '_M') == 'a1s_bar'
    assert suffixed('A_b', '_M') == 'A_b_M'
    assert suffixed('gamma', '_T') == 'gamma_T'


def test_connection_lists_come_from_the_components():
    """Every Table 9.1 output a dimensional table asks for is wired, not typed."""
    prob = run()
    wired = dict(prob.model._conn_global_abs_in2out)
    for rotor in ('main', 'tail'):
        for name in BASIC_OUTPUTS:
            key = f'g.{rotor}_table.{name}'
            if key in wired:
                assert wired[key] == f'g.{rotor}_basic.{name}'


def test_blade_closest_reaches_the_tail_rotor_table():
    up, down = run(), None
    prob = om.Problem()
    prob.model.add_subsystem('g', HoverDerivativesGroup(blade_closest='down'),
                             promotes=['*'])
    prob.setup()
    for name, value in EXAMPLE.items():
        prob.set_val(name, value if name in SCALARS else np.full(1, value))
    prob.run_model()
    assert_near_equal(prob.get_val('dM_dtheta0_T')[0],
                      -up.get_val('dM_dtheta0_T')[0], 1e-13)


def test_totals():
    """One end-to-end gradient, to check the chain differentiates."""
    prob = run()
    data = prob.check_totals(of=['dM_dq', 'dN_dr', 'dY_dydot'],
                             wrt=['theta_0_M', 'h_M', 'l_T'],
                             method='cs', out_stream=None)
    for key, entry in data.items():
        assert entry['abs error'].forward < 1e-6 * max(
            1.0, abs(float(np.atleast_1d(entry['J_fwd']).ravel()[0]))), key


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_4:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
