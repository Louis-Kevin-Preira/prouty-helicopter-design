"""Chapter 2, G4 -- ThrustDampingGroup, pp. 101-102."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import (assert_check_partials, assert_check_totals,
                                         assert_near_equal)

from prouty.stability.basic_rotor_derivatives_hover_comp import BasicRotorDerivativesHoverComp
from prouty.vertical import (ClimbCollectiveComp, ClimbPowerGroup, FlowStatesGroup,
                             ThrustDampingComp, ThrustDampingGroup)

# Example helicopter main rotor, Table 9.1 p. 564 and Table 9.2 p. 566
A_SLOPE, SIGMA, V_TIP, RHO, A_B = 6.0, 0.085, 650.0, 0.002378, 240.0
CT_SIGMA_EX = 0.0849


def _run_damping(CT_sigma, V_c=0.0, inflow='exact', a=A_SLOPE, sigma=SIGMA):
    CT_sigma = np.atleast_1d(np.asarray(CT_sigma, dtype=float))
    nn = CT_sigma.size
    p = om.Problem()
    p.model.add_subsystem('d', ThrustDampingComp(num_nodes=nn, inflow=inflow), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('CT_sigma', CT_sigma)
    if inflow == 'exact':
        p.set_val('V_c', np.broadcast_to(V_c, (nn,)).astype(float), units='ft/s')
    p.set_val('a', a)
    p.set_val('sigma', sigma)
    p.set_val('V_tip', V_TIP, units='ft/s')
    p.set_val('rho', RHO)
    p.set_val('A', A_B / sigma, units='ft**2')
    p.run_model()
    return p


def _run_group(theta_T, V_c, inflow):
    theta_T = np.atleast_1d(np.asarray(theta_T, dtype=float))
    nn = theta_T.size
    p = om.Problem()
    p.model.add_subsystem('g4', ThrustDampingGroup(num_nodes=nn, inflow=inflow), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('theta_T', theta_T, units='rad')
    p.set_val('V_c', np.broadcast_to(V_c, (nn,)).astype(float), units='ft/s')
    p.set_val('a', A_SLOPE)
    p.set_val('sigma', SIGMA)
    p.set_val('V_tip', V_TIP, units='ft/s')
    p.set_val('rho', RHO)
    p.set_val('A', A_B / SIGMA, units='ft**2')
    p.run_model()
    return p


# ------------------------------------------------------------------------
# book anchors
# ------------------------------------------------------------------------

@pytest.mark.parametrize('inflow', ['small_climb', 'exact'])
def test_table_9_1_main_rotor(inflow):
    """dCT_sigma/dlambda = .49 for the example main rotor in hover (p. 564)."""
    p = _run_damping(CT_SIGMA_EX, inflow=inflow)
    assert_near_equal(p.get_val('dCT_sigma_dlambda')[0], 0.49, 0.005)


@pytest.mark.parametrize('inflow', ['small_climb', 'exact'])
def test_table_9_2_heave_damping(inflow):
    """dT/dV_c is Z_w in hover: dZ/dzdot = -182 lb/(ft/s) (p. 566)."""
    p = _run_damping(CT_SIGMA_EX, inflow=inflow)
    assert_near_equal(p.get_val('dT_dVc', units='lbf*s/ft')[0], -182.0, 0.005)


@pytest.mark.parametrize('sigma', [0.085, 0.14614])
def test_identical_to_chapter9_in_hover(sigma):
    """G4 at V_c = 0 and Table 9.1 of Chapter 9 are the same equation."""
    CT_sigma = np.array([0.04, 0.07, 0.10, 0.13])
    p9 = om.Problem()
    p9.model.add_subsystem('t91', BasicRotorDerivativesHoverComp(num_nodes=4), promotes=['*'])
    p9.setup()
    p9.set_val('CT_sigma', CT_sigma)
    p9.set_val('sigma', sigma)
    p9.set_val('a', A_SLOPE)
    p9.run_model()
    for inflow in ('small_climb', 'exact'):
        p = _run_damping(CT_sigma, inflow=inflow, sigma=sigma)
        assert_near_equal(p.get_val('dCT_sigma_dlambda'), p9.get_val('dCT_sigma_dlambda'), 1e-12)


# ------------------------------------------------------------------------
# internal consistency
# ------------------------------------------------------------------------

@pytest.mark.parametrize('inflow', ['small_climb', 'exact'])
def test_thrust_equation_satisfied(inflow):
    V_c = np.array([-8.0, 0.0, 10.0, 30.0])
    p = _run_group(np.full(4, 0.10), V_c, inflow)
    x, lam = p.get_val('CT_sigma'), V_c / V_TIP
    root = np.sqrt(SIGMA * x / 2) if inflow == 'small_climb' else np.sqrt(lam ** 2 / 4
                                                                         + SIGMA * x / 2)
    assert_near_equal(x, A_SLOPE / 4 * (0.10 - lam / 2 - root), 1e-12)


@pytest.mark.parametrize('inflow', ['small_climb', 'exact'])
def test_damping_is_the_slope_of_the_thrust(inflow):
    """dCT_sigma_dVc at the thrust point equals the slope of CT_sigma(theta_T, V_c)."""
    V_c, h = np.array([-8.0, 0.0, 12.0, 30.0]), 1e-4
    slope = (_run_group(np.full(4, 0.1), V_c + h, inflow).get_val('CT_sigma')
             - _run_group(np.full(4, 0.1), V_c - h, inflow).get_val('CT_sigma')) / (2 * h)
    assert_near_equal(_run_group(np.full(4, 0.1), V_c, inflow).get_val('dCT_sigma_dVc'),
                      slope, 1e-7)


def test_options_agree_in_hover_and_differ_in_climb():
    V_c = np.array([0.0, 20.0, -8.0])
    small = _run_damping(np.full(3, CT_SIGMA_EX), V_c, 'small_climb').get_val('dCT_sigma_dlambda')
    exact = _run_damping(np.full(3, CT_SIGMA_EX), V_c, 'exact').get_val('dCT_sigma_dlambda')
    assert_near_equal(exact[0], small[0], 1e-14)
    assert np.all(np.abs(exact[1:] / small[1:] - 1.0) > 0.01)


def test_thrust_decreases_with_climb():
    p = _run_group(np.full(3, 0.1), np.array([-5.0, 0.0, 20.0]), 'exact')
    assert np.all(p.get_val('dCT_sigma_dVc') < 0.0) and np.all(np.diff(p.get_val('CT_sigma')) < 0)


def test_promoted_defaults_are_consistent_across_the_chapter():
    p = om.Problem()
    m = p.model
    m.add_subsystem('g0', FlowStatesGroup(num_nodes=2), promotes=['*'])
    m.add_subsystem('g2', ClimbPowerGroup(num_nodes=2, tail_rotor='hover_power', inflow='external'),
                    promotes=['*'])
    m.add_subsystem('g3', ClimbCollectiveComp(num_nodes=2), promotes=['*'])
    m.add_subsystem('g4', ThrustDampingGroup(num_nodes=2), promotes=['*'])
    p.setup()
    p.final_setup()


# ------------------------------------------------------------------------
# derivatives
# ------------------------------------------------------------------------

@pytest.mark.parametrize('inflow', ['small_climb', 'exact'])
def test_partials(inflow):
    p = _run_group(np.array([0.06, 0.10, 0.14]), np.array([-6.0, 0.0, 25.0]), inflow)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-9)


def test_totals():
    p = _run_group(np.array([0.08, 0.12]), np.array([5.0, 20.0]), 'exact')
    data = p.check_totals(of=['CT_sigma', 'dT_dVc'], wrt=['theta_T', 'V_c', 'a', 'sigma', 'V_tip'],
                          method='cs', out_stream=None)
    assert_check_totals(data, atol=1e-8, rtol=1e-8)


# ------------------------------------------------------------------------
# link to Chapter 9: HoverDerivativesGroup(thrust_damping='external')
# ------------------------------------------------------------------------

from prouty.stability import HoverDerivativesGroup  # noqa: E402

ROTORS = {'_M': dict(CT_sigma=0.0849, sigma=0.085), '_T': dict(CT_sigma=0.10, sigma=0.14614)}


def _run_chapter9(thrust_damping, V_c=0.0, inflow='exact'):
    p = om.Problem()
    m = p.model
    if thrust_damping == 'external':
        for suffix in ROTORS:
            m.add_subsystem(f'g4{suffix}', ThrustDampingComp(inflow=inflow),
                            promotes_inputs=[('CT_sigma', f'CT_sigma{suffix}'),
                                             ('sigma', f'sigma{suffix}'), ('a', f'a{suffix}')],
                            promotes_outputs=[('dCT_sigma_dlambda', f'dCT_sigma_dlambda{suffix}')])
            m.set_input_defaults(f'a{suffix}', 6.0, units='1/rad')
            m.set_input_defaults(f'sigma{suffix}', ROTORS[suffix]['sigma'])
    m.add_subsystem('ch9', HoverDerivativesGroup(thrust_damping=thrust_damping), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for suffix, vals in ROTORS.items():
        p.set_val(f'CT_sigma{suffix}', vals['CT_sigma'])
        p.set_val(f'sigma{suffix}', vals['sigma'])
        p.set_val(f'a{suffix}', 6.0)
        if thrust_damping == 'external':
            p.set_val(f'g4{suffix}.V_tip', 650.0)
            if inflow == 'exact':
                p.set_val(f'g4{suffix}.V_c', V_c)
    p.run_model()
    return p


def _table_9_4(p):
    names = [n for n in p.model.ch9.total._var_rel_names['output']]
    return {n: p.get_val(n)[0] for n in names}


def test_chapter9_fed_by_g4_reproduces_table_mode_in_hover():
    table, fed = _table_9_4(_run_chapter9('table')), _table_9_4(_run_chapter9('external'))
    for name, value in table.items():
        assert_near_equal(fed[name], value, 1e-12)


def test_chapter9_heave_damping_follows_the_climb():
    """In a climb only the damping rows move, in the ratio of the exact g."""
    hover, climb = _run_chapter9('external', 0.0), _run_chapter9('external', 20.0)
    g0 = _run_damping(0.0849, 0.0).get_val('dCT_sigma_dlambda')[0]
    g20 = _run_damping(0.0849, 20.0).get_val('dCT_sigma_dlambda')[0]
    assert_near_equal(climb.get_val('dZ_dzdot')[0] / hover.get_val('dZ_dzdot')[0], g20 / g0, 1e-12)
    assert_near_equal(climb.get_val('dZ_dtheta0_M')[0], hover.get_val('dZ_dtheta0_M')[0], 1e-12)


def test_chapter9_external_partials():
    p = _run_chapter9('external', 10.0)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None,
                            includes=['*basic*', '*g4*'])
    assert_check_partials(data, atol=1e-8, rtol=1e-8)
    tot = p.check_totals(of=['dZ_dzdot'], wrt=['CT_sigma_M', 'a_M'], method='cs',
                         out_stream=None)
    assert_check_totals(tot, atol=1e-8, rtol=1e-8)
