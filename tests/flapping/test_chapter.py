"""Chapter 7 assembled, p. 455-479."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_near_equal

from prouty.flapping import Chapter7FlappingGroup


# ------------------------------------------------------------------------
# chapter7_flapping_group
# ------------------------------------------------------------------------

R_EX, C_EX, I_B_EX, E_OVER_R_EX, B_EX = 30.0, 2.0, 2870.0, 0.05, 4.0


A_EX, SIGMA, RHO_SL, OMEGA_EX, GW, H_M = 6.0, 0.085, 0.002378, 650.0 / 30.0, \
    20000.0, 7.5


def _build(nn=1, **opts):
    over = {k: opts.pop(k) for k in list(opts) if k in
            ('mu', 'theta_0', 'theta_1', 'alpha_s', 'A_1', 'B_1', 'V', 'n',
             'p', 'q', 'l_M')}

    p = om.Problem()
    p.model.add_subsystem('ch7', Chapter7FlappingGroup(num_nodes=nn, **opts),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    p.set_val('R', R_EX)
    p.set_val('c', C_EX)
    p.set_val('I_b_ref', I_B_EX)
    p.set_val('e_over_R', E_OVER_R_EX)
    if opts.get('include_moments', True):
        p.set_val('b', B_EX)          # only RotorStiffnessComp uses it
        p.set_val('T', np.full(nn, GW))
        p.set_val('h_M', H_M)
    p.set_val('sigma', SIGMA)
    p.set_val('a', np.full(nn, A_EX))
    p.set_val('rho', np.full(nn, RHO_SL))
    p.set_val('Omega', np.full(nn, OMEGA_EX))
    p.set_val('mu', np.full(nn, 0.3))
    p.set_val('theta_0', np.full(nn, np.radians(14.0)))
    p.set_val('theta_1', np.full(nn, np.radians(-10.0)))
    p.set_val('alpha_s', np.full(nn, np.radians(-5.0)))
    p.set_val('A_1', np.full(nn, np.radians(-2.2)))
    p.set_val('B_1', np.full(nn, np.radians(1.5)))
    if opts.get('include_rates', True):
        p.set_val('A_1_level', np.full(nn, np.radians(-2.2)))
        if opts.get('rate_source', 'maneuver') == 'maneuver':
            p.set_val('V', np.full(nn, 115.0), units='kn')
            p.set_val('n', np.full(nn, 1.0))
    for k, v in over.items():
        p.set_val(k, v)

    p.run_model()
    return p


def test_everything_runs_and_the_hover_anchors_survive():
    """The G1 results must be unchanged by the rest of the chapter."""
    p = _build()

    assert_near_equal(p.get_val('omega_n_ratio')[0], 1.04, 2e-3)
    assert_near_equal(p.get_val('zeta')[0], 0.42, 1.2e-2)
    assert_near_equal(p.get_val('phi', units='deg')[0], 84.8, 2e-3)
    assert_near_equal(p.get_val('psi_63', units='deg')[0], 130.0, 6e-3)


def test_lock_number_flows_from_the_hover_group():
    """gamma is built once, in G1, and used by every section downstream."""
    p = _build()
    expected = C_EX * RHO_SL * A_EX * R_EX ** 4 / I_B_EX
    assert_near_equal(p.get_val('gamma')[0], expected, 1e-12)


def test_only_one_blade_inertia_is_built():
    p = _build()
    names = [n for n, _ in p.model.list_outputs(prom_name=True, val=False,
                                                out_stream=None)
             if n.endswith('.I_b')]
    assert len(names) == 1


def test_moments_use_the_total_flapping_when_rates_are_on():
    p = _build(n=1.5)

    assert p.get_val('a_1s_rate')[0] < 0.0
    assert_near_equal(p.get_val('M_M')[0],
                      p.get_val('dMM_da1s')[0] * p.get_val('a_1s_total')[0],
                      1e-12)


def test_totals_equal_steady_flapping_without_rates():
    """The two agree at n = 1, so switching the source changes nothing."""
    p = _build(n=1.0)
    assert_near_equal(p.get_val('a_1s_total')[0], p.get_val('a_1s')[0], 1e-14)
    assert_near_equal(p.get_val('M_M')[0],
                      p.get_val('dMM_da1s')[0] * p.get_val('a_1s')[0], 1e-12)


def test_excluding_rates_falls_back_to_the_steady_flapping():
    with_rates = _build(n=1.0)
    without = _build(include_rates=False)

    for name in ('a_1s', 'M_M', 'dCHsigma_da1s', 'dMCG_da1s'):
        assert_near_equal(without.get_val(name)[0],
                          with_rates.get_val(name)[0], 1e-11)


def test_sections_can_be_switched_off():
    p = _build(include_rates=False, include_moments=False,
               include_h_force=False)

    names = {meta['prom_name'] for _, meta in
             p.model.list_outputs(prom_name=True, val=False, out_stream=None)}

    assert 'a_1s' in names
    assert 'M_M' not in names
    assert 'dCHsigma_da1s' not in names


def test_hingeless_rotor_topology():
    p = _build(hinge_input='omega_n_ratio')
    p.set_val('omega_n_ratio', 1.11)
    p.run_model()

    assert 0.10 < p.get_val('e_over_R')[0] < 0.16
    assert p.get_val('dMM_da1s')[0] > 0.0


def test_closed_form_route_runs_through_the_whole_chapter():
    num = _build()
    clo = _build(method='closed_form')

    assert abs(clo.get_val('a_1s')[0] - num.get_val('a_1s')[0]) > 1e-4
    assert np.isfinite(clo.get_val('dMCG_da1s')[0])


def test_group_is_feed_forward():
    p = _build(nn=2, mu=np.array([0.2, 0.35]),
               theta_0=np.full(2, np.radians(14.0)),
               theta_1=np.full(2, np.radians(-10.0)),
               alpha_s=np.full(2, np.radians(-5.0)),
               A_1=np.full(2, np.radians(-2.2)),
               B_1=np.full(2, np.radians(1.5)),
               V=np.full(2, 115.0), n=np.full(2, 1.5))
    before = p.get_val('dMCG_da1s').copy()
    p.run_model()
    assert_near_equal(p.get_val('dMCG_da1s'), before, 1e-14)


def test_totals_across_the_whole_chapter():
    nn = 2
    p = _build(nn, mu=np.array([0.2, 0.35]),
               theta_0=np.full(nn, np.radians(14.0)),
               theta_1=np.full(nn, np.radians(-10.0)),
               alpha_s=np.full(nn, np.radians(-5.0)),
               A_1=np.full(nn, np.radians(-2.2)),
               B_1=np.full(nn, np.radians(1.5)),
               V=np.full(nn, 115.0), n=np.full(nn, 1.5), l_M=0.4)

    data = p.check_totals(
        of=['zeta', 'phi', 'b1s_over_a1s', 'psi_63', 'a_1s', 'b_1s',
            'a_1s_total', 'b_1s_total', 'delta_A_1', 'delta_B_1',
            'M_M', 'M_CG', 'dCHsigma_da1s', 'dMCG_da1s'],
        wrt=['R', 'c', 'I_b_ref', 'e_over_R', 'b', 'sigma', 'a', 'rho',
             'Omega', 'mu', 'theta_0', 'alpha_s', 'B_1', 'V', 'n', 'T',
             'h_M', 'l_M'],
        method='cs', compact_print=True, out_stream=None)

    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-7, atol=1e-4)



@pytest.mark.parametrize('convention', ['hinge', 'center'])
def test_stiffness_convention_reaches_the_moments(convention):
    p = _build(convention=convention)
    ratio = (_build(convention='hinge').get_val('dMM_da1s')[0]
             / _build(convention='center').get_val('dMM_da1s')[0])
    assert_near_equal(ratio, 1.0 / (1 - E_OVER_R_EX), 1e-13)
    assert p.get_val('dMM_da1s')[0] > 0.0
