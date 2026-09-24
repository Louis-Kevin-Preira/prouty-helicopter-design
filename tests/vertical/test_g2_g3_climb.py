"""Chapter 2, G2 -- ClimbPowerGroup, pp. 97-101, and G3 -- ClimbCollectiveComp, p. 101."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import (assert_check_partials, assert_check_totals,
                                         assert_near_equal)

from prouty.vertical import ClimbCollectiveComp, ClimbPowerGroup, FlowStatesGroup

RHO_SL = 0.002377

# Figure 2.3 p. 99, AH-1G: parameters and test conditions printed on the figure
AH1G = dict(GW=7600.0, rho=0.91 * RHO_SL, R=22.0, tr_R=4.67, l_T=27.5, Dv_GW=0.025,
            dAz_CD=5.0, V_tip=746.0, P_MR=870.0)
# "Calculated" line, digitized (ft/min, hp)
FIG_2_3 = np.array([[200, 23.9], [400, 51.4], [600, 81.7], [800, 113.2], [1000, 145.9],
                    [1200, 181.5], [1400, 219.4]])

# Figure 2.4 p. 100, example helicopter (Appendix A, l_T from p. 309), "full equation"
EXAMPLE = dict(GW=20000.0, rho=RHO_SL, R=30.0, tr_R=6.5, l_T=36.8, Dv_GW=0.04,
               dAz_CD=0.0, V_tip=650.0, P_MR=1600.0)
FIG_2_4 = np.array([[251, 100], [502, 200], [934, 400], [1498, 700], [2000, 1000],
                    [2606, 1400], [3136, 1800], [3505, 2100], [3854, 2400]])


def _run_climb(rc_fpm, case, T=None, tail_rotor='hover_power'):
    rc = np.atleast_1d(np.asarray(rc_fpm, dtype=float))
    nn = rc.size
    c = dict(case)
    p = om.Problem()
    p.model.add_subsystem('g2', ClimbPowerGroup(num_nodes=nn, tail_rotor=tail_rotor),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('GW', np.full(nn, c['GW']), units='lbf')
    p.set_val('T', np.full(nn, c['GW'] if T is None else T), units='lbf')
    p.set_val('V_c', rc, units='ft/min')
    p.set_val('rho', c['rho'], units='slug/ft**3')
    p.set_val('A', np.pi * c['R'] ** 2, units='ft**2')
    p.set_val('R', c['R'], units='ft')
    p.set_val('V_tip', c['V_tip'], units='ft/s')
    p.set_val('l_T', c['l_T'], units='ft')
    p.set_val('Dv_GW', np.full(nn, c['Dv_GW']))
    p.set_val('dAz_CD', c['dAz_CD'], units='ft**2')
    if tail_rotor == 'hover_power':
        p.set_val('P_MR', c['P_MR'], units='hp')
        p.set_val('tr_R', c['tr_R'], units='ft')
    else:
        p.set_val('v_hov_T', 40.0, units='ft/s')
    p.run_model()
    return p


# ------------------------------------------------------------------------
# book anchors
# ------------------------------------------------------------------------

def test_figure_2_3_ah1g_calculated_line():
    """Full equation of p. 98 on the AH-1G: within 3 % (4 hp at the low end)."""
    p = _run_climb(FIG_2_3[:, 0], AH1G)
    dP, book = p.get_val('dP', units='hp'), FIG_2_3[:, 1]
    assert np.all(np.abs(dP - book) <= np.maximum(4.0, 0.03 * book))


def test_figure_2_4_example_gap_with_appendix_data():
    """With the Appendix A data the full equation runs 4-10 % under Figure 2.4 (C2-6)."""
    p = _run_climb(FIG_2_4[:, 0], EXAMPLE)
    ratio = FIG_2_4[:, 1] / p.get_val('dP', units='hp')
    assert np.all((ratio > 1.03) & (ratio < 1.11))


@pytest.mark.parametrize('change, tol', [({'Dv_GW': 0.06}, 0.025), ({'GW': 21900.0}, 0.05)])
def test_figure_2_4_candidate_explanations(change, tol):
    """Either change alone reproduces Figure 2.4; both contradict Figure 4.36 (C2-6, closed)."""
    p = _run_climb(FIG_2_4[:, 0], dict(EXAMPLE, **change))
    assert_near_equal(p.get_val('dP', units='hp'), FIG_2_4[:, 1].astype(float), tol)


def test_quick_estimate_500_fpm_p101():
    """G.W. (R/C) / 66,000 = 150 h.p. at 500 ft/min; (V_c/2)^2 = 17 << v_1hov^2 = 1,521."""
    p = _run_climb(500.0, EXAMPLE)
    assert_near_equal(p.get_val('dP_low', units='hp')[0], 150.0, 0.02)
    assert p.get_val('dP_low')[0] < p.get_val('dP_mom')[0] < p.get_val('dP')[0]
    V_c, v_hov = p.get_val('V_c', units='ft/s')[0], p.get_val('v_hov')[0]
    assert (V_c / 2) ** 2 < 0.02 * v_hov ** 2


def test_collective_increment_500_fpm_p101():
    """C2-1: the equation gives 0.52 deg; the printed 0.4 deg is the value over Omega R alone."""
    p = _run_climb(500.0, EXAMPLE)
    comp = om.Problem()
    comp.model.add_subsystem('g3', ClimbCollectiveComp(), promotes=['*'])
    comp.setup()
    comp.set_val('v_sum', p.get_val('v_sum'))
    comp.set_val('v_hov', p.get_val('v_hov'))
    comp.set_val('V_tip', 650.0)
    comp.run_model()
    d_theta = comp.get_val('d_theta_0', units='deg')[0]
    assert_near_equal(d_theta, 0.52, 0.02)
    assert_near_equal(0.75 * d_theta, 0.4, 0.05)


# ------------------------------------------------------------------------
# internal consistency
# ------------------------------------------------------------------------

@pytest.mark.parametrize('tail_rotor', ['input', 'hover_power'])
def test_zero_increment_at_hover(tail_rotor):
    p = _run_climb(0.0, EXAMPLE, tail_rotor=tail_rotor)
    for name in ('dP', 'dP_mom', 'dP_low'):
        assert abs(p.get_val(name)[0]) < 1e-10


def test_tail_rotor_thrust_matches_p98():
    """T_T = P_M R_M / [(Omega R)_M l_T]; the Chapter 4 example gives 0.69 h.p._M (p. 309)."""
    p = _run_climb(500.0, EXAMPLE)
    assert_near_equal(p.get_val('T_T', units='lbf')[0], 0.69 * 1600.0, 0.01)


def test_full_equation_above_quick_estimate():
    """Vertical drag and tail rotor only add power."""
    p = _run_climb([300.0, 1500.0, 3000.0], EXAMPLE)
    assert np.all(p.get_val('dP') > p.get_val('dP_mom'))


def test_promoted_defaults_are_consistent_across_the_chapter():
    p = om.Problem()
    p.model.add_subsystem('g0', FlowStatesGroup(num_nodes=2), promotes=['*'])
    p.model.add_subsystem('g2', ClimbPowerGroup(num_nodes=2, tail_rotor='hover_power',
                                                inflow='external'),
                          promotes=['*'])
    p.model.add_subsystem('g3', ClimbCollectiveComp(num_nodes=2), promotes=['*'])
    p.setup()
    p.final_setup()


# ------------------------------------------------------------------------
# derivatives
# ------------------------------------------------------------------------

@pytest.mark.parametrize('tail_rotor', ['input', 'hover_power'])
def test_partials(tail_rotor):
    p = _run_climb([200.0, 1500.0, 3500.0], EXAMPLE, T=20800.0, tail_rotor=tail_rotor)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_partials_collective():
    p = om.Problem()
    p.model.add_subsystem('g3', ClimbCollectiveComp(num_nodes=3), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('v_sum', [40.0, 45.0, 60.0])
    p.set_val('v_hov', [39.0, 39.0, 39.0])
    p.set_val('V_tip', 650.0)
    p.run_model()
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_totals_hover_power_link():
    p = _run_climb([500.0, 2000.0], EXAMPLE)
    data = p.check_totals(of=['dP'], wrt=['V_c', 'GW', 'P_MR', 'tr_R', 'Dv_GW'],
                          method='cs', out_stream=None)
    assert_check_totals(data, atol=1e-8, rtol=1e-8)
