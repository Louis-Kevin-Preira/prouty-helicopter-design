"""Chapter 7 -- flapping equations in forward flight, p. 463-469."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.assert_utils import assert_near_equal

from prouty.flapping import ForwardFlightFlappingGroup
from prouty.flapping.closed_form_flapping_comp import ClosedFormFlappingComp
from prouty.flapping.flapping_matrix_comp import FlappingMatrixComp
from prouty.flapping.flapping_solve_comp import FlappingSolveComp
from prouty.flapping.thrust_inflow_comp import ThrustInflowComp


# ------------------------------------------------------------------------
# thrust_inflow_comp
# ------------------------------------------------------------------------

A_EX_thrust_inflow, SIGMA_EX_thrust_inflow, E_OVER_R_EX_thrust_inflow, MU_EX = 6.0, 0.08488, 0.05, 0.3


TWIST_EX = np.radians(-10.0)


def _run_thrust_inflow(nn=1, inflow='internal', **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', ThrustInflowComp(num_nodes=nn, inflow=inflow),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('e_over_R', E_OVER_R_EX_thrust_inflow)
    p.set_val('a', np.full(nn, A_EX_thrust_inflow))
    p.set_val('mu', np.full(nn, MU_EX))
    p.set_val('theta_1', np.full(nn, TWIST_EX))
    if inflow == 'internal':
        p.set_val('sigma', SIGMA_EX_thrust_inflow)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _bracket(theta_0, theta_1, alpha_s, B_1, mu):
    return (theta_0 * (2 / 3 + mu ** 2) + theta_1 * (0.5 + mu ** 2 / 2)
            + mu * (alpha_s - B_1))


def test_thrust_equation_as_printed():
    """C_T/sigma = (1-e/R)(a/4)[bracket - v1/OmegaR], p. 467."""
    th0, als = np.radians(14.0), np.radians(-5.0)
    p = _run_thrust_inflow(theta_0=th0, alpha_s=als)

    K = (1 - E_OVER_R_EX_thrust_inflow) * A_EX_thrust_inflow / 4
    S = _bracket(th0, TWIST_EX, als, 0.0, MU_EX)
    expected = K * (S - p.get_val('v1_over_OmegaR')[0])

    assert_near_equal(p.get_val('CT_sigma')[0], expected, 1e-13)


def test_momentum_inflow_as_printed():
    """v1/OmegaR = (C_T/sigma) sigma / (2 mu), p. 468."""
    p = _run_thrust_inflow(theta_0=np.radians(14.0), alpha_s=np.radians(-5.0))
    expected = p.get_val('CT_sigma')[0] * SIGMA_EX_thrust_inflow / (2 * MU_EX)
    assert_near_equal(p.get_val('v1_over_OmegaR')[0], expected, 1e-13)


def test_grouping_matches_naive_substitution():
    """2 mu K S / P is the same as K S / (1 + K sigma / 2 mu)."""
    th0, als = np.radians(14.0), np.radians(-5.0)
    K = (1 - E_OVER_R_EX_thrust_inflow) * A_EX_thrust_inflow / 4
    S = _bracket(th0, TWIST_EX, als, 0.0, MU_EX)
    naive = K * S / (1 + K * SIGMA_EX_thrust_inflow / (2 * MU_EX))

    p = _run_thrust_inflow(theta_0=th0, alpha_s=als)
    assert_near_equal(p.get_val('CT_sigma')[0], naive, 1e-13)


def test_zero_offset_recovers_chapter_3_form():
    """p. 166: C_T/sigma = (a/4)[theta_0(2/3+mu^2)+theta_1(1/2+mu^2/2)+lambda]."""
    th0, als, b1 = np.radians(14.0), np.radians(-5.0), np.radians(1.0)
    p = _run_thrust_inflow(inflow='external', e_over_R=0.0, theta_0=th0, alpha_s=als, B_1=b1,
             v1_over_OmegaR=0.0117)

    lam = MU_EX * (als - b1) - 0.0117
    expected = (A_EX_thrust_inflow / 4) * (th0 * (2 / 3 + MU_EX ** 2)
                             + TWIST_EX * (0.5 + MU_EX ** 2 / 2) + lam)
    assert_near_equal(p.get_val('CT_sigma')[0], expected, 1e-13)


def test_external_mode_reproduces_internal():
    """Feeding back the internal inflow gives the same thrust coefficient."""
    th0, als = np.radians(14.0), np.radians(-5.0)
    ref = _run_thrust_inflow(theta_0=th0, alpha_s=als)

    ext = _run_thrust_inflow(inflow='external', theta_0=th0, alpha_s=als,
               v1_over_OmegaR=ref.get_val('v1_over_OmegaR'))
    assert_near_equal(ext.get_val('CT_sigma')[0], ref.get_val('CT_sigma')[0],
                      1e-13)


def test_representative_forward_flight_point():
    """The collective needed for C_T/sigma = 0.0829 at 115 kt is plausible.

    20,000 lb at sea level, R = 30 ft, Omega R = 650 ft/s, A_b = 240 ft^2.
    """
    als = np.radians(-5.0)
    p = _run_thrust_inflow(theta_0=0.0, alpha_s=als)

    ct0 = p.get_val('CT_sigma')[0]
    dct = p.compute_totals(of=['CT_sigma'], wrt=['theta_0'])[
        ('CT_sigma', 'theta_0')][0, 0]
    theta_0 = (0.0829 - ct0) / dct

    assert 10.0 < np.degrees(theta_0) < 20.0


def test_stays_finite_at_zero_advance_ratio():
    """The rewritten grouping has no 1/mu, unlike the printed substitution."""
    p = _run_thrust_inflow(mu=0.0, theta_0=np.radians(14.0))

    assert np.isfinite(p.get_val('CT_sigma')[0])
    assert_near_equal(p.get_val('CT_sigma')[0], 0.0, 1e-14)

    K = (1 - E_OVER_R_EX_thrust_inflow) * A_EX_thrust_inflow / 4
    S = _bracket(np.radians(14.0), TWIST_EX, 0.0, 0.0, 0.0)
    assert_near_equal(p.get_val('v1_over_OmegaR')[0], S, 1e-13)


def test_inflow_reduces_thrust():
    """Solving the loop must give less thrust than ignoring the inflow."""
    th0 = np.radians(14.0)
    coupled = _run_thrust_inflow(theta_0=th0).get_val('CT_sigma')[0]
    no_inflow = _run_thrust_inflow(inflow='external', theta_0=th0,
                     v1_over_OmegaR=0.0).get_val('CT_sigma')[0]
    assert 0.0 < coupled < no_inflow


def test_vectorized_thrust_inflow():
    nn = 3
    mu = np.array([0.15, 0.30, 0.40])
    th0 = np.radians(np.array([10.0, 14.0, 16.0]))
    p = _run_thrust_inflow(nn, mu=mu, theta_0=th0, theta_1=np.full(nn, TWIST_EX),
             a=np.full(nn, A_EX_thrust_inflow))

    K = (1 - E_OVER_R_EX_thrust_inflow) * A_EX_thrust_inflow / 4
    S = _bracket(th0, TWIST_EX, 0.0, 0.0, mu)
    P = 2 * mu + K * SIGMA_EX_thrust_inflow
    assert_near_equal(p.get_val('CT_sigma'), 2 * mu * K * S / P, 1e-13)
    assert_near_equal(p.get_val('v1_over_OmegaR'), SIGMA_EX_thrust_inflow * K * S / P, 1e-13)



@pytest.mark.parametrize('inflow,nn', [('internal', 1), ('internal', 4),
                                       ('external', 1), ('external', 4)])
def test_partials_thrust_inflow(inflow, nn):
    extra = {} if inflow == 'internal' else {
        'v1_over_OmegaR': np.linspace(0.008, 0.015, nn)}
    p = _run_thrust_inflow(nn, inflow, mu=np.linspace(0.15, 0.40, nn),
             theta_0=np.radians(np.linspace(10.0, 16.0, nn)),
             theta_1=np.full(nn, TWIST_EX),
             alpha_s=np.radians(np.linspace(-6.0, -3.0, nn)),
             B_1=np.radians(np.linspace(0.5, 2.0, nn)),
             a=np.linspace(5.7, 6.1, nn), **extra)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_partials_at_zero_advance_ratio():
    p = _run_thrust_inflow(mu=0.0, theta_0=np.radians(14.0))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------------
# flapping_matrix_comp
# ------------------------------------------------------------------------

I_B_EX_flapping_matrix, MG_EX_flapping_matrix, MB_EX_flapping_matrix, E_EX = 2870.0, 151.05, 4860.0, 1.5


OMEGA_EX_flapping_matrix, GAMMA_EX_flapping_matrix = 650.0 / 30.0, 8.1


FLIGHT_flapping_matrix = dict(mu=0.3, theta_0=np.radians(14.0), theta_1=np.radians(-10.0),
              alpha_s=np.radians(-5.0), A_1=np.radians(-2.2),
              B_1=np.radians(1.5), v1_over_OmegaR=0.0117)


def _build_flapping_matrix(nn=1, e_over_R=0.05, include_weight=True, **over):
    p = om.Problem()
    p.model.add_subsystem('matrix',
                          FlappingMatrixComp(num_nodes=nn,
                                             include_weight=include_weight),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    p.set_val('I_b', I_B_EX_flapping_matrix)
    p.set_val('M_b_over_g', MG_EX_flapping_matrix)
    p.set_val('M_b', MB_EX_flapping_matrix)
    p.set_val('e', e_over_R * 30.0)
    p.set_val('e_over_R', e_over_R)
    p.set_val('Omega', np.full(nn, OMEGA_EX_flapping_matrix))
    p.set_val('gamma', np.full(nn, GAMMA_EX_flapping_matrix))
    for k, v in FLIGHT_flapping_matrix.items():
        p.set_val(k, np.full(nn, v))
    for k, v in over.items():
        p.set_val(k, v)

    p.run_model()
    return p


def _solve(p):
    """(a_0, a_1s, b_1s) at every node, solved as FlappingSolveComp does."""
    A, b = p.get_val('flap_A'), p.get_val('flap_b')
    return np.linalg.solve(A, b[..., None])[..., 0]


def _flap(p, node=0):
    return _solve(p)[node]


def test_system_is_actually_solved():
    p = _build_flapping_matrix()
    A, b = p.get_val('flap_A'), p.get_val('flap_b')
    assert_near_equal(np.einsum('nij,nj->ni', A, _solve(p)), b, 1e-12)


def test_scaling_groupings():
    """The matrix carries G, h and W as documented."""
    p = _build_flapping_matrix()
    A, b = p.get_val('flap_A')[0], p.get_val('flap_b')[0]

    x = 0.05
    G = GAMMA_EX_flapping_matrix * (1 - x) ** 2 / 2
    h = E_EX * MG_EX_flapping_matrix / I_B_EX_flapping_matrix
    W = MB_EX_flapping_matrix / (OMEGA_EX_flapping_matrix ** 2 * I_B_EX_flapping_matrix)

    assert_near_equal(A[0, 0], -(1 + h), 1e-13)
    assert_near_equal(A[1, 2], h, 1e-13)
    assert_near_equal(A[2, 1], h, 1e-13)
    assert_near_equal(A[0, 1], G * FLIGHT_flapping_matrix['mu'] * x / 4, 1e-13)

    # h is exactly (omega_n/Omega)^2 - 1, p. 456
    assert_near_equal(h, 1.5 * x / (1 - x), 1e-3)
    assert A[0, 2] == 0.0 and A[1, 0] == 0.0
    assert np.isfinite(b).all()


def test_coning_matches_page_467_at_zero_offset():
    """Row 1 decouples at e/R = 0 and reproduces the printed coning."""
    p = _build_flapping_matrix(e_over_R=0.0)
    a0 = _flap(p)[0]

    mu = FLIGHT_flapping_matrix['mu']
    bracket = (FLIGHT_flapping_matrix['theta_0'] * (0.75 + 0.75 * mu ** 2)
               + FLIGHT_flapping_matrix['theta_1'] * (0.6 + mu ** 2 / 2)
               + mu * (FLIGHT_flapping_matrix['alpha_s'] - FLIGHT_flapping_matrix['B_1'])
               - FLIGHT_flapping_matrix['v1_over_OmegaR'])
    expected = (GAMMA_EX_flapping_matrix / 6) * bracket - MB_EX_flapping_matrix / (OMEGA_EX_flapping_matrix ** 2 * I_B_EX_flapping_matrix)

    assert_near_equal(a0, expected, 1e-12)


def test_dropping_e_over_R_from_brackets_costs_a_lot_on_coning():
    """Documents entry C7-8 rather than accepting the p. 466 simplification.

    p. 466 says e/R can reasonably be eliminated from inside the square
    brackets. That holds for a_1s and b_1s, whose brackets are dominated by a
    single term, but not for coning: the constant bracket is a difference of
    comparable contributions (collective against twist against inflow), so a
    3 % change in each term moves the sum by 20 %.
    """
    x, mu = 0.05, FLIGHT_flapping_matrix['mu']
    h = E_EX * MG_EX_flapping_matrix / I_B_EX_flapping_matrix

    a0 = _flap(_build_flapping_matrix(e_over_R=x))[0]

    bracket = (FLIGHT_flapping_matrix['theta_0'] * (0.75 + 0.75 * mu ** 2)
               + FLIGHT_flapping_matrix['theta_1'] * (0.6 + mu ** 2 / 2)
               + mu * (FLIGHT_flapping_matrix['alpha_s'] - FLIGHT_flapping_matrix['B_1'])
               - FLIGHT_flapping_matrix['v1_over_OmegaR'])
    reduced = ((GAMMA_EX_flapping_matrix / 6) * (1 - x) ** 2 * bracket
               - MB_EX_flapping_matrix / (OMEGA_EX_flapping_matrix ** 2 * I_B_EX_flapping_matrix)) / (1 + h)

    assert 0.15 < a0 / reduced - 1.0 < 0.30
    assert a0 > reduced


def test_longitudinal_flapping_matches_page_469_at_zero_offset():
    """The sine row alone gives the basic a_1s printed on p. 469."""
    p = _build_flapping_matrix(e_over_R=0.0)
    a1s = _flap(p)[1]

    mu = FLIGHT_flapping_matrix['mu']
    lam = mu * FLIGHT_flapping_matrix['alpha_s'] - FLIGHT_flapping_matrix['v1_over_OmegaR']
    expected = ((8 / 3) * FLIGHT_flapping_matrix['theta_0'] * mu + 2 * FLIGHT_flapping_matrix['theta_1'] * mu
                - FLIGHT_flapping_matrix['B_1'] * (1 + 1.5 * mu ** 2) + 2 * mu * lam) \
        / (1 - mu ** 2 / 2)

    assert_near_equal(a1s, expected, 1e-12)


def test_lateral_flapping_matches_the_chapter_3_relation():
    """Validates entry C7-3: the cosine row uses 1/4 for v_1, not 1/3.

    p. 474 quotes from Chapter 3:
        A_1 - b_1s = -[(4/3) mu a_0 + v_1/(Omega R)] / (1 + mu^2/2)
    Only the p. 466 coefficient of 1/4 reproduces it; the 1/3 printed on
    p. 468 would give (4/3) v_1/(Omega R).
    """
    p = _build_flapping_matrix(e_over_R=0.0)
    a0, _, b1s = _flap(p)

    mu = FLIGHT_flapping_matrix['mu']
    expected = -((4 / 3) * mu * a0 + FLIGHT_flapping_matrix['v1_over_OmegaR']) \
        / (1 + mu ** 2 / 2)

    assert_near_equal(FLIGHT_flapping_matrix['A_1'] - b1s, expected, 1e-12)

    # The printed 1/3 would be off by exactly this much
    wrong = -((4 / 3) * mu * a0 + (4 / 3) * FLIGHT_flapping_matrix['v1_over_OmegaR']) \
        / (1 + mu ** 2 / 2)
    assert abs(wrong - expected) > 1e-3


def test_three_by_three_coupling_is_real():
    """Coning and a_1s are coupled once e/R is nonzero, so a 2x2 will not do."""
    p = _build_flapping_matrix(e_over_R=0.05)
    A = p.get_val('flap_A')[0]

    assert A[0, 1] != 0.0   # mu (e/R) a_1s / 4 in M_A,const
    assert A[2, 0] != 0.0   # -mu a_0 [1/3 + e/6R] in M_A,cosine

    zero = _build_flapping_matrix(e_over_R=0.0).get_val('flap_A')[0]
    assert zero[0, 1] == 0.0
    assert zero[2, 0] != 0.0


def test_weight_option():
    """Dropping M_W removes the gravity term the p. 468 closed forms ignore."""
    heavy = _flap(_build_flapping_matrix())[0]
    light = _flap(_build_flapping_matrix(include_weight=False))[0]

    h = E_EX * MG_EX_flapping_matrix / I_B_EX_flapping_matrix
    shift = MB_EX_flapping_matrix / (OMEGA_EX_flapping_matrix ** 2 * I_B_EX_flapping_matrix) / (1 + h)
    assert_near_equal(light - heavy, shift, 2e-2)
    assert light > heavy


def test_flapping_magnitudes_are_plausible():
    p = _build_flapping_matrix()
    a0, a1s, b1s = _flap(p)
    assert 2.0 < np.degrees(a0) < 8.0
    assert 0.0 < np.degrees(a1s) < 10.0
    assert abs(np.degrees(b1s)) < 6.0


def test_vectorized_flapping_matrix():
    nn = 3
    mu = np.array([0.15, 0.30, 0.40])
    p = _build_flapping_matrix(nn, mu=mu, theta_0=np.full(nn, FLIGHT_flapping_matrix['theta_0']),
               theta_1=np.full(nn, FLIGHT_flapping_matrix['theta_1']),
               alpha_s=np.full(nn, FLIGHT_flapping_matrix['alpha_s']),
               A_1=np.full(nn, FLIGHT_flapping_matrix['A_1']), B_1=np.full(nn, FLIGHT_flapping_matrix['B_1']),
               v1_over_OmegaR=np.full(nn, FLIGHT_flapping_matrix['v1_over_OmegaR']))

    A, b, x = p.get_val('flap_A'), p.get_val('flap_b'), _solve(p)
    assert_near_equal(np.einsum('nij,nj->ni', A, x), b, 1e-12)
    assert np.all(np.diff(x[:, 1]) > 0)   # a_1s grows with advance ratio



@pytest.mark.parametrize('nn,weight', [(1, True), (4, True), (4, False)])
def test_partials_flapping_matrix(nn, weight):
    p = _build_flapping_matrix(nn, include_weight=weight,
               mu=np.linspace(0.15, 0.40, nn),
               gamma=np.linspace(6.9, 8.1, nn),
               Omega=np.linspace(19.0, 21.7, nn),
               theta_0=np.radians(np.linspace(10.0, 16.0, nn)),
               theta_1=np.full(nn, FLIGHT_flapping_matrix['theta_1']),
               alpha_s=np.radians(np.linspace(-6.0, -3.0, nn)),
               A_1=np.radians(np.linspace(-2.5, -1.5, nn)),
               B_1=np.radians(np.linspace(0.5, 2.0, nn)),
               v1_over_OmegaR=np.linspace(0.008, 0.015, nn))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_matrix_is_well_conditioned():
    """The Omega^2 I_b scaling should keep the system O(1) across the envelope."""
    for mu in (0.0, 0.15, 0.30, 0.45):
        for x in (0.0, 0.05, 0.13):
            p = _build_flapping_matrix(e_over_R=x, mu=mu)
            A = p.get_val('flap_A')[0]
            assert np.linalg.cond(A) < 50.0


# ------------------------------------------------------------------------
# flapping_solve_comp
# ------------------------------------------------------------------------

I_B_EX_flapping_solve, MG_EX_flapping_solve, MB_EX_flapping_solve = 2870.0, 151.05, 4860.0


OMEGA_EX_flapping_solve, GAMMA_EX_flapping_solve = 650.0 / 30.0, 8.1


def _random_system(nn, seed=0):
    """Diagonally dominant, so well conditioned and safely invertible."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nn, 3, 3))
    A += 4.0 * np.tile(np.eye(3), (nn, 1, 1))
    return A, rng.normal(size=(nn, 3))


def _run_flapping_solve(A, b):
    nn = A.shape[0]
    p = om.Problem()
    p.model.add_subsystem('comp', FlappingSolveComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('flap_A', A)
    p.set_val('flap_b', b)
    p.run_model()
    return p


def _chained(nn=1, **over):
    """FlappingMatrixComp -> FlappingSolveComp, as wired in G2."""
    p = om.Problem()
    m = p.model
    m.add_subsystem('matrix', FlappingMatrixComp(num_nodes=nn), promotes=['*'])
    m.add_subsystem('solve', FlappingSolveComp(num_nodes=nn), promotes=['*'])
    p.setup(force_alloc_complex=True)

    p.set_val('I_b', I_B_EX_flapping_solve)
    p.set_val('M_b_over_g', MG_EX_flapping_solve)
    p.set_val('M_b', MB_EX_flapping_solve)
    p.set_val('e', 1.5)
    p.set_val('e_over_R', 0.05)
    p.set_val('Omega', np.full(nn, OMEGA_EX_flapping_solve))
    p.set_val('gamma', np.full(nn, GAMMA_EX_flapping_solve))
    p.set_val('mu', np.full(nn, 0.3))
    p.set_val('theta_0', np.full(nn, np.radians(14.0)))
    p.set_val('theta_1', np.full(nn, np.radians(-10.0)))
    p.set_val('alpha_s', np.full(nn, np.radians(-5.0)))
    p.set_val('A_1', np.full(nn, np.radians(-2.2)))
    p.set_val('B_1', np.full(nn, np.radians(1.5)))
    p.set_val('v1_over_OmegaR', np.full(nn, 0.0117))
    for k, v in over.items():
        p.set_val(k, v)

    p.run_model()
    return p


def test_residual_is_zero():
    A, b = _random_system(4)
    p = _run_flapping_solve(A, b)
    x = np.column_stack([p.get_val(n) for n in ('a_0', 'a_1s', 'b_1s')])
    assert_near_equal(np.einsum('nij,nj->ni', A, x), b, 1e-12)


def test_matches_numpy():
    A, b = _random_system(3, seed=7)
    p = _run_flapping_solve(A, b)
    ref = np.linalg.solve(A, b[..., None])[..., 0]
    for k, name in enumerate(('a_0', 'a_1s', 'b_1s')):
        assert_near_equal(p.get_val(name), ref[:, k], 1e-13)


def test_shapes_are_uniform_at_one_node():
    """The reason for not using om.LinearSystemComp: no leading-dim squeeze."""
    A, b = _random_system(1)
    p = _run_flapping_solve(A, b)
    assert p.get_val('flap_A').shape == (1, 3, 3)
    assert p.get_val('flap_b').shape == (1, 3)
    assert p.get_val('a_0').shape == (1,)


def test_identity_system_passes_the_right_hand_side_through():
    nn = 2
    A = np.tile(np.eye(3), (nn, 1, 1))
    b = np.array([[0.08, 0.05, -0.01], [0.06, 0.04, -0.02]])
    p = _run_flapping_solve(A, b)
    assert_near_equal(p.get_val('a_0'), b[:, 0], 1e-14)
    assert_near_equal(p.get_val('a_1s'), b[:, 1], 1e-14)
    assert_near_equal(p.get_val('b_1s'), b[:, 2], 1e-14)


def test_nodes_are_independent():
    """Perturbing one node must not move the others."""
    A, b = _random_system(3, seed=3)
    ref = _run_flapping_solve(A, b)

    b2 = b.copy()
    b2[1] += 0.5
    alt = _run_flapping_solve(A, b2)

    for name in ('a_0', 'a_1s', 'b_1s'):
        assert_near_equal(alt.get_val(name)[[0, 2]],
                          ref.get_val(name)[[0, 2]], 1e-14)
        assert abs(alt.get_val(name)[1] - ref.get_val(name)[1]) > 1e-6



@pytest.mark.parametrize('nn', [1, 4])
def test_partials_flapping_solve(nn):
    A, b = _random_system(nn, seed=11)
    p = _run_flapping_solve(A, b)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_chained_gives_the_expected_flapping():
    p = _chained()
    assert 2.0 < np.degrees(p.get_val('a_0')[0]) < 8.0
    assert 0.0 < np.degrees(p.get_val('a_1s')[0]) < 10.0
    assert abs(np.degrees(p.get_val('b_1s')[0])) < 6.0


def test_chained_reproduces_page_469_at_zero_offset():
    p = _chained(e_over_R=0.0, e=0.0)

    mu = 0.3
    lam = mu * np.radians(-5.0) - 0.0117
    expected = ((8 / 3) * np.radians(14.0) * mu + 2 * np.radians(-10.0) * mu
                - np.radians(1.5) * (1 + 1.5 * mu ** 2) + 2 * mu * lam) \
        / (1 - mu ** 2 / 2)

    assert_near_equal(p.get_val('a_1s')[0], expected, 1e-12)


def test_chained_totals():
    """Analytic totals through matrix assembly and solve, in one shot."""
    nn = 2
    p = _chained(nn, mu=np.array([0.2, 0.35]))

    data = p.check_totals(
        of=['a_0', 'a_1s', 'b_1s'],
        wrt=['e_over_R', 'gamma', 'mu', 'theta_0', 'theta_1', 'alpha_s',
             'A_1', 'B_1', 'v1_over_OmegaR', 'I_b', 'M_b', 'e'],
        method='cs', compact_print=True, out_stream=None)

    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-8, atol=1e-11)


def test_group_stays_feed_forward():
    p = _chained(nn=2, mu=np.array([0.2, 0.35]))
    before = p.get_val('a_1s').copy()
    p.run_model()
    assert_near_equal(p.get_val('a_1s'), before, 1e-14)


# ------------------------------------------------------------------------
# closed_form_flapping_comp
# ------------------------------------------------------------------------

I_B_EX_closed_form_flapping, MG_EX_closed_form_flapping, MB_EX_closed_form_flapping, R_EX_closed_form_flapping = 2870.0, 151.05, 4860.0, 30.0


OMEGA_EX_closed_form_flapping, GAMMA_EX_closed_form_flapping, A_EX_closed_form_flapping, SIGMA_EX_closed_form_flapping = 650.0 / 30.0, 8.1, 6.0, 0.08488


FLIGHT_closed_form_flapping = dict(mu=0.3, CT_sigma=0.0829, v1_over_OmegaR=0.0117,
              theta_0=np.radians(14.0), theta_1=np.radians(-10.0),
              alpha_s=np.radians(-5.0), A_1=np.radians(-2.2),
              B_1=np.radians(1.5))


def _run_closed_form_flapping(nn=1, e_over_R=0.05, exact_denominator=True, include_weight=True,
         **over):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', ClosedFormFlappingComp(num_nodes=nn,
                                       exact_denominator=exact_denominator,
                                       include_weight=include_weight),
        promotes=['*'])
    p.setup(force_alloc_complex=True)

    p.set_val('e_over_R', e_over_R)
    p.set_val('gamma', np.full(nn, GAMMA_EX_closed_form_flapping))
    p.set_val('a', np.full(nn, A_EX_closed_form_flapping))
    for k, v in FLIGHT_closed_form_flapping.items():
        p.set_val(k, np.full(nn, v))
    if include_weight:
        p.set_val('R', R_EX_closed_form_flapping)
        p.set_val('Omega', np.full(nn, OMEGA_EX_closed_form_flapping))
    for k, v in over.items():
        p.set_val(k, v)

    p.run_model()
    return p


def _pieces(x, mu, a0, gamma=GAMMA_EX_closed_form_flapping):
    """N1, N2 and kappa as the docstring defines them."""
    N1 = ((8 / 3) * FLIGHT_closed_form_flapping['theta_0'] * mu + 2 * FLIGHT_closed_form_flapping['theta_1'] * mu
          - FLIGHT_closed_form_flapping['B_1'] * (1 + 1.5 * mu ** 2)
          + 2 * mu * (mu * FLIGHT_closed_form_flapping['alpha_s'] - FLIGHT_closed_form_flapping['v1_over_OmegaR']))
    N2 = (FLIGHT_closed_form_flapping['A_1'] * (1 + mu ** 2 / 2) + (4 / 3) * mu * a0
          + FLIGHT_closed_form_flapping['v1_over_OmegaR'])
    kap = 12 * x / (gamma * (1 - x) ** 3)
    return N1, N2, kap


def test_coning_matches_page_467():
    p = _run_closed_form_flapping()
    x = 0.05
    aero = (2 / 3) * GAMMA_EX_closed_form_flapping / A_EX_closed_form_flapping * FLIGHT_closed_form_flapping['CT_sigma'] \
        * (1 - x) ** 2 / (1 + x / 2)
    grav = 1.5 * 32.174 * R_EX_closed_form_flapping / (OMEGA_EX_closed_form_flapping * R_EX_closed_form_flapping) ** 2 / (1 + x / 2)
    assert_near_equal(p.get_val('a_0')[0], aero - grav, 1e-13)


def test_weight_option_matches_page_468():
    """p. 468 drops the gravity term when deriving the flapping equations."""
    p = _run_closed_form_flapping(include_weight=False)
    x = 0.05
    aero = (2 / 3) * GAMMA_EX_closed_form_flapping / A_EX_closed_form_flapping * FLIGHT_closed_form_flapping['CT_sigma'] \
        * (1 - x) ** 2 / (1 + x / 2)
    assert_near_equal(p.get_val('a_0')[0], aero, 1e-13)
    assert p.get_val('a_0')[0] > _run_closed_form_flapping().get_val('a_0')[0]


def test_matches_printed_page_468_expression():
    """The 2x2 solve is the nested fraction of p. 468, written out."""
    p = _run_closed_form_flapping()
    x, mu = 0.05, FLIGHT_closed_form_flapping['mu']
    N1, N2, kap = _pieces(x, mu, p.get_val('a_0')[0])

    D = (1 - mu ** 2 / 2) + 144 * x ** 2 \
        / (GAMMA_EX_closed_form_flapping ** 2 * (1 - x) ** 6 * (1 + mu ** 2 / 2))
    printed = N1 / D + 12 * x / (GAMMA_EX_closed_form_flapping * (1 - x) ** 3 * (1 + mu ** 2 / 2)) \
        * N2 / D

    assert_near_equal(p.get_val('a_1s')[0], printed, 1e-12)


def test_matches_printed_page_469_expressions():
    """With kappa^2 dropped, and (1 - mu^4/4) on both, per entry C7-4."""
    p = _run_closed_form_flapping(exact_denominator=False)
    x, mu = 0.05, FLIGHT_closed_form_flapping['mu']
    N1, N2, kap = _pieces(x, mu, p.get_val('a_0')[0])

    a1s = N1 / (1 - mu ** 2 / 2) + kap * N2 / (1 - mu ** 4 / 4)
    b1s = N2 / (1 + mu ** 2 / 2) - kap * N1 / (1 - mu ** 4 / 4)

    assert_near_equal(p.get_val('a_1s')[0], a1s, 1e-12)
    assert_near_equal(p.get_val('b_1s')[0], b1s, 1e-12)



@pytest.mark.parametrize('mu', [0.15, 0.30, 0.45])
def test_printed_mu4_sign_shifts_a1s_by_a_known_amount(mu):
    """Entry C7-4: p. 469 prints (1 + mu^4/4) on a_1s, (1 - mu^4/4) on b_1s.

    The gap is kappa N_2 [1/(1 - mu^4/4) - 1/(1 + mu^4/4)], which grows as
    mu^4. It is invisible at mu = 0.3 for this particular trim only because
    N_2 nearly vanishes there; the identity below holds regardless.
    """
    p = _run_closed_form_flapping(exact_denominator=False, mu=mu)
    x = 0.05
    N1, N2, kap = _pieces(x, mu, p.get_val('a_0')[0])

    printed = N1 / (1 - mu ** 2 / 2) + kap * N2 / (1 + mu ** 4 / 4)
    gap = kap * N2 * (1 / (1 - mu ** 4 / 4) - 1 / (1 + mu ** 4 / 4))

    assert_near_equal(p.get_val('a_1s')[0] - printed, gap, 1e-9)


def test_exact_and_simplified_denominators_are_close():
    """kappa^2 is small: 12(0.05)/(8.1 x 0.857) is about 0.086."""
    exact = _run_closed_form_flapping().get_val('a_1s')[0]
    simple = _run_closed_form_flapping(exact_denominator=False).get_val('a_1s')[0]
    assert abs(exact / simple - 1.0) < 0.02


def test_zero_offset_gives_the_basic_flapping():
    """kappa = 0 decouples the 2x2, leaving the p. 469 leading terms."""
    p = _run_closed_form_flapping(e_over_R=0.0)
    mu = FLIGHT_closed_form_flapping['mu']
    N1, N2, _ = _pieces(0.0, mu, p.get_val('a_0')[0])

    assert_near_equal(p.get_val('a_1s')[0], N1 / (1 - mu ** 2 / 2), 1e-13)
    assert_near_equal(p.get_val('b_1s')[0], N2 / (1 + mu ** 2 / 2), 1e-13)


def test_agrees_with_the_numerical_path_at_zero_offset():
    """Both paths must coincide when every e/R simplification is inert."""
    closed = _run_closed_form_flapping(e_over_R=0.0, include_weight=False)

    num = om.Problem()
    m = num.model
    m.add_subsystem('matrix', FlappingMatrixComp(include_weight=False),
                    promotes=['*'])
    m.add_subsystem('solve', FlappingSolveComp(), promotes=['*'])
    num.setup()
    num.set_val('I_b', I_B_EX_closed_form_flapping)
    num.set_val('M_b_over_g', MG_EX_closed_form_flapping)
    num.set_val('e', 0.0)
    num.set_val('e_over_R', 0.0)
    num.set_val('Omega', OMEGA_EX_closed_form_flapping)
    num.set_val('gamma', GAMMA_EX_closed_form_flapping)
    for k, v in FLIGHT_closed_form_flapping.items():
        if k != 'CT_sigma':
            num.set_val(k, v)
    num.run_model()

    # a_1s comes from the sine row alone and must agree outright
    np.testing.assert_allclose(closed.get_val('a_1s')[0],
                               num.get_val('a_1s')[0], atol=1e-12)

    # The numerical path builds coning from the moments, the closed form from
    # C_T/sigma, so b_1s differs only by the coning gap fed through the cosine
    # row.
    mu = FLIGHT_closed_form_flapping['mu']
    da0 = closed.get_val('a_0')[0] - num.get_val('a_0')[0]
    shift = (4 / 3) * mu * da0 / (1 + mu ** 2 / 2)
    np.testing.assert_allclose(
        closed.get_val('b_1s')[0] - num.get_val('b_1s')[0], shift, atol=1e-12)


def test_printed_coning_factor_in_N2_is_higher():
    """Third deviation: p. 469 writes 1/(1 + 1.5 e/R) where the coning of
    p. 467 carries (1 - e/R)^2 / (1 + e/2R). 5.6 % apart at e/R = 0.05."""
    x = 0.05
    consistent = (1 - x) ** 2 / (1 + x / 2)
    printed = 1.0 / (1 + 1.5 * x)
    assert_near_equal(printed / consistent, 1.056, 1e-3)


def test_vectorized_closed_form_flapping():
    nn = 3
    mu = np.array([0.15, 0.30, 0.45])
    p = _run_closed_form_flapping(nn, mu=mu, CT_sigma=np.full(nn, FLIGHT_closed_form_flapping['CT_sigma']),
             v1_over_OmegaR=np.full(nn, FLIGHT_closed_form_flapping['v1_over_OmegaR']),
             theta_0=np.full(nn, FLIGHT_closed_form_flapping['theta_0']),
             theta_1=np.full(nn, FLIGHT_closed_form_flapping['theta_1']),
             alpha_s=np.full(nn, FLIGHT_closed_form_flapping['alpha_s']),
             A_1=np.full(nn, FLIGHT_closed_form_flapping['A_1']), B_1=np.full(nn, FLIGHT_closed_form_flapping['B_1']))
    assert np.all(np.diff(p.get_val('a_1s')) > 0)
    assert np.all(np.isfinite(p.get_val('b_1s')))



@pytest.mark.parametrize('nn,exact,weight',
                         [(1, True, True), (4, True, True),
                          (4, False, True), (4, True, False)])
def test_partials_closed_form_flapping(nn, exact, weight):
    p = _run_closed_form_flapping(nn, exact_denominator=exact, include_weight=weight,
             gamma=np.linspace(6.9, 8.1, nn), a=np.linspace(5.7, 6.1, nn),
             mu=np.linspace(0.15, 0.40, nn),
             CT_sigma=np.linspace(0.06, 0.085, nn),
             v1_over_OmegaR=np.linspace(0.008, 0.020, nn),
             theta_0=np.radians(np.linspace(10.0, 16.0, nn)),
             theta_1=np.full(nn, FLIGHT_closed_form_flapping['theta_1']),
             alpha_s=np.radians(np.linspace(-6.0, -3.0, nn)),
             A_1=np.radians(np.linspace(-2.5, -1.5, nn)),
             B_1=np.radians(np.linspace(0.5, 2.0, nn)))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------------
# forward_flight_flapping_group
# ------------------------------------------------------------------------

R_EX_forward_flight_flapping_grp, I_B_EX_forward_flight_flapping_grp, E_OVER_R_EX_forward_flight_flapping_grp = 30.0, 2870.0, 0.05


A_EX_forward_flight_flapping_grp, SIGMA_EX_forward_flight_flapping_grp, OMEGA_EX_forward_flight_flapping_grp, GAMMA_EX_forward_flight_flapping_grp = 6.0, 0.08488, 650.0 / 30.0, 8.1


FLIGHT_forward_flight_flapping_grp = dict(mu=0.3, theta_0=np.radians(14.0), theta_1=np.radians(-10.0),
              alpha_s=np.radians(-5.0), A_1=np.radians(-2.2),
              B_1=np.radians(1.5))


def _build_forward_flight_flapping_grp(nn=1, e_over_R=E_OVER_R_EX_forward_flight_flapping_grp, **opts):
    over = {k: opts.pop(k) for k in list(opts) if k in
            ('mu', 'theta_0', 'theta_1', 'alpha_s', 'A_1', 'B_1',
             'v1_over_OmegaR', 'gamma')}

    p = om.Problem()
    p.model.add_subsystem('ff', ForwardFlightFlappingGroup(num_nodes=nn, **opts),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    numerical = opts.get('method', 'numerical') == 'numerical'
    builds_inertia = numerical and opts.get('blade_input', 'I_b') != 'external'
    with_weight = opts.get('include_weight', True)
    needs_R = builds_inertia or (not numerical and with_weight)
    needs_Omega = numerical or with_weight

    p.set_val('e_over_R', e_over_R)
    if needs_R:
        p.set_val('R', R_EX_forward_flight_flapping_grp)
    p.set_val('a', np.full(nn, A_EX_forward_flight_flapping_grp))
    p.set_val('gamma', np.full(nn, GAMMA_EX_forward_flight_flapping_grp))
    if needs_Omega:
        p.set_val('Omega', np.full(nn, OMEGA_EX_forward_flight_flapping_grp))
    if opts.get('inflow', 'internal') == 'internal':
        p.set_val('sigma', SIGMA_EX_forward_flight_flapping_grp)
    if builds_inertia and opts.get('blade_input', 'I_b') == 'I_b':
        p.set_val('I_b_ref', I_B_EX_forward_flight_flapping_grp)
    for k, v in FLIGHT_forward_flight_flapping_grp.items():
        p.set_val(k, np.full(nn, v))
    for k, v in over.items():
        p.set_val(k, v)

    p.run_model()
    return p


def _angles(p, node=0):
    return tuple(np.degrees(p.get_val(n)[node]) for n in
                 ('a_0', 'a_1s', 'b_1s'))


def test_default_chain_runs_and_is_plausible():
    p = _build_forward_flight_flapping_grp()
    a0, a1s, b1s = _angles(p)

    assert 2.0 < a0 < 8.0
    assert 0.0 < a1s < 10.0
    assert abs(b1s) < 6.0
    assert 0.02 < p.get_val('CT_sigma')[0] < 0.15
    assert 0.0 < p.get_val('v1_over_OmegaR')[0] < 0.05


def test_inertia_is_built_inside_the_group():
    """blade_input='I_b' spares the caller from wiring e and M_b by hand."""
    p = _build_forward_flight_flapping_grp()
    assert_near_equal(p.get_val('e')[0], E_OVER_R_EX_forward_flight_flapping_grp * R_EX_forward_flight_flapping_grp, 1e-13)
    assert_near_equal(p.get_val('I_b')[0], I_B_EX_forward_flight_flapping_grp, 1e-13)
    assert_near_equal(p.get_val('M_b_over_g')[0],
                      1.5 * I_B_EX_forward_flight_flapping_grp / (R_EX_forward_flight_flapping_grp * (1 - E_OVER_R_EX_forward_flight_flapping_grp)), 1e-13)


def test_blade_input_modes_agree():
    m_equiv = 3 * I_B_EX_forward_flight_flapping_grp / (R_EX_forward_flight_flapping_grp ** 3 * (1 - E_OVER_R_EX_forward_flight_flapping_grp) ** 3)

    ref = _build_forward_flight_flapping_grp()
    alt = om.Problem()
    alt.model.add_subsystem('ff', ForwardFlightFlappingGroup(blade_input='m'),
                            promotes=['*'])
    alt.setup()
    alt.set_val('e_over_R', E_OVER_R_EX_forward_flight_flapping_grp)
    alt.set_val('R', R_EX_forward_flight_flapping_grp)
    alt.set_val('m', m_equiv)
    alt.set_val('a', A_EX_forward_flight_flapping_grp)
    alt.set_val('gamma', GAMMA_EX_forward_flight_flapping_grp)
    alt.set_val('Omega', OMEGA_EX_forward_flight_flapping_grp)
    alt.set_val('sigma', SIGMA_EX_forward_flight_flapping_grp)
    for k, v in FLIGHT_forward_flight_flapping_grp.items():
        alt.set_val(k, v)
    alt.run_model()

    for name in ('a_0', 'a_1s', 'b_1s'):
        assert_near_equal(alt.get_val(name)[0], ref.get_val(name)[0], 1e-11)


def test_external_inflow_reproduces_internal():
    ref = _build_forward_flight_flapping_grp()
    ext = _build_forward_flight_flapping_grp(inflow='external',
                 v1_over_OmegaR=ref.get_val('v1_over_OmegaR'))

    for name in ('a_0', 'a_1s', 'b_1s', 'CT_sigma'):
        assert_near_equal(ext.get_val(name)[0], ref.get_val(name)[0], 1e-12)


def test_both_methods_agree_at_zero_offset():
    """Every e/R simplification is inert at e/R = 0, so the routes coincide.

    The two build coning differently (hinge moments against C_T/sigma), so
    b_1s carries that gap through the cosine equation; a_1s does not.
    """
    num = _build_forward_flight_flapping_grp(e_over_R=0.0, include_weight=False)
    clo = _build_forward_flight_flapping_grp(e_over_R=0.0, include_weight=False, method='closed_form')

    np.testing.assert_allclose(clo.get_val('a_1s')[0], num.get_val('a_1s')[0],
                               atol=1e-12)

    mu = FLIGHT_forward_flight_flapping_grp['mu']
    da0 = clo.get_val('a_0')[0] - num.get_val('a_0')[0]
    shift = (4 / 3) * mu * da0 / (1 + mu ** 2 / 2)
    np.testing.assert_allclose(clo.get_val('b_1s')[0] - num.get_val('b_1s')[0],
                               shift, atol=1e-12)


def test_methods_diverge_once_the_offset_is_real():
    """Quantifies what the p. 466 bracket simplification costs, entry C7-8."""
    num = _build_forward_flight_flapping_grp(include_weight=False, mu=0.45)
    clo = _build_forward_flight_flapping_grp(include_weight=False, method='closed_form', mu=0.45)

    assert abs(np.degrees(clo.get_val('a_1s')[0] - num.get_val('a_1s')[0])) > 0.5
    assert abs(np.degrees(clo.get_val('b_1s')[0] - num.get_val('b_1s')[0])) < 0.3


def test_weight_lowers_the_coning():
    heavy = _build_forward_flight_flapping_grp().get_val('a_0')[0]
    light = _build_forward_flight_flapping_grp(include_weight=False).get_val('a_0')[0]
    assert light > heavy


def test_flapping_trends_with_advance_ratio():
    nn = 5
    mu = np.array([0.10, 0.20, 0.30, 0.40, 0.45])
    p = _build_forward_flight_flapping_grp(nn, mu=mu, theta_0=np.full(nn, FLIGHT_forward_flight_flapping_grp['theta_0']),
               theta_1=np.full(nn, FLIGHT_forward_flight_flapping_grp['theta_1']),
               alpha_s=np.full(nn, FLIGHT_forward_flight_flapping_grp['alpha_s']),
               A_1=np.full(nn, FLIGHT_forward_flight_flapping_grp['A_1']), B_1=np.full(nn, FLIGHT_forward_flight_flapping_grp['B_1']))

    assert np.all(np.diff(p.get_val('a_1s')) > 0)          # rotor tilts back
    assert np.all(np.diff(p.get_val('v1_over_OmegaR')) < 0)  # inflow decays


def test_group_is_feed_forward():
    p = _build_forward_flight_flapping_grp(nn=2, mu=np.array([0.2, 0.35]),
               theta_0=np.full(2, FLIGHT_forward_flight_flapping_grp['theta_0']),
               theta_1=np.full(2, FLIGHT_forward_flight_flapping_grp['theta_1']),
               alpha_s=np.full(2, FLIGHT_forward_flight_flapping_grp['alpha_s']),
               A_1=np.full(2, FLIGHT_forward_flight_flapping_grp['A_1']), B_1=np.full(2, FLIGHT_forward_flight_flapping_grp['B_1']))
    before = p.get_val('a_1s').copy()
    p.run_model()
    assert_near_equal(p.get_val('a_1s'), before, 1e-14)



@pytest.mark.parametrize('method', ['numerical', 'closed_form'])
def test_totals(method):
    nn = 3
    mu = np.array([0.15, 0.30, 0.45])
    p = _build_forward_flight_flapping_grp(nn, method=method, mu=mu,
               theta_0=np.full(nn, FLIGHT_forward_flight_flapping_grp['theta_0']),
               theta_1=np.full(nn, FLIGHT_forward_flight_flapping_grp['theta_1']),
               alpha_s=np.full(nn, FLIGHT_forward_flight_flapping_grp['alpha_s']),
               A_1=np.full(nn, FLIGHT_forward_flight_flapping_grp['A_1']), B_1=np.full(nn, FLIGHT_forward_flight_flapping_grp['B_1']))

    wrt = ['e_over_R', 'R', 'a', 'gamma', 'mu', 'sigma', 'theta_0', 'theta_1',
           'alpha_s', 'A_1', 'B_1', 'Omega']
    if method == 'numerical':
        wrt.append('I_b_ref')   # R is used by BladeInertiaComp here

    data = p.check_totals(of=['a_0', 'a_1s', 'b_1s', 'CT_sigma'], wrt=wrt,
                          method='cs', compact_print=True, out_stream=None)

    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-8, atol=1e-10)
