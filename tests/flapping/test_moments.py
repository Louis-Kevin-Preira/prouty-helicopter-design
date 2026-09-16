"""Chapter 7 -- moments produced by flapping, p. 476-477 and 479."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.assert_utils import assert_near_equal

from prouty.flapping import FlappingMomentsGroup
from prouty.flapping.cg_moment_comp import CGMomentComp
from prouty.flapping.hub_moment_comp import HubMomentComp
from prouty.flapping.rotor_stiffness_comp import RotorStiffnessComp


# ------------------------------------------------------------------------
# rotor_stiffness_comp
# ------------------------------------------------------------------------

R_EX_rotor_stiffness, E_OVER_R_EX_rotor_stiffness, I_B_EX_rotor_stiffness, B_EX_rotor_stiffness = 30.0, 0.05, 2870.0, 4.0


OMEGA_EX_rotor_stiffness = 650.0 / 30.0


MG_EX_rotor_stiffness = 1.5 * I_B_EX_rotor_stiffness / (R_EX_rotor_stiffness * (1 - E_OVER_R_EX_rotor_stiffness))


C_EX, RHO_SL_rotor_stiffness, A_EX_rotor_stiffness, GAMMA_EX = 2.0, 0.002378, 6.0, 8.1


I_B_FROM_GAMMA = C_EX * RHO_SL_rotor_stiffness * A_EX_rotor_stiffness * R_EX_rotor_stiffness ** 4 / GAMMA_EX


def _run_rotor_stiffness(nn=1, mode='stiffness', convention='hinge', **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', RotorStiffnessComp(num_nodes=nn, mode=mode,
                                   convention=convention),
        promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('b', B_EX_rotor_stiffness)
    p.set_val('Omega', np.full(nn, OMEGA_EX_rotor_stiffness))
    p.set_val('I_b', I_B_EX_rotor_stiffness)
    if mode == 'stiffness':
        p.set_val('e', E_OVER_R_EX_rotor_stiffness * R_EX_rotor_stiffness)
        p.set_val('e_over_R', E_OVER_R_EX_rotor_stiffness)
        p.set_val('M_b_over_g', MG_EX_rotor_stiffness)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_hinge_convention_follows_the_derivation():
    """dM_M/da_1s = (1/2) e b Omega^2 (M_b/g), p. 477."""
    p = _run_rotor_stiffness()
    expected = 0.5 * (E_OVER_R_EX_rotor_stiffness * R_EX_rotor_stiffness) * B_EX_rotor_stiffness * OMEGA_EX_rotor_stiffness ** 2 * MG_EX_rotor_stiffness
    assert_near_equal(p.get_val('dMM_da1s')[0], expected, 1e-13)


def test_center_convention_reproduces_the_printed_number():
    """p. 477 gets 200,940 ft-lb/rad, using gamma = 8.1 rather than I_b."""
    p = _run_rotor_stiffness(convention='center', I_b=I_B_FROM_GAMMA)
    assert_near_equal(p.get_val('dMM_da1s')[0], 200940.0, 1e-4)


def test_center_convention_equals_the_printed_A_b_form():
    """(3/4)(e/R) b Omega^2 I_b is the printed A_b expression rearranged.

    Since c rho a R^4 = gamma I_b, the two are algebraically identical.
    """
    A_b = B_EX_rotor_stiffness * C_EX * R_EX_rotor_stiffness
    printed = (0.75 * E_OVER_R_EX_rotor_stiffness * A_b * RHO_SL_rotor_stiffness * R_EX_rotor_stiffness
               * (OMEGA_EX_rotor_stiffness * R_EX_rotor_stiffness) ** 2 * A_EX_rotor_stiffness / GAMMA_EX)

    p = _run_rotor_stiffness(convention='center', I_b=I_B_FROM_GAMMA)
    assert_near_equal(p.get_val('dMM_da1s')[0], printed, 1e-12)


def test_center_convention_equals_the_printed_linear_mass_form():
    """(1/4)(e/R) b m R (Omega R)^2 with m = 3 I_b / R^3."""
    m = 3 * I_B_FROM_GAMMA / R_EX_rotor_stiffness ** 3
    printed = 0.25 * E_OVER_R_EX_rotor_stiffness * B_EX_rotor_stiffness * m * R_EX_rotor_stiffness * (OMEGA_EX_rotor_stiffness * R_EX_rotor_stiffness) ** 2

    p = _run_rotor_stiffness(convention='center', I_b=I_B_FROM_GAMMA)
    assert_near_equal(p.get_val('dMM_da1s')[0], printed, 1e-12)


def test_the_gap_between_conventions_is_the_static_moment():
    """Entry C7-6. The printed shortcuts silently use the first static moment
    about the centre of rotation, m R^2/2, where p. 456 defines it about the
    hinge, (m R^2/2)(1 - e/R)^2. The ratio is exactly the two moments.
    """
    hinge = _run_rotor_stiffness().get_val('dMM_da1s')[0]
    center = _run_rotor_stiffness(convention='center').get_val('dMM_da1s')[0]

    mg_center = 1.5 * I_B_EX_rotor_stiffness / R_EX_rotor_stiffness
    assert_near_equal(hinge / center, MG_EX_rotor_stiffness / mg_center, 1e-13)
    assert_near_equal(hinge / center, 1.0 / (1 - E_OVER_R_EX_rotor_stiffness), 1e-13)
    assert hinge > center


def test_stiffness_scales_as_expected():
    doubled = _run_rotor_stiffness(Omega=2 * OMEGA_EX_rotor_stiffness).get_val('dMM_da1s')[0]
    base = _run_rotor_stiffness().get_val('dMM_da1s')[0]
    assert_near_equal(doubled / base, 4.0, 1e-13)

    assert_near_equal(_run_rotor_stiffness(b=8.0).get_val('dMM_da1s')[0] / base, 2.0, 1e-13)


def test_zero_offset_gives_no_stiffness():
    p = _run_rotor_stiffness(e=0.0, e_over_R=0.0)
    assert_near_equal(p.get_val('dMM_da1s')[0], 0.0, 1e-14)



@pytest.mark.parametrize('convention', ['hinge', 'center'])
def test_round_trip(convention):
    fwd = _run_rotor_stiffness(convention=convention)
    inv = _run_rotor_stiffness(mode='e_over_R', convention=convention,
               dMM_da1s=fwd.get_val('dMM_da1s'))
    assert_near_equal(inv.get_val('e_over_R_eff')[0], E_OVER_R_EX_rotor_stiffness, 1e-12)


def test_center_inversion_matches_the_printed_form():
    """p. 477: (e/R)_eff = 1 / [(3/4) b Omega^2 I_b / (dM_M/da_1s)]."""
    K = 200940.0
    p = _run_rotor_stiffness(mode='e_over_R', convention='center', dMM_da1s=K,
             I_b=I_B_FROM_GAMMA)
    printed = 1.0 / (0.75 * B_EX_rotor_stiffness * OMEGA_EX_rotor_stiffness ** 2 * I_B_FROM_GAMMA / K)
    assert_near_equal(p.get_val('e_over_R_eff')[0], printed, 1e-12)


def test_hingeless_rotor_anchor():
    """p. 477: the AH-56A hingeless rotor had about 13 % effective offset.

    Order of magnitude only; the AH-56A blade data is not in the book.
    """
    K = 0.13 * 0.75 * B_EX_rotor_stiffness * OMEGA_EX_rotor_stiffness ** 2 * I_B_EX_rotor_stiffness
    p = _run_rotor_stiffness(mode='e_over_R', convention='center', dMM_da1s=K)
    assert_near_equal(p.get_val('e_over_R_eff')[0], 0.13, 1e-12)


def test_the_two_inversions_differ_by_one_minus_x():
    hinge = _run_rotor_stiffness(mode='e_over_R', dMM_da1s=200000.0).get_val('e_over_R_eff')[0]
    center = _run_rotor_stiffness(mode='e_over_R', convention='center',
                  dMM_da1s=200000.0).get_val('e_over_R_eff')[0]
    assert_near_equal(hinge, center / (1 + center), 1e-13)
    assert hinge < center


def test_vectorized_rotor_stiffness():
    nn = 3
    Om = np.array([19.0, 20.0, 21.67])
    p = _run_rotor_stiffness(nn, Omega=Om)
    expected = 0.5 * (E_OVER_R_EX_rotor_stiffness * R_EX_rotor_stiffness) * B_EX_rotor_stiffness * Om ** 2 * MG_EX_rotor_stiffness
    assert_near_equal(p.get_val('dMM_da1s'), expected, 1e-13)



@pytest.mark.parametrize('nn,mode,convention',
                         [(1, 'stiffness', 'hinge'),
                          (4, 'stiffness', 'hinge'),
                          (4, 'stiffness', 'center'),
                          (4, 'e_over_R', 'hinge'),
                          (4, 'e_over_R', 'center')])
def test_partials_rotor_stiffness(nn, mode, convention):
    extra = ({'dMM_da1s': np.linspace(1.8e5, 2.2e5, nn)}
             if mode == 'e_over_R' else {})
    p = _run_rotor_stiffness(nn, mode, convention, Omega=np.linspace(19.0, 21.7, nn), **extra)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-10)


def test_chained_from_blade_inertia():
    """BladeInertiaComp -> RotorStiffnessComp, as wired in G4."""
    from prouty.flapping.blade_inertia_comp import BladeInertiaComp

    p = om.Problem()
    m = p.model
    m.add_subsystem('inertia', BladeInertiaComp(input_mode='I_b'),
                    promotes=['*'])
    m.add_subsystem('stiffness', RotorStiffnessComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('I_b_ref', I_B_EX_rotor_stiffness)
    p.set_val('R', R_EX_rotor_stiffness)
    p.set_val('e_over_R', E_OVER_R_EX_rotor_stiffness)
    p.set_val('b', B_EX_rotor_stiffness)
    p.set_val('Omega', OMEGA_EX_rotor_stiffness)
    p.run_model()

    assert_near_equal(p.get_val('dMM_da1s')[0], 212732.0, 1e-4)

    data = p.check_totals(of=['dMM_da1s'],
                          wrt=['I_b_ref', 'R', 'e_over_R', 'b', 'Omega'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-6)


# ------------------------------------------------------------------------
# hub_moment_comp
# ------------------------------------------------------------------------

R_EX_hub_moment, E_OVER_R_EX_hub_moment, I_B_EX_hub_moment, B_EX_hub_moment = 30.0, 0.05, 2870.0, 4.0


OMEGA_EX_hub_moment = 650.0 / 30.0


MG_EX_hub_moment = 1.5 * I_B_EX_hub_moment / (R_EX_hub_moment * (1 - E_OVER_R_EX_hub_moment))


K_EX = 0.5 * (E_OVER_R_EX_hub_moment * R_EX_hub_moment) * B_EX_hub_moment * OMEGA_EX_hub_moment ** 2 * MG_EX_hub_moment


def _run_hub_moment(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', HubMomentComp(num_nodes=nn), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('dMM_da1s', np.full(nn, K_EX))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_pitching_moment_matches_the_printed_form():
    """M_M = (1/2) e b Omega^2 a_1s (M_b/g), p. 477."""
    a1s = np.radians(2.87)
    p = _run_hub_moment(a_1s=a1s)
    expected = 0.5 * (E_OVER_R_EX_hub_moment * R_EX_hub_moment) * B_EX_hub_moment * OMEGA_EX_hub_moment ** 2 * MG_EX_hub_moment * a1s
    assert_near_equal(p.get_val('M_M')[0], expected, 1e-13)


def test_roll_moment_is_the_symmetric_analogue():
    a1s, b1s = np.radians(2.87), np.radians(2.87)
    p = _run_hub_moment(a_1s=a1s, b_1s=b1s)
    assert_near_equal(p.get_val('L_M')[0], p.get_val('M_M')[0], 1e-14)


def test_signs_follow_the_tip_path_plane():
    """Disc back gives nose up, disc down to the right gives right roll."""
    p = _run_hub_moment(a_1s=np.radians(3.0), b_1s=np.radians(-1.0))
    assert p.get_val('M_M')[0] > 0.0
    assert p.get_val('L_M')[0] < 0.0


def test_no_flapping_no_moment():
    p = _run_hub_moment(a_1s=0.0, b_1s=0.0)
    assert_near_equal(p.get_val('M_M')[0], 0.0, 1e-14)
    assert_near_equal(p.get_val('L_M')[0], 0.0, 1e-14)


def test_channels_are_independent():
    """a_1s must not produce roll, nor b_1s pitch, at the hub."""
    only_a = _run_hub_moment(a_1s=np.radians(3.0))
    assert_near_equal(only_a.get_val('L_M')[0], 0.0, 1e-14)

    only_b = _run_hub_moment(b_1s=np.radians(3.0))
    assert_near_equal(only_b.get_val('M_M')[0], 0.0, 1e-14)


def test_azimuthal_average_holds_for_three_blades_or_more():
    """The b/2 factor of p. 477 is the sum of cos^2 over equally spaced
    blades. It is constant only for b >= 3.
    """
    psi = np.linspace(0.0, 2 * np.pi, 1441)
    for b in (3, 4, 5, 6):
        total = sum(np.cos(psi + 2 * np.pi * k / b) ** 2 for k in range(b))
        assert_near_equal(total.max(), b / 2, 1e-12)
        assert total.max() - total.min() < 1e-12


def test_two_bladed_rotor_has_a_two_per_rev_hub_moment():
    """Documented limitation: b = 2 gives 1 + cos(2 psi), not a constant."""
    psi = np.linspace(0.0, 2 * np.pi, 1441)
    total = sum(np.cos(psi + np.pi * k) ** 2 for k in range(2))

    assert_near_equal(total.min(), 0.0, 1e-12)
    assert_near_equal(total.max(), 2.0, 1e-12)
    np.testing.assert_allclose(total, 1 + np.cos(2 * psi), atol=1e-12)


def test_moment_is_linear_in_flapping():
    one = _run_hub_moment(a_1s=np.radians(2.0)).get_val('M_M')[0]
    two = _run_hub_moment(a_1s=np.radians(4.0)).get_val('M_M')[0]
    assert_near_equal(two / one, 2.0, 1e-13)


def test_example_helicopter_magnitude():
    """One degree of longitudinal flapping on the example helicopter."""
    p = _run_hub_moment(a_1s=np.radians(1.0))
    assert_near_equal(p.get_val('M_M')[0], K_EX * np.radians(1.0), 1e-13)
    assert 3000.0 < p.get_val('M_M')[0] < 4000.0


def test_vectorized_hub_moment():
    nn = 3
    a1s = np.radians(np.array([1.0, 2.87, 4.15]))
    p = _run_hub_moment(nn, a_1s=a1s, b_1s=np.zeros(nn))
    assert_near_equal(p.get_val('M_M'), K_EX * a1s, 1e-13)



@pytest.mark.parametrize('nn', [1, 4])
def test_partials_hub_moment(nn):
    p = _run_hub_moment(nn, dMM_da1s=np.linspace(1.8e5, 2.2e5, nn),
             a_1s=np.radians(np.linspace(1.0, 4.0, nn)),
             b_1s=np.radians(np.linspace(-1.5, 0.5, nn)))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-10)


def test_chained_from_blade_inertia_and_stiffness():
    """BladeInertiaComp -> RotorStiffnessComp -> HubMomentComp."""
    from prouty.flapping.blade_inertia_comp import BladeInertiaComp
    from prouty.flapping.rotor_stiffness_comp import RotorStiffnessComp

    p = om.Problem()
    m = p.model
    m.add_subsystem('inertia', BladeInertiaComp(input_mode='I_b'),
                    promotes=['*'])
    m.add_subsystem('stiffness', RotorStiffnessComp(), promotes=['*'])
    m.add_subsystem('hub', HubMomentComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('I_b_ref', I_B_EX_hub_moment)
    p.set_val('R', R_EX_hub_moment)
    p.set_val('e_over_R', E_OVER_R_EX_hub_moment)
    p.set_val('b', B_EX_hub_moment)
    p.set_val('Omega', OMEGA_EX_hub_moment)
    p.set_val('a_1s', np.radians(2.87))
    p.set_val('b_1s', np.radians(-0.78))
    p.run_model()

    assert_near_equal(p.get_val('M_M')[0], K_EX * np.radians(2.87), 1e-10)
    assert p.get_val('L_M')[0] < 0.0

    data = p.check_totals(of=['M_M', 'L_M'],
                          wrt=['I_b_ref', 'R', 'e_over_R', 'b', 'Omega',
                               'a_1s', 'b_1s'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-6)


# ------------------------------------------------------------------------
# cg_moment_comp
# ------------------------------------------------------------------------

GW_cg_moment, R_EX_cg_moment, OMEGA_R, SIGMA, A_EX_cg_moment, H_M_cg_moment = 20000.0, 30.0, 650.0, 0.085, 6.0, 7.5


RHO_SL_cg_moment = 0.002378


DISK_Q = RHO_SL_cg_moment * np.pi * R_EX_cg_moment ** 2 * OMEGA_R ** 2


CT_SIGMA = GW_cg_moment / DISK_Q / SIGMA


def _run_cg_moment(nn=1, inplane_force='simple', **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', CGMomentComp(num_nodes=nn, inplane_force=inplane_force),
        promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('T', np.full(nn, GW_cg_moment))
    p.set_val('h_M', H_M_cg_moment)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_equation_as_printed():
    """M_CG = M_M + (T a_1s + H_0) h_M - T l_M, p. 476."""
    M_M, a1s, H_0, l_M = 10655.0, np.radians(2.87), -400.0, 0.4
    p = _run_cg_moment(M_M=M_M, a_1s=a1s, H_0=H_0, l_M=l_M)

    expected = M_M + (GW_cg_moment * a1s + H_0) * H_M_cg_moment - GW_cg_moment * l_M
    assert_near_equal(p.get_val('M_CG')[0], expected, 1e-13)


def test_three_contributions_add_up():
    M_M, a1s, H_0, l_M = 10655.0, np.radians(2.87), -400.0, 0.4

    hub = _run_cg_moment(M_M=M_M).get_val('M_CG')[0]
    force = _run_cg_moment(a_1s=a1s, H_0=H_0).get_val('M_CG')[0]
    offset = _run_cg_moment(l_M=l_M).get_val('M_CG')[0]
    total = _run_cg_moment(M_M=M_M, a_1s=a1s, H_0=H_0, l_M=l_M).get_val('M_CG')[0]

    assert_near_equal(hub + force + offset, total, 1e-12)
    assert_near_equal(hub, M_M, 1e-14)
    assert_near_equal(offset, -GW_cg_moment * l_M, 1e-14)


def test_cg_ahead_of_the_shaft_gives_nose_down():
    assert _run_cg_moment(l_M=0.5).get_val('M_CG')[0] < 0.0
    assert _run_cg_moment(l_M=-0.5).get_val('M_CG')[0] > 0.0


def test_disc_back_gives_nose_up():
    assert _run_cg_moment(a_1s=np.radians(3.0)).get_val('M_CG')[0] > 0.0


def test_roll_is_the_symmetric_analogue():
    val = np.radians(2.0)
    p = _run_cg_moment(M_M=5000.0, L_M=5000.0, a_1s=val, b_1s=val,
             H_0=-300.0, Y_0=-300.0, l_M=0.4, y_M=0.4)
    assert_near_equal(p.get_val('L_CG')[0], p.get_val('M_CG')[0], 1e-13)


def test_pitch_and_roll_channels_are_independent():
    p = _run_cg_moment(M_M=5000.0, a_1s=np.radians(3.0), H_0=-400.0)
    assert_near_equal(p.get_val('L_CG')[0], 0.0, 1e-14)


def test_external_mode_reproduces_simple_mode():
    a1s, H_0 = np.radians(2.87), -400.0
    ref = _run_cg_moment(M_M=10655.0, a_1s=a1s, H_0=H_0, l_M=0.4)
    ext = _run_cg_moment(inplane_force='external', M_M=10655.0,
               H=GW_cg_moment * a1s + H_0, l_M=0.4)
    assert_near_equal(ext.get_val('M_CG')[0], ref.get_val('M_CG')[0], 1e-13)


def test_inflow_halves_the_rotor_force_stiffness_in_hover():
    """Entry to the p. 479 table: the perpendicular assumption of p. 476
    overstates dH/da_1s by more than a factor of two in hover.

    p. 479 gives d(C_H/sigma)/da_1s = C_T/sigma + (a/8) lambda', which in
    dimensional form is T + (a/8) lambda' sigma q.
    """
    lam = -np.sqrt(GW_cg_moment / DISK_Q / 2.0)          # hover, lambda' = -v/(Omega R)
    corrected = (CT_SIGMA + A_EX_cg_moment / 8 * lam) * SIGMA * DISK_Q

    assert_near_equal(corrected, 9255.0, 1e-3)
    assert_near_equal(GW_cg_moment / corrected, 2.16, 1e-2)

    # p. 479 table, hover: rotor force contributes 73,500 ft-lb per radian
    assert_near_equal(corrected * H_M_cg_moment, 73500.0, 6e-2)
    assert GW_cg_moment * H_M_cg_moment > 2 * corrected * H_M_cg_moment


def test_flapping_derivative_of_the_cg_moment():
    """d(M_CG)/da_1s = dM_M/da_1s + (T + dH/da_1s) h_M.

    p. 479 tabulates the two contributions separately for the example
    helicopter in hover: 200,940 and 73,500, totalling 274,440.
    """
    K = 200940.0
    p = _run_cg_moment(inplane_force='external', M_M=K * np.radians(1.0),
             H=73500.0 / H_M_cg_moment * np.radians(1.0))

    per_radian = p.get_val('M_CG')[0] / np.radians(1.0)
    assert_near_equal(per_radian, 274440.0, 1e-6)


def test_vectorized_cg_moment():
    nn = 3
    a1s = np.radians(np.array([1.44, 2.87, 4.15]))
    p = _run_cg_moment(nn, M_M=np.zeros(nn), a_1s=a1s, H_0=np.zeros(nn))
    assert_near_equal(p.get_val('M_CG'), GW_cg_moment * a1s * H_M_cg_moment, 1e-13)



@pytest.mark.parametrize('nn,mode', [(1, 'simple'), (4, 'simple'),
                                     (4, 'external')])
def test_partials_cg_moment(nn, mode):
    common = dict(M_M=np.linspace(4000.0, 15000.0, nn),
                  L_M=np.linspace(-3000.0, -1000.0, nn),
                  T=np.linspace(18000.0, 30000.0, nn),
                  h_M=H_M_cg_moment, l_M=0.4, y_M=-0.15)
    extra = ({'a_1s': np.radians(np.linspace(1.0, 4.0, nn)),
              'b_1s': np.radians(np.linspace(-1.5, 0.5, nn)),
              'H_0': np.linspace(-600.0, -200.0, nn),
              'Y_0': np.linspace(-100.0, 100.0, nn)} if mode == 'simple'
             else {'H': np.linspace(200.0, 1500.0, nn),
                   'Y': np.linspace(-400.0, -100.0, nn)})
    p = _run_cg_moment(nn, mode, **common, **extra)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-10)


def test_chained_from_the_hub_moment():
    """RotorStiffness -> HubMoment -> CGMoment, the G4 chain."""
    from prouty.flapping.blade_inertia_comp import BladeInertiaComp
    from prouty.flapping.hub_moment_comp import HubMomentComp
    from prouty.flapping.rotor_stiffness_comp import RotorStiffnessComp

    p = om.Problem()
    m = p.model
    m.add_subsystem('inertia', BladeInertiaComp(input_mode='I_b'),
                    promotes=['*'])
    m.add_subsystem('stiffness', RotorStiffnessComp(), promotes=['*'])
    m.add_subsystem('hub', HubMomentComp(), promotes=['*'])
    m.add_subsystem('cg', CGMomentComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('I_b_ref', 2870.0)
    p.set_val('R', R_EX_cg_moment)
    p.set_val('e_over_R', 0.05)
    p.set_val('b', 4.0)
    p.set_val('Omega', OMEGA_R / R_EX_cg_moment)
    p.set_val('a_1s', np.radians(2.87))
    p.set_val('b_1s', np.radians(-0.78))
    p.set_val('T', GW_cg_moment)
    p.set_val('h_M', H_M_cg_moment)
    p.set_val('l_M', 0.4)
    p.run_model()

    # The three terms of p. 476, checked individually. A 0.4 ft forward c.g.
    # offset alone outweighs the whole rotor force contribution here.
    a1s = np.radians(2.87)
    hub = p.get_val('M_M')[0]
    force = GW_cg_moment * a1s * H_M_cg_moment
    offset = -GW_cg_moment * 0.4
    assert_near_equal(p.get_val('M_CG')[0], hub + force + offset, 1e-10)
    assert abs(offset) > force

    data = p.check_totals(of=['M_CG', 'L_CG'],
                          wrt=['I_b_ref', 'e_over_R', 'b', 'Omega', 'a_1s',
                               'b_1s', 'T', 'h_M', 'l_M'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-6)


# ------------------------------------------------------------------------
# flapping_moments_group
# ------------------------------------------------------------------------

R_EX_flapping_moments_grp, E_OVER_R_EX_flapping_moments_grp, I_B_EX_flapping_moments_grp, B_EX_flapping_moments_grp = 30.0, 0.05, 2870.0, 4.0


OMEGA_EX_flapping_moments_grp, GW_flapping_moments_grp, H_M_flapping_moments_grp = 650.0 / 30.0, 20000.0, 7.5


MG_EX_flapping_moments_grp = 1.5 * I_B_EX_flapping_moments_grp / (R_EX_flapping_moments_grp * (1 - E_OVER_R_EX_flapping_moments_grp))


K_HINGE = 0.5 * (E_OVER_R_EX_flapping_moments_grp * R_EX_flapping_moments_grp) * B_EX_flapping_moments_grp * OMEGA_EX_flapping_moments_grp ** 2 * MG_EX_flapping_moments_grp


A1S, B1S = np.radians(2.87), np.radians(-0.78)


def _build(nn=1, **opts):
    over = {k: opts.pop(k) for k in list(opts) if k in
            ('a_1s', 'b_1s', 'T', 'l_M', 'y_M', 'H_0', 'Y_0', 'H', 'Y')}

    p = om.Problem()
    p.model.add_subsystem('g4', FlappingMomentsGroup(num_nodes=nn, **opts),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    if opts.get('blade_input', 'I_b') != 'external':
        p.set_val('R', R_EX_flapping_moments_grp)
        p.set_val('e_over_R', E_OVER_R_EX_flapping_moments_grp)
        if opts.get('blade_input', 'I_b') == 'I_b':
            p.set_val('I_b_ref', I_B_EX_flapping_moments_grp)
    p.set_val('b', B_EX_flapping_moments_grp)
    p.set_val('Omega', np.full(nn, OMEGA_EX_flapping_moments_grp))
    p.set_val('a_1s', np.full(nn, A1S))
    p.set_val('b_1s', np.full(nn, B1S))
    p.set_val('T', np.full(nn, GW_flapping_moments_grp))
    p.set_val('h_M', H_M_flapping_moments_grp)
    for k, v in over.items():
        p.set_val(k, v)

    p.run_model()
    return p


def test_default_chain():
    p = _build()

    assert_near_equal(p.get_val('dMM_da1s')[0], K_HINGE, 1e-12)
    assert_near_equal(p.get_val('M_M')[0], K_HINGE * A1S, 1e-12)
    assert_near_equal(p.get_val('L_M')[0], K_HINGE * B1S, 1e-12)
    assert_near_equal(p.get_val('M_CG')[0],
                      K_HINGE * A1S + GW_flapping_moments_grp * A1S * H_M_flapping_moments_grp, 1e-12)


def test_center_convention_reproduces_the_printed_stiffness():
    """p. 477: 200,940 ft-lb/rad, with the I_b that gamma = 8.1 implies."""
    I_b_from_gamma = 2.0 * 0.002378 * 6.0 * R_EX_flapping_moments_grp ** 4 / 8.1

    p = om.Problem()
    p.model.add_subsystem('g4', FlappingMomentsGroup(convention='center'),
                          promotes=['*'])
    p.setup()
    p.set_val('R', R_EX_flapping_moments_grp)
    p.set_val('e_over_R', E_OVER_R_EX_flapping_moments_grp)
    p.set_val('I_b_ref', I_b_from_gamma)
    p.set_val('b', B_EX_flapping_moments_grp)
    p.set_val('Omega', OMEGA_EX_flapping_moments_grp)
    p.run_model()

    assert_near_equal(p.get_val('dMM_da1s')[0], 200940.0, 1e-4)


def test_the_two_conventions_differ_by_one_over_one_minus_x():
    hinge = _build().get_val('dMM_da1s')[0]
    center = _build(convention='center').get_val('dMM_da1s')[0]
    assert_near_equal(hinge / center, 1.0 / (1 - E_OVER_R_EX_flapping_moments_grp), 1e-13)


def test_blade_input_modes_agree():
    m_equiv = 3 * I_B_EX_flapping_moments_grp / (R_EX_flapping_moments_grp ** 3 * (1 - E_OVER_R_EX_flapping_moments_grp) ** 3)
    ref = _build()

    alt = om.Problem()
    alt.model.add_subsystem('g4', FlappingMomentsGroup(blade_input='m'),
                            promotes=['*'])
    alt.setup()
    alt.set_val('R', R_EX_flapping_moments_grp)
    alt.set_val('e_over_R', E_OVER_R_EX_flapping_moments_grp)
    alt.set_val('m', m_equiv)
    alt.set_val('b', B_EX_flapping_moments_grp)
    alt.set_val('Omega', OMEGA_EX_flapping_moments_grp)
    alt.set_val('a_1s', A1S)
    alt.set_val('T', GW_flapping_moments_grp)
    alt.set_val('h_M', H_M_flapping_moments_grp)
    alt.run_model()

    assert_near_equal(alt.get_val('M_CG')[0], ref.get_val('M_CG')[0], 1e-10)


def test_external_inplane_force_halves_the_rotor_contribution():
    """The p. 479 correction, applied through the external route."""
    simple = _build(l_M=0.0)

    corrected_H = 9255.0 * A1S / np.radians(1.0) * np.radians(1.0)
    ext = _build(inplane_force='external', H=9255.0 * A1S, l_M=0.0)

    hub = simple.get_val('M_M')[0]
    assert_near_equal(simple.get_val('M_CG')[0] - hub, GW_flapping_moments_grp * A1S * H_M_flapping_moments_grp, 1e-12)
    assert_near_equal(ext.get_val('M_CG')[0] - hub, 9255.0 * A1S * H_M_flapping_moments_grp, 1e-12)
    assert (simple.get_val('M_CG')[0] - hub) > 2 * (ext.get_val('M_CG')[0] - hub)


def test_cg_offset_can_dominate():
    """A 0.4 ft forward c.g. offset outweighs the whole rotor force term."""
    p = _build(l_M=0.4)
    assert_near_equal(p.get_val('M_CG')[0],
                      K_HINGE * A1S + GW_flapping_moments_grp * A1S * H_M_flapping_moments_grp - GW_flapping_moments_grp * 0.4, 1e-12)
    assert GW_flapping_moments_grp * 0.4 > GW_flapping_moments_grp * A1S * H_M_flapping_moments_grp


def test_zero_flapping_leaves_only_the_offset_moment():
    p = _build(a_1s=0.0, b_1s=0.0, l_M=0.4)
    assert_near_equal(p.get_val('M_M')[0], 0.0, 1e-14)
    assert_near_equal(p.get_val('M_CG')[0], -GW_flapping_moments_grp * 0.4, 1e-12)


def test_group_is_feed_forward():
    p = _build(nn=2, a_1s=np.full(2, A1S), b_1s=np.full(2, B1S))
    before = p.get_val('M_CG').copy()
    p.run_model()
    assert_near_equal(p.get_val('M_CG'), before, 1e-14)



@pytest.mark.parametrize('convention,inplane',
                         [('hinge', 'simple'), ('center', 'simple'),
                          ('hinge', 'external')])
def test_totals(convention, inplane):
    nn = 3
    extra = ({'H': np.linspace(300.0, 1200.0, nn),
              'Y': np.linspace(-400.0, -100.0, nn)}
             if inplane == 'external' else {})
    p = _build(nn, convention=convention, inplane_force=inplane, l_M=0.4,
               a_1s=np.radians(np.linspace(1.4, 4.2, nn)),
               b_1s=np.radians(np.linspace(-0.8, -0.5, nn)),
               T=np.linspace(18000.0, 26000.0, nn), **extra)

    wrt = ['R', 'e_over_R', 'I_b_ref', 'b', 'Omega', 'a_1s', 'b_1s', 'T',
           'h_M', 'l_M']
    wrt += ['H', 'Y'] if inplane == 'external' else ['H_0', 'Y_0']

    data = p.check_totals(of=['dMM_da1s', 'M_M', 'L_M', 'M_CG', 'L_CG'],
                          wrt=wrt, method='cs', compact_print=True,
                          out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-8, atol=1e-5)


def test_chains_with_the_forward_flight_group():
    """G2 into G4: promoted names and defaults must line up."""
    from prouty.flapping import ForwardFlightFlappingGroup

    nn = 2
    p = om.Problem()
    m = p.model
    m.add_subsystem('steady', ForwardFlightFlappingGroup(num_nodes=nn),
                    promotes=['*'])
    m.add_subsystem('moments',
                    FlappingMomentsGroup(num_nodes=nn, blade_input='external'),
                    promotes=['*'])
    p.setup()

    p.set_val('e_over_R', E_OVER_R_EX_flapping_moments_grp)
    p.set_val('R', R_EX_flapping_moments_grp)
    p.set_val('I_b_ref', I_B_EX_flapping_moments_grp)
    p.set_val('sigma', 0.08488)
    p.set_val('b', B_EX_flapping_moments_grp)
    p.set_val('a', np.full(nn, 6.0))
    p.set_val('gamma', np.full(nn, 8.1))
    p.set_val('Omega', np.full(nn, OMEGA_EX_flapping_moments_grp))
    p.set_val('mu', np.array([0.2, 0.35]))
    p.set_val('theta_0', np.full(nn, np.radians(14.0)))
    p.set_val('theta_1', np.full(nn, np.radians(-10.0)))
    p.set_val('alpha_s', np.full(nn, np.radians(-5.0)))
    p.set_val('A_1', np.full(nn, np.radians(-2.2)))
    p.set_val('B_1', np.full(nn, np.radians(1.5)))
    p.set_val('T', np.full(nn, GW_flapping_moments_grp))
    p.set_val('h_M', H_M_flapping_moments_grp)
    p.run_model()

    # blade_input='external' lets G2's BladeInertiaComp feed G4
    assert_near_equal(p.get_val('M_M'),
                      p.get_val('dMM_da1s') * p.get_val('a_1s'), 1e-13)
    assert np.all(np.diff(p.get_val('M_CG')) > 0)
