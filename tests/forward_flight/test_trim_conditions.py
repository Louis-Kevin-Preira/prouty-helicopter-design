"""Tests for G1b, TrimConditionsGroup, and the tail rotor it drives.

Book anchors: Table 3.1 (p. 191), Table 3.2 (p. 193), Table 3.3 (p. 196),
Figure A.2 (p. 679).

Three disagreements with the book are known and deliberately not chased; they
are asserted as documented facts rather than tolerated silently:

  * climb thrust runs about 1 % high, because T carries the G.W. sin(gamma)
    term that p. 194 adds to alpha_TPP but not to T (see TppAngleComp);
  * the autorotation column of Table 3.3 is not a fixed point of its own
    equations -- its H_M of 690 lb and its power of -40 hp cannot both hold;
  * theta_0 for delta_3 != 0 in Table 3.1 is a reported value, not one that
    reproduces the flapping printed beside it.
"""

import numpy as np
import openmdao.api as om
import pytest

from prouty.forward_flight.tail_rotor_group import TailRotorGroup
from prouty.forward_flight.trim_conditions_group import TrimConditionsGroup

# Example helicopter, Appendix A p. 669-670
REF = dict(mu=0.3, V_tip=650.0, rho=0.002377, A_b=240.0, sigma=0.084883,
           theta_1=np.deg2rad(-10.0), a=6.0, gamma=8.05033, R=30.0,
           GW=20000.0, i_s=0.0, a1s=0.0, l_T_R=1.23, cd_bar=0.0100,
           delta_3=np.deg2rad(-30.0))
# compressibility is off by default in G1b, so M_tip and M_190 are absent;
# see the group docstring for why Table 3.4 requires it that way

IN_DEG = ('alpha_F', 'alpha_TPP', 'a0', 'theta_0', 'A1_b1s', 'B1_a1s')

# Table 3.3 p. 196, the quantities that are reproducible
TABLE_33 = {
    'level': dict(alpha_F=-6.1, T=20790.0, alpha_TPP=-3.7, a0=4.3,
                  theta_0=15.8, A1_b1s=-2.3, B1_a1s=4.9, H_M=401.0,
                  hp_M=1097.0, T_T=755.0),
    'climb': dict(alpha_F=-11.7, alpha_TPP=-9.2, a0=4.4, theta_0=18.6,
                  A1_b1s=-2.4, B1_a1s=6.0, H_M=660.0, hp_M=1760.0,
                  T_T=1211.0),
}


def build(mode='level', maxiter=120, **overrides):
    p = om.Problem()
    group = p.model.add_subsystem('g', TrimConditionsGroup(mode=mode),
                                  promotes=['*'])
    p.setup()
    group.nonlinear_solver.options['maxiter'] = maxiter
    p.final_setup()
    for name, value in {**REF, **overrides}.items():
        p.set_val(name, value)
    return p


def residual(problem, mode='level'):
    names = ['res_alpha_F', 'res_CT_sigma']
    if mode == 'autorotation':
        names.append('res_CQ_sigma')
    return max(abs(problem.get_val(n)[0]) for n in names)


def errors(entry):
    data = entry['abs error']
    return [e for e in (data.forward, data.reverse) if e is not None]


# ------------------------------------------------------------------ solving
def test_level_flight_converges_to_table_33():
    p = build('level')
    p.run_model()

    assert residual(p) < 1e-12
    for name, book in TABLE_33['level'].items():
        value = p.get_val(name, units='deg' if name in IN_DEG else None)[0]
        assert value == pytest.approx(book, rel=0.03), name


def test_climb_converges_to_table_33():
    p = build('climb', R_C=1000.0)
    p.run_model()

    assert residual(p) < 1e-12
    for name, book in TABLE_33['climb'].items():
        value = p.get_val(name, units='deg' if name in IN_DEG else None)[0]
        assert value == pytest.approx(book, rel=0.03), name

    # documented: T carries the climb term, Table 3.3's 21,290 does not
    assert p.get_val('T')[0] == pytest.approx(21290.0, rel=0.015)
    assert p.get_val('T')[0] > 21290.0


@pytest.mark.parametrize('alpha_F0, CT_sigma0',
                         [(-0.10, 0.085), (0.26, 0.050), (-0.26, 0.120),
                          (0.00, 0.100), (0.10, 0.060)])
def test_level_flight_is_robust_to_the_starting_point(alpha_F0, CT_sigma0):
    """A loop that converges from one guess proves nothing."""
    p = build('level')
    p.set_val('alpha_F', alpha_F0)
    p.set_val('CT_sigma', CT_sigma0)
    p.run_model()

    assert residual(p) < 1e-12
    assert p.get_val('alpha_F', units='deg')[0] == pytest.approx(-5.97, abs=0.02)
    assert p.get_val('T')[0] == pytest.approx(20773.8, rel=1e-4)


def test_gauss_seidel_reproduces_prouty_iteration():
    """Prouty's hand iteration (p. 193) is Gauss-Seidel on this cycle.

    Reproducing the PATH, not just the fixed point, is what catches a wrong
    execution order -- Newton hides it, because the residuals vanish either
    way. This test found exactly that: lambda' was being consumed one pass
    before it was computed.
    """
    p = om.Problem()
    group = p.model.add_subsystem('g', TrimConditionsGroup(), promotes=['*'])
    p.setup()
    group.nonlinear_solver = om.NonlinearRunOnce()
    p.final_setup()
    for name, value in REF.items():
        p.set_val(name, value)

    alpha_F, CT_sigma = 0.0, 20000.0 / 241028.0     # H = H_0, alpha_F = 0
    history = []
    for _ in range(3):
        p.set_val('g.balance.alpha_F', alpha_F)
        p.set_val('g.balance.CT_sigma', CT_sigma)
        p.run_model()
        history.append((np.degrees(alpha_F), p.get_val('L_F')[0],
                        p.get_val('T')[0], p.get_val('alpha_TPP')[0]))
        alpha_F = p.get_val('alpha_F_computed')[0]
        CT_sigma = p.get_val('CT_sigma_computed')[0]

    # Table 3.2 p. 193, first three iterations
    assert history[0][0] == pytest.approx(0.0)
    assert history[0][1] == pytest.approx(-200.0, abs=15.0)
    assert history[0][2] == pytest.approx(20230.0, rel=0.01)

    # alpha_F must move on the second pass, not the third
    assert history[1][0] < -4.0
    assert history[2][0] == pytest.approx(-6.1, abs=0.4)

    # monotone approach to the fixed point
    thrust = [h[2] for h in history]
    assert thrust[0] < thrust[1] < thrust[2]


def test_book_values_leave_a_small_residual():
    """Inject Table 3.3's own numbers and see whether they solve the system.

    This is the most direct test of 'does the book satisfy these equations',
    and it is cheap. Level flight leaves a small residual; the autorotation
    column does not, which is the discrepancy documented in the notes.
    """
    p = om.Problem()
    group = p.model.add_subsystem('g', TrimConditionsGroup(), promotes=['*'])
    p.setup()
    group.nonlinear_solver = om.NonlinearRunOnce()
    p.final_setup()
    for name, value in REF.items():
        p.set_val(name, value)

    p.set_val('g.balance.alpha_F', np.deg2rad(-6.1))
    p.set_val('g.balance.CT_sigma', 20790.0 / 241028.0)
    p.run_model()

    assert abs(p.get_val('res_alpha_F')[0]) < np.deg2rad(0.5)
    assert abs(p.get_val('res_CT_sigma')[0]) < 0.002


# ----------------------------------------------------------- autorotation
def test_autorotation_descent_rate():
    p = build('autorotation', hp_T0=25.0, hp_trans=10.0, hp_acc=5.0)
    p.set_val('gamma_fp', np.deg2rad(-9.0))
    p.set_val('alpha_F', np.deg2rad(2.0))
    p.run_model()

    assert residual(p, 'autorotation') < 1e-12
    # the engineering answer of the whole section, p. 196
    assert p.get_val('R_C', units='ft/min')[0] == pytest.approx(-1803.0, rel=0.02)
    assert p.get_val('hp_M')[0] == pytest.approx(-40.0, abs=0.1)
    assert p.get_val('T')[0] == pytest.approx(20060.0, rel=0.01)


def test_autorotation_column_of_table_33_is_inconsistent():
    """Documented, not tolerated: H_M = 690 lb cannot coexist with -40 hp.

    Forcing the book's H_M reproduces its attitude exactly but gives +127 hp;
    solving honestly gives -40 hp and H_M near 200 lb. The two cannot both be
    satisfied, so this asserts the discrepancy rather than a tolerance on it.
    """
    p = build('autorotation', hp_T0=25.0, hp_trans=10.0, hp_acc=5.0)
    p.set_val('gamma_fp', np.deg2rad(-9.0))
    p.set_val('alpha_F', np.deg2rad(2.0))
    p.run_model()

    assert p.get_val('H_M')[0] == pytest.approx(200.0, rel=0.1)
    assert p.get_val('H_M')[0] < 0.5 * 690.0


def test_gamma_is_bounded_against_the_periodic_root():
    """sin(gamma) has infinitely many roots; without a bound Newton finds one."""
    p = build('autorotation', hp_T0=25.0, hp_trans=10.0, hp_acc=5.0)
    p.set_val('gamma_fp', np.deg2rad(-9.0))
    p.run_model()
    assert abs(p.get_val('gamma_fp')[0]) < 1.4


# -------------------------------------------------------------- tail rotor
def build_tail(T_T=755.0, delta_3=0.0):
    p = om.Problem()
    p.model.add_subsystem('t', TailRotorGroup(), promotes=['*'])
    p.setup()
    for name, value in dict(T_T=T_T, mu=0.3, V_tip=650.0, rho=0.002377,
                            a=6.0, delta_3=np.deg2rad(delta_3)).items():
        p.set_val(name, value)
    return p


# Table 3.1 p. 191: delta_3 -> (a_1s, b_1s, hp_T, H_T), theta_0 excluded
TABLE_31 = [(0.0, 1.77, 0.89, 26.5, 36.1),
            (30.0, 1.03, 1.49, 29.9, 26.9),
            (-30.0, 1.72, -0.10, 26.7, 35.5)]


@pytest.mark.parametrize('delta_3, a1s, b1s, hp_T, H_T', TABLE_31)
def test_tail_rotor_against_table_31(delta_3, a1s, b1s, hp_T, H_T):
    p = build_tail(delta_3=delta_3)
    p.run_model()

    assert p.get_val('a1s_T', units='deg')[0] == pytest.approx(a1s, abs=0.03)
    assert p.get_val('b1s_T', units='deg')[0] == pytest.approx(b1s, abs=0.03)
    assert p.get_val('hp_T')[0] == pytest.approx(hp_T, rel=0.02)
    assert p.get_val('H_T')[0] == pytest.approx(H_T, rel=0.02)


def test_tail_rotor_thrust_reverses_in_autorotation():
    forward = build_tail(T_T=755.0)
    forward.run_model()
    reverse = build_tail(T_T=-27.0)
    reverse.run_model()

    assert forward.get_val('theta_0_T')[0] > 0.0
    assert reverse.get_val('theta_0_T')[0] < forward.get_val('theta_0_T')[0]
    # profile power survives the thrust reversal
    assert reverse.get_val('hp_T')[0] > 20.0


# --------------------------------------------------------------- gradients
def test_totals_against_a_retrimmed_central_difference():
    """check_totals with method='fd' does NOT reconverge the Newton solver
    while perturbing, so its 'reference' is meaningless here. The finite
    difference has to be built by hand, re-solving at each perturbed point.
    """
    of, wrt, step = 'hp_M', 'GW', 20.0

    p = build('level')
    p.run_model()
    totals = p.compute_totals(of=[of], wrt=[wrt])
    analytic = totals[(of, wrt)][0, 0]

    values = []
    for sign in (-1.0, 1.0):
        q = build('level', GW=REF['GW'] + sign * step)
        q.run_model()
        values.append(q.get_val(of)[0])
    numeric = (values[1] - values[0]) / (2.0 * step)

    assert analytic == pytest.approx(numeric, rel=1e-5)
    assert analytic > 0.0          # heavier helicopter, more power


# ------------------------------------------------- Table 3.5 cases 7 and 8
from prouty.forward_flight.fixed_collective_trim_group import (  # noqa: E402
    FixedCollectiveTrimGroup)

DIVE = dict(mu=0.3, V_tip=650.0, rho=0.002377, A_b=240.0, sigma=0.084883,
            theta_0=np.deg2rad(13.0), theta_1=np.deg2rad(-10.0), a=6.0,
            gamma=8.05033, R=30.0, GW=20000.0, i_s=0.0, a1s=0.0, l_T_R=1.23,
            cd_bar=0.0100, delta_3=np.deg2rad(-30.0), B=0.97, x_0=0.15)


def build_fixed(mode='dive', cross_check=False, **overrides):
    p = om.Problem()
    p.model.add_subsystem('g', FixedCollectiveTrimGroup(
        mode=mode, cross_check=cross_check), promotes=['*'])
    p.setup()
    for name, value in {**DIVE, **overrides}.items():
        p.set_val(name, value)
    return p


def test_dive_reproduces_the_robust_quantities_of_case_7():
    """p. 242-244. C_T/sigma and C_Q/sigma are reproduced; lambda' is not, and
    that is documented rather than fitted — see validation_forward_flight section 4.
    The rotor comes out very nearly autorotating, which is the point of the
    case."""
    p = build_fixed('dive')
    p.run_model()

    assert abs(p.get_val('res_lift')[0]) < 1e-10
    assert p.get_val('CT_sigma')[0] == pytest.approx(0.083, rel=0.01)
    assert p.get_val('CQ_sigma')[0] == pytest.approx(0.0008, abs=1e-4)
    # 228 hp against 20,000 lb: the rotor is barely being driven
    assert p.get_val('hp_M')[0] < 400.0


def test_dive_and_auxiliary_give_the_same_propulsive_force():
    """p. 246: 'note similarity to results of previous case'. Gravity in one
    and a propeller in the other, same number."""
    dive = build_fixed('dive')
    dive.run_model()
    auxiliary = build_fixed('auxiliary')
    auxiliary.run_model()

    assert auxiliary.get_val('T_aux')[0] == pytest.approx(
        dive.get_val('F_prop')[0], rel=1e-8)
    assert dive.get_val('F_prop')[0] == pytest.approx(
        20000.0 * np.sin(dive.get_val('gamma_D')[0]), rel=1e-8)


def test_the_two_descent_routes_agree():
    """PropulsiveBalanceComp works in forces through f_M, DescentAngleComp in
    rotor coefficients (p. 239). Different paths, same balance."""
    p = build_fixed('dive', cross_check=True)
    p.run_model()

    force_route = p.get_val('gamma_D', units='deg')[0]
    coefficient_route = p.get_val('gamma_D_coef', units='deg')[0]
    assert coefficient_route == pytest.approx(force_route, rel=0.15)
    assert p.get_val('R_D_coef')[0] == pytest.approx(p.get_val('R_D')[0],
                                                     rel=0.15)


def test_level_flight_is_the_zero_of_the_propulsive_balance():
    """p. 244 note: for level flight f_M = -(f + H_T/q), so F = 0. Checked
    on the component rather than through a trim, since the dive case is not
    in level flight."""
    from prouty.forward_flight.propulsive_balance_comp import (
        PropulsiveBalanceComp)

    p = om.Problem()
    p.model.add_subsystem('b', PropulsiveBalanceComp(mode='auxiliary'),
                          promotes=['*'])
    p.setup()
    q, f, H_T = 45.2, 19.5, 59.0
    p.set_val('q', q)
    p.set_val('f', f)
    p.set_val('f_M', -(f + H_T / q))
    p.set_val('H_T', H_T)
    p.run_model()

    assert abs(p.get_val('T_aux')[0]) < 1e-9


# ------------------------------------------------------- Table 3.5 case 6
from prouty.forward_flight.climb_drag_area_comp import ClimbDragAreaComp  # noqa

def test_climb_drag_area_matches_case_6():
    """p. 242 step dd: f_climb = f + H_T/q + (G.W./q) tan(gamma_c)."""
    p = om.Problem()
    p.model.add_subsystem('c', ClimbDragAreaComp(), promotes=['*'])
    p.setup()
    p.set_val('f', 20.6)
    p.set_val('H_T', 139.0)
    p.set_val('q', 45.2)
    p.set_val('gamma_fp', np.deg2rad(4.9))
    p.set_val('GW', 20000.0)
    p.run_model()

    assert p.get_val('f_climb')[0] == pytest.approx(61.6, abs=0.1)


def test_climb_drag_area_reduces_to_the_level_flight_value():
    p = om.Problem()
    p.model.add_subsystem('c', ClimbDragAreaComp(), promotes=['*'])
    p.setup()
    p.set_val('f', 20.6)
    p.set_val('H_T', 139.0)
    p.set_val('q', 45.2)
    p.set_val('gamma_fp', 0.0)
    p.run_model()

    assert p.get_val('f_climb')[0] == pytest.approx(20.6 + 139.0 / 45.2)


def build_g1b(mode, **overrides):
    p = om.Problem()
    p.model.add_subsystem('g', TrimConditionsGroup(mode=mode), promotes=['*'])
    p.setup()
    for name, value in {**REF, **overrides}.items():
        p.set_val(name, value)
    return p


def test_climb_trim_reproduces_the_closed_form_not_the_charts():
    """Table 3.3 and Table 3.5 case 6 compute the same 1,000 ft/min climb and
    disagree by 20 % on power. We land on the closed-form side; see
    validation_forward_flight section 4 for why, and do not chase case 6."""
    p = build_g1b('climb', R_C=1000.0)
    p.run_model()

    assert p.get_val('hp_M')[0] == pytest.approx(1760.0, rel=0.02)
    assert p.get_val('lambda_p')[0] == pytest.approx(-0.0607, rel=0.02)
    # case 6 would want 2,109 hp; make the gap explicit
    assert p.get_val('hp_M')[0] < 0.9 * 2109.0


def test_climb_at_zero_rate_is_level_flight():
    """Chapter 4 diagnostic D: with subsystem solves off the climb trim walked onto the
    alpha_F bound and settled on a false root -- five times the weight in thrust and
    negative rotor power -- while both trim residuals read zero. Climb at zero rate must
    reproduce level flight from a cold start."""
    level = build('level')
    climb = build('climb', R_C=0.0)
    for p in (level, climb):
        p.run_model()

    for name in ('alpha_F', 'CT_sigma', 'hp_M', 'T'):
        assert climb.get_val(name)[0] == pytest.approx(level.get_val(name)[0], rel=1e-9)
    assert climb.get_val('hp_M')[0] > 0.0
    assert climb.get_val('T')[0] / 20000.0 < 1.1


def test_steep_climb_trims_past_the_old_fuselage_bound():
    """3,000 ft/min at 80 kt needs the fuselage more than 25.8 deg nose down, which the old
    +-0.45 rad bound cut off; climb and autorotation now get a wider bracket."""
    p = build('climb', R_C=3000.0, mu=0.208)
    p.run_model()
    assert np.degrees(p.get_val('alpha_F')[0]) < -25.8
    assert 2000.0 < p.get_val('hp_M')[0] < 4000.0
