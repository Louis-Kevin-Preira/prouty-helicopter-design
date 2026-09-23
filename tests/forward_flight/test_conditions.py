"""Tests for G0, ForwardFlightConditionsGroup.

Book anchors: Table 3.2 (p. 193), Table 3.3 (p. 196), Table 3.4 (p. 207),
Table 3.5 (p. 232-237), Appendix A (p. 669).
"""

import numpy as np
import openmdao.api as om
import pytest

from prouty.forward_flight import ForwardFlightConditionsGroup

# Example helicopter, Appendix A p. 669
REF = dict(c=2.0, R=30.0, I_b=2870.0, V_tip=650.0, V_son=1116.0,
           rho=0.002377, a=6.0, f=19.3)

# Table 3.2 p. 193, the three level flight iterations at mu = .3.
# (L_F, D_F, H_M, H_T, T, alpha_TPP, lambda')
TABLE_32 = [(-200., 872., 181., 12., 20230., -.0526, -.0276),
            (-730., 904., 364., 30., 20770., -.0629, -.0311),
            (-746., 904., 399., 36., 20790., -.0647, -.0316)]


def errors(entry):
    """Non-empty entries of an _ErrorData: check_totals fills forward or
    reverse depending on the derivative mode the problem was set up in."""
    data = entry['abs error']
    return [e for e in (data.forward, data.reverse) if e is not None]


def build(nn=1, **options):
    p = om.Problem()
    p.model.add_subsystem(
        'g0', ForwardFlightConditionsGroup(num_nodes=nn, **options),
        promotes=['*'])
    p.setup(force_alloc_complex=True)

    names = {meta['prom_name'] for _, meta in
             p.model.list_inputs(out_stream=None, prom_name=True, val=False)}
    for name, value in REF.items():
        if name in names:
            p.set_val(name, value)
    return p


def test_geometry_matches_appendix_a():
    """A_b = b c R exactly; sigma is the rounded quantity in Appendix A."""
    p = build()
    p.run_model()
    assert p.get_val('A_b')[0] == pytest.approx(240.0)
    assert p.get_val('A')[0] == pytest.approx(np.pi * 900.0)
    assert p.get_val('sigma')[0] == pytest.approx(0.085, abs=1.5e-4)


def test_lock_number_appendix_a():
    """gamma = 8.1 with a = 6.0 /rad (p. 31, 213; cross-check p. 477)."""
    p = build()
    p.run_model()
    assert p.get_val('gamma')[0] == pytest.approx(8.1, rel=7e-3)


def test_lift_curve_slope_unit_conversion():
    """`a` in 1/deg must be converted, not passed through (Chapter 6 trap)."""
    p = build()
    p.set_val('a', 6.0 * np.pi / 180.0, units='1/deg')      # 6.0 /rad in 1/deg
    p.run_model()
    assert p.get_val('gamma')[0] == pytest.approx(8.05, rel=1e-3)


def test_advance_ratio_and_tip_mach():
    """mu at 115 kt and M_1,90, Table 3.5 case 1 step g (p. 232)."""
    p = build()
    p.set_val('V', 115.0, units='kn')
    p.run_model()
    assert p.get_val('mu')[0] == pytest.approx(0.3, abs=2e-3)
    assert p.get_val('M_190')[0] == pytest.approx(0.756, abs=2e-3)
    assert p.get_val('M_075')[0] == pytest.approx(0.75 * 650.0 / 1116.0)


@pytest.mark.parametrize('L_F, D_F, H_M, H_T, T, alpha_TPP, lambda_p', TABLE_32)
def test_table_32_iterations(L_F, D_F, H_M, H_T, T, alpha_TPP, lambda_p):
    """Force balance then lambda', against the three iterations of Table 3.2."""
    p = build(tpp_form='forces')
    p.set_val('V', 115.0, units='kn')
    p.set_val('GW', 20000.0)
    for name, value in dict(L_F=L_F, D_F=D_F, H_M=H_M, H_T=H_T, T=T).items():
        p.set_val(name, value)
    p.run_model()

    assert p.get_val('alpha_TPP')[0] == pytest.approx(alpha_TPP, abs=5e-4)
    assert p.get_val('lambda_p')[0] == pytest.approx(lambda_p, abs=5e-4)


def test_thrust_coef_table_34():
    """C_T/sigma at both tip speed ratios of Table 3.4 (p. 207)."""
    p = build(nn=2)
    p.set_val('V', [0.30 * 650.0, 0.45 * 650.0])
    p.set_val('T', [20780.0, 23537.0])
    p.run_model()
    assert p.get_val('CT_sigma') == pytest.approx([0.0862, 0.0977], abs=5e-5)


def test_parasite_form_matches_closed_expression():
    """The G0 chain reproduces the expanded lambda' of p. 167 exactly."""
    mu, T = 0.3, 20734.0
    p = build()
    p.set_val('V', mu * REF['V_tip'])
    p.set_val('T', T)
    p.run_model()

    CT_sigma = p.get_val('CT_sigma')[0]
    sigma = p.get_val('sigma')[0]
    closed = -((REF['f'] / 240.0) * mu ** 3 / 2 / CT_sigma
               + CT_sigma * sigma / (2 * mu))
    assert p.get_val('lambda_p')[0] == pytest.approx(closed, rel=1e-12)


def test_induced_velocity_forms_agree_at_high_speed():
    """p. 123 vs p. 167: within 0.5 % above mu = 0.2, 5.8 % apart at mu = 0.10."""
    out = {}
    for form in ('high_speed', 'exact'):
        p = build(nn=2, vi_form=form)
        p.set_val('V', [0.10 * REF['V_tip'], 0.30 * REF['V_tip']])
        p.set_val('T', 20790.0)
        p.run_model()
        out[form] = p.get_val('vi_OR')

    gap = (out['high_speed'] - out['exact']) / out['exact']
    assert gap[0] == pytest.approx(0.058, abs=5e-3)
    assert abs(gap[1]) < 5e-3


def test_partials():
    p = build(nn=3, tpp_form='forces')
    p.set_val('V', [0.10, 0.30, 0.45] * np.array(REF['V_tip']))
    p.set_val('T', [19000.0, 20790.0, 23537.0])
    p.set_val('L_F', [-200.0, -746.0, -1228.0])
    p.set_val('D_F', [100.0, 904.0, 2000.0])
    p.set_val('H_M', [50.0, 399.0, 660.0])
    p.set_val('H_T', [5.0, 36.0, 77.0])
    p.run_model()

    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    checked = 0
    for comp, derivatives in data.items():
        for key, value in derivatives.items():
            for error in errors(value):
                assert error < 1e-8, f'{comp} {key}'
                checked += 1
    assert checked > 0


def test_totals():
    """Composition of the chain: units, ordering, no duplicated source."""
    p = build(nn=1)
    p.set_val('V', 0.30 * REF['V_tip'])
    p.set_val('T', 20790.0)
    p.run_model()

    data = p.check_totals(
        of=['lambda_p', 'CT_sigma', 'gamma', 'M_190'],
        wrt=['V', 'T', 'c', 'R', 'I_b', 'V_tip'],
        method='cs', compact_print=True, out_stream=None)
    checked = 0
    for key, value in data.items():
        for error in errors(value):
            assert error < 1e-8, key
            checked += 1
    assert checked > 0


# --------------------------------------------------------------------- G1
from prouty.forward_flight import ClosedFormRotorGroup   # noqa: E402

Q_REF = 241028.0          # rho_0 A_b (Omega R)^2, p. 235
P_REF = 284851.0          # rho_0 A_b (Omega R)^3 / 550, p. 236
SIGMA_REF = 0.084883
CD_REF = 0.0100           # backed out of Table 3.3, see validation notes

# Table 3.3 p. 196: (T_M, lambda', a0, theta_0, B_1, A_1, hp_M, H_M)
TABLE_33 = [(20790.0, -0.0316, 4.3, 15.8, 4.9, -2.3, 1097.0, 401.0),
            (21290.0, -0.0607, 4.4, 18.6, 6.0, -2.4, 1760.0, 660.0)]


def build_g1(mode='collective', nn=1, **overrides):
    p = om.Problem()
    p.model.add_subsystem('g1', ClosedFormRotorGroup(num_nodes=nn, mode=mode),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    values = dict(mu=0.3, theta_1=np.deg2rad(-10.0), a=6.0, gamma=8.05033,
                  R=30.0, V_tip=650.0, M_tip=650.0 / 1116.0,
                  M_190=1.3 * 650.0 / 1116.0, cd_bar=CD_REF)
    values.update(overrides)
    for name, value in values.items():
        p.set_val(name, value)
    return p


@pytest.mark.parametrize('T, lambda_p, a0, theta_0, B1, A1, hp, H_M', TABLE_33)
def test_g1_table_33(T, lambda_p, a0, theta_0, B1, A1, hp, H_M):
    """Level flight and climb, all six rotor quantities at once."""
    CT_sigma = T / Q_REF
    p = build_g1(CT_sigma=CT_sigma, lambda_p=lambda_p,
                 vi_OR=SIGMA_REF * CT_sigma / 0.6)
    p.run_model()

    assert p.get_val('a0', units='deg')[0] == pytest.approx(a0, abs=0.1)
    assert p.get_val('theta_0', units='deg')[0] == pytest.approx(theta_0, abs=0.15)
    assert p.get_val('B1_a1s', units='deg')[0] == pytest.approx(B1, abs=0.1)
    assert p.get_val('A1_b1s', units='deg')[0] == pytest.approx(A1, abs=0.1)
    assert p.get_val('CQ_sigma_total')[0] * P_REF == pytest.approx(hp, rel=0.02)
    assert p.get_val('CH_sigma')[0] * Q_REF == pytest.approx(H_M, rel=0.02)


def test_g1_modes_are_inverses():
    """'collective' and 'thrust' must invert each other exactly."""
    CT_sigma, lambda_p = 20790.0 / Q_REF, -0.0316
    vi = SIGMA_REF * CT_sigma / 0.6

    forward = build_g1(CT_sigma=CT_sigma, lambda_p=lambda_p, vi_OR=vi)
    forward.run_model()
    theta_0 = forward.get_val('theta_0')[0]

    back = build_g1('thrust', theta_0=theta_0, lambda_p=lambda_p, vi_OR=vi)
    back.run_model()
    assert back.get_val('CT_sigma')[0] == pytest.approx(CT_sigma, rel=1e-12)


def test_g1_no_compressibility_below_threshold():
    """At OmegaR = 650 the example rotor sits just below M_dr3 (p. 184)."""
    CT_sigma = 20790.0 / Q_REF
    p = build_g1(CT_sigma=CT_sigma, lambda_p=-0.0316,
                 vi_OR=SIGMA_REF * CT_sigma / 0.6)
    p.run_model()
    assert p.get_val('M_ratio')[0] < 1.0
    assert p.get_val('dCQ_sigma_comp')[0] == 0.0
    assert p.get_val('CQ_sigma_total')[0] == pytest.approx(
        p.get_val('CQ_sigma')[0])


def test_g1_compressibility_table_35_case_2():
    """OmegaR = 750 ft/s, Table 3.5 case 2 steps g-k (p. 233)."""
    p = build_g1(CT_sigma=0.086, lambda_p=-0.023,
                 vi_OR=SIGMA_REF * 0.086 / 0.6,
                 M_tip=750.0 / 1116.0, M_190=1.3 * 750.0 / 1116.0,
                 V_tip=750.0)
    p.run_model()
    assert p.get_val('M_dr3')[0] == pytest.approx(0.764, abs=1e-3)
    assert p.get_val('M_ratio')[0] == pytest.approx(1.143, abs=2e-3)
    assert p.get_val('dCQ_sigma_comp')[0] == pytest.approx(0.0013, abs=1e-4)


def test_g1_partials_and_totals():
    CT_sigma = np.array([0.06, 0.086, 0.098])
    p = build_g1(nn=3, CT_sigma=CT_sigma, mu=[0.10, 0.30, 0.45],
                 lambda_p=[-0.034, -0.0316, -0.020],
                 vi_OR=SIGMA_REF * CT_sigma / (2 * np.array([.1, .3, .45])),
                 gamma=[8.05, 8.05, 8.05], cd_bar=[0.0098, 0.0100, 0.0120],
                 M_tip=[0.58, 0.582, 0.582], M_190=[0.64, 0.757, 0.844],
                 B=[0.99, 0.98, 0.97], x_0=0.15)
    p.run_model()

    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    checked = 0
    for comp, derivatives in data.items():
        for key, value in derivatives.items():
            for error in errors(value):
                assert error < 1e-8, f'{comp} {key}'
                checked += 1
    assert checked > 0

    totals = p.check_totals(
        of=['CQ_sigma_total', 'CH_sigma', 'theta_0', 'a0', 'B1_a1s', 'A1_b1s'],
        wrt=['CT_sigma', 'lambda_p', 'mu', 'cd_bar', 'B'],
        method='cs', compact_print=True, out_stream=None)
    checked = 0
    for key, value in totals.items():
        for error in errors(value):
            assert error < 1e-8, key
            checked += 1
    assert checked > 0
