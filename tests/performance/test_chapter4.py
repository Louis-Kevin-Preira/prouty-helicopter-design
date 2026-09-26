"""Chapter 4 -- Performance Analysis, p. 273-338.

Anchors are the example helicopter: engine ratings Figs 4.1-4.3 pp. 274-276,
hover performance Figs 4.33-4.35 pp. 314-316. The piston lapse law is external
to the book (C4-5) and is checked against its own closed form.
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import (assert_check_partials, assert_check_totals,
                                         assert_near_equal)

from prouty.hover import AtmosphereComp
from prouty.hover.atmosphere_comp import EXP, LAPSE
from prouty.performance.wake_dynamic_pressure_comp import _Q_MINUS4, _Q_MINUS10, _R, wake_q_ratio
from prouty.performance import (AccessoryLossComp, AtmosphereGroup, DayTemperatureComp,
                                EngineGroup,
                                EnginePowerRequiredComp, ExhaustDragComp, GearboxLossComp,
                                PowerLossesGroup,
                                FinInterferenceRatioComp, FuselageDragComp, HubPylonInterferenceComp,
                                LandingGearDragComp, StabilizerDragComp,
                                GroundProximityDownloadComp, HoverCeilingBalance,
                                ClimbFlatPlateComp, ClimbInducedVelocityComp,
                                ClimbCeilingBalance, ClimbTimeDistanceComp,
                                ForwardClimbGroup,
                                EquivalentRotorLDComp,
                                ForwardFlightPowerGroup,
                                BestEnduranceSpeedBalance, BestRangeSpeedBalance,
                                CruisePerformanceGroup,
                                CruiseSpeedBalance, CruiseSpeedComp,
                                FerryReservesComp, ForwardClimbBalance, FuelFlowSlopeComp,
                                HoverPerformanceGroup,
                                MaxSpeedBalance, MilitaryMissionGroup, MissionIntegralComp,
                                PayloadRangeComp,
                                SpecificEnduranceComp, SpecificRangeComp, TangencyProductComp,
                                VerticalClimbBalance, VerticalClimbGroup,
                                VerticalClimbPowerComp,
                                NacelleDragComp, RotorHubDragComp,
                                RotorFuselageInterferenceComp, RotorShaftDragComp,
                                InducedVelocityRatioComp,
                                PseudoGroundEffectComp, TailRotorFinPowerComp,
                                TailRotorFinInterferenceGroup, TailRotorGrossThrustComp,
                                TailRotorThrustComp,
                                ParasiteDragGroup, RotorLoadingMatchComp,
                                TotalParasiteDragComp,
                                VerticalDragGroup,
                                VerticalDragComp,
                                WakeDynamicPressureComp,
                                InstalledPowerComp, PistonFuelFlowComp, PistonPowerLapseComp,
                                TurboshaftFuelFlowComp, TurboshaftRatingsComp)

# Figure 4.33 p. 314: density ratio -> (standard day, 95 deg F day) altitude, ft
FIG_4_33 = [(0.533, 20000.0, 14700.0),
            (0.739, 10000.0, 6300.0),
            (0.862, 5000.0, 2200.0)]


def _day(day, nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('day', DayTemperatureComp(num_nodes=nn, day=day),
                          promotes=['*'])
    p.model.add_subsystem('atm', AtmosphereComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in ivc.items():
        p.set_val(name, val)
    p.run_model()
    return p


# ------------------------------------------------------------------------
# DayTemperatureComp
# ------------------------------------------------------------------------

def test_day_temperature_offsets():
    assert_near_equal(_day('standard', altitude=8000.0).get_val('dT'), 0.0)
    assert_near_equal(_day('offset', altitude=8000.0, dT_offset=36.0).get_val('dT'),
                      36.0, 1e-12)
    # isothermal 95 F: the air temperature is 95 F whatever the altitude
    for h in (0.0, 10000.0, 25000.0):
        p = _day('isothermal', altitude=h)
        assert_near_equal(p.get_val('T_air', units='degF'), 95.0, 1e-9)


@pytest.mark.parametrize('rho_ratio, h_std, h_hot', FIG_4_33)
def test_fig_4_33_altitudes_need_an_isothermal_hot_day(rho_ratio, h_std, h_hot):
    """C4-4: the hot day of Chapter 4 is isothermal, not a +36 F offset."""
    std = _day('standard', altitude=h_std).get_val('density_ratio')
    hot = _day('isothermal', altitude=h_hot).get_val('density_ratio')
    shifted = _day('offset', altitude=h_hot, dT_offset=36.0).get_val('density_ratio')

    assert_near_equal(std, rho_ratio, 0.005)
    assert_near_equal(hot, rho_ratio, 0.005)
    assert abs(shifted - rho_ratio) / rho_ratio > 0.015


@pytest.mark.parametrize('day', ['standard', 'offset', 'isothermal'])
def test_day_temperature_partials(day):
    p = om.Problem()
    p.model.add_subsystem('comp', DayTemperatureComp(num_nodes=3, day=day),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('altitude', [0.0, 7000.0, 21000.0])
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# PistonPowerLapseComp
# ------------------------------------------------------------------------

P_SL_PISTON = np.array([180.0, 170.0, 145.0])       # generic, no engine in particular


def _piston(altitudes, day='standard', supercharged=False, h_crit=None):
    nn = len(altitudes)
    p = om.Problem()
    p.model.add_subsystem('day', DayTemperatureComp(day=day), promotes=['*'])
    p.model.add_subsystem('atm', AtmosphereComp(), promotes=['*'])
    p.model.add_subsystem('eng', PistonPowerLapseComp(supercharged=supercharged),
                          promotes=['*'])
    p.setup()
    p.set_val('P_SL', P_SL_PISTON)
    if h_crit is not None:
        p.set_val('h_crit', h_crit)
    out = []
    for h in altitudes:
        p.set_val('altitude', h)
        p.run_model()
        out.append(p.get_val('P_eng', units='hp')[0].copy())
    return np.array(out)


def test_piston_sea_level_standard_gives_the_ratings():
    assert_near_equal(_piston([0.0])[0], P_SL_PISTON, 1e-12)


@pytest.mark.parametrize('h', [5000.0, 10000.0, 20000.0])
def test_piston_aspirated_lapse_is_delta_over_sqrt_theta(h):
    theta = 1.0 - LAPSE * h
    expected = P_SL_PISTON * theta ** EXP / np.sqrt(theta)
    assert_near_equal(_piston([h])[0], expected, 1e-10)


def test_piston_hot_day_loses_sqrt_of_temperature_ratio():
    """Same pressure altitude: P_hot / P_std = sqrt(T_std / T_hot)."""
    h = 6000.0
    theta = 1.0 - LAPSE * h
    ratio = _piston([h], day='isothermal')[0] / _piston([h])[0]
    assert_near_equal(ratio, np.full(3, np.sqrt(518.67 * theta / 554.67)), 1e-10)


def test_piston_supercharged_is_flat_then_aspirated():
    h_c = 12000.0
    phi_c = (1.0 - LAPSE * h_c) ** (EXP - 0.5)
    low, high = _piston([4000.0, 20000.0], supercharged=True, h_crit=h_c)
    aspirated = _piston([20000.0])[0]
    assert_near_equal(low, P_SL_PISTON, 1e-12)
    assert_near_equal(high, aspirated / phi_c, 1e-10)


def test_piston_supercharged_blend_is_continuous():
    h_c = 12000.0
    h = np.linspace(9000.0, 15000.0, 601)
    P = _piston(h, supercharged=True, h_crit=h_c)[:, 0]
    assert np.all(np.diff(P) <= 1e-9)                       # never rises with altitude
    assert np.max(np.abs(np.diff(P, 2))) < 1e-2             # no slope jump


@pytest.mark.parametrize('supercharged', [False, True])
def test_piston_partials(supercharged):
    p = om.Problem()
    p.model.add_subsystem('comp', PistonPowerLapseComp(num_nodes=3, supercharged=supercharged),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    # nodes below, inside and above the smoothmin of an 8,000 ft critical altitude
    p.set_val('density_ratio', [0.94, 0.786, 0.63])
    p.set_val('T_air', [510.0, 490.0, 470.0])
    p.set_val('P_SL', P_SL_PISTON)
    if supercharged:
        p.set_val('h_crit', 8000.0)
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# TurboshaftRatingsComp -- Figures 4.1-4.2 pp. 274-275, torque limit p. 320
# ------------------------------------------------------------------------

KT = 1.68781        # ft/s per knot


def _turboshaft(h, V_kt=0.0, day='standard'):
    p = om.Problem()
    p.model.add_subsystem('day', DayTemperatureComp(day=day), promotes=['*'])
    p.model.add_subsystem('atm', AtmosphereComp(), promotes=['*'])
    p.model.add_subsystem('eng', TurboshaftRatingsComp(), promotes=['*'])
    p.setup()
    p.set_val('altitude', h)
    p.set_val('V', V_kt * KT)
    p.run_model()
    return p.get_val('P_eng', units='hp')[0]


def test_turboshaft_torque_limit_below_1200_ft():
    """p. 320: full takeoff power is not available below about 1,200 ft."""
    assert_near_equal(_turboshaft(500.0)[0], 2080.0, 0.002)
    assert _turboshaft(2500.0)[0] < 2040.0


def test_turboshaft_max_continuous_joins_intermediate_above_23500_ft():
    below, above = _turboshaft(10000.0), _turboshaft(27000.0)
    assert below[1] - below[2] > 100.0
    assert_near_equal(above[2], above[1], 0.005)


# Figure 4.2 p. 275 at 200 kt: (day, altitude, rating index, hp)
FIG_4_2_200KT = [('standard', 0.0, 2, 1693.0),
                 ('isothermal', 4000.0, 0, 1725.0),
                 ('isothermal', 4000.0, 1, 1533.0),
                 ('isothermal', 4000.0, 2, 1127.0)]


@pytest.mark.parametrize('day, h, r, P_book', FIG_4_2_200KT)
def test_turboshaft_ram_effect_fig_4_2(day, h, r, P_book):
    """C4-6: Figs 4.1 and 4.2 disagree by about 13 hp at 4,000 ft, 95 F."""
    assert abs(_turboshaft(h, 200.0, day)[r] - P_book) < 20.0


def test_turboshaft_intermediate_reaches_torque_limit_in_forward_flight():
    """Figure 4.2: sea level intermediate meets the torque limit near 165 kt."""
    assert _turboshaft(0.0, 140.0)[1] < 2070.0
    assert_near_equal(_turboshaft(0.0, 200.0)[1], 2080.0, 0.002)


def test_turboshaft_warns_outside_the_two_days():
    p = om.Problem()
    p.model.add_subsystem('eng', TurboshaftRatingsComp(), promotes=['*'])
    p.setup()
    p.set_val('T_air', 580.0)
    with pytest.warns(UserWarning, match='Figure 4.1'):
        p.run_model()


def test_turboshaft_partials():
    p = om.Problem()
    p.model.add_subsystem('comp', TurboshaftRatingsComp(num_nodes=4), promotes=['*'])
    p.setup(force_alloc_complex=True)
    # torque-limit fillet, MCP/INT fillet, hot day at speed, mid temperature
    p.set_val('altitude', [1225.0, 23500.0, 4000.0, 10000.0])
    p.set_val('T_air', [515.0, 438.9, 554.67, 520.0])
    p.set_val('V', [50.0, 250.0, 330.0, 150.0])
    p.set_val('V_son', [1115.0, 1027.0, 1154.0, 1100.0])
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# InstalledPowerComp -- pp. 276-277, anchors pp. 311 and 320
# ------------------------------------------------------------------------

def test_installed_power_example_helicopter_sea_level():
    """p. 320: 3,920 hp intermediate installed; p. 311: 98 % of the ratings."""
    p = om.Problem()
    p.model.add_subsystem('atm', AtmosphereComp(), promotes=['*'])
    p.model.add_subsystem('eng', TurboshaftRatingsComp(), promotes=['*'])
    p.model.add_subsystem('inst', InstalledPowerComp(), promotes=['*'])
    p.setup()
    p.set_val('n_eng', 2.0)
    p.set_val('k_inst', 0.02)
    p.run_model()
    P_takeoff, P_intermediate, _ = p.get_val('P_avail', units='hp')[0]
    assert_near_equal(P_intermediate, 3920.0, 0.002)
    assert_near_equal(P_takeoff, 2 * 0.98 * 2080.0, 0.002)       # torque limited


def _installed(P_eng, n_eng, k_inst=0.0, P_acc=0.0, P_trans=None):
    nn = len(n_eng)
    p = om.Problem()
    p.model.add_subsystem('inst', InstalledPowerComp(num_nodes=nn,
                          transmission_limit=P_trans is not None), promotes=['*'])
    p.setup()
    p.set_val('P_eng', P_eng)
    p.set_val('n_eng', n_eng)
    p.set_val('k_inst', k_inst)
    p.set_val('P_acc', P_acc)
    if P_trans is not None:
        p.set_val('P_trans', P_trans)
    p.run_model()
    return p.get_val('P_avail', units='hp')


def test_installed_power_losses_and_one_engine_inoperative():
    P_eng = np.array([[1000.0, 900.0, 800.0]] * 2)
    P = _installed(P_eng, [2.0, 1.0], k_inst=0.05, P_acc=20.0)
    assert_near_equal(P[0], 2.0 * (0.95 * P_eng[0] - 20.0), 1e-12)
    assert_near_equal(P[1], 0.95 * P_eng[1] - 20.0, 1e-12)


def test_installed_power_transmission_limit():
    P = _installed(np.array([[1000.0, 900.0, 600.0]]), [2.0], P_trans=1500.0)[0]
    assert_near_equal(P[:2], [1500.0, 1500.0], 1e-12)
    assert_near_equal(P[2], 1200.0, 1e-12)


@pytest.mark.parametrize('limit', [False, True])
def test_installed_power_partials(limit):
    p = om.Problem()
    p.model.add_subsystem('comp', InstalledPowerComp(num_nodes=2, transmission_limit=limit),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('P_eng', [[1000.0, 760.0, 700.0], [1100.0, 900.0, 600.0]])
    p.set_val('n_eng', [2.0, 1.0])
    p.set_val('k_inst', 0.03)
    p.set_val('P_acc', 15.0)
    if limit:
        p.set_val('P_trans', 1450.0)          # node 0: fillet, limited, free
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# TurboshaftFuelFlowComp -- Figure 4.3 p. 276
# ------------------------------------------------------------------------

def _fuel_flow(h, P_per_engine, day='standard', n_eng=1.0, k_det=0.05):
    P_per_engine = np.atleast_1d(P_per_engine)
    p = om.Problem()
    p.model.add_subsystem('day', DayTemperatureComp(day=day), promotes=['*'])
    p.model.add_subsystem('atm', AtmosphereComp(), promotes=['*'])
    p.model.add_subsystem('ff', TurboshaftFuelFlowComp(), promotes=['*'])
    p.setup()
    p.set_val('altitude', h)
    p.set_val('n_eng', n_eng)
    p.set_val('k_det', k_det)
    out = []
    for P in P_per_engine:
        p.set_val('P_req', n_eng * P)
        p.run_model()
        out.append(p.get_val('FF', units='lbm/h')[0] / n_eng)
    return np.array(out)


# Figure 4.3: axis intercepts of the standard-day lines, lb/hr
FIG_4_3_INTERCEPTS = [(0.0, 243.0), (5000.0, 198.0), (10000.0, 158.0), (15000.0, 120.0),
                      (20000.0, 87.0), (25000.0, 56.0), (30000.0, 28.0)]


@pytest.mark.parametrize('h, FF0', FIG_4_3_INTERCEPTS)
def test_turboshaft_fuel_flow_intercepts_fig_4_3(h, FF0):
    assert abs(_fuel_flow(h, 0.0)[0] - FF0) < 3.0


def test_turboshaft_fuel_flow_lines_converge_fig_4_3():
    """Figure 4.3: the lines meet near 1,045 lb/hr at 2,100 hp."""
    FF = [_fuel_flow(h, 2100.0)[0] for h, _ in FIG_4_3_INTERCEPTS]
    assert np.all(np.abs(np.array(FF) - 1045.0) < 15.0)


def test_turboshaft_fuel_flow_hot_day_fig_4_3():
    """Dashed line, 4,000 ft 95 F: 218 lb/hr at zero power, above the standard day."""
    hot = _fuel_flow(4000.0, [0.0, 1000.0], day='isothermal')
    std = _fuel_flow(4000.0, [0.0, 1000.0])
    assert abs(hot[0] - 218.0) < 3.0
    assert np.all(hot > std)


def test_turboshaft_fuel_flow_deterioration_and_engines_out():
    """p. 275: the chart includes 5 %; p. 325: one engine pays one intercept."""
    assert_near_equal(_fuel_flow(0.0, 800.0, k_det=0.0) * 1.05,
                      _fuel_flow(0.0, 800.0), 1e-12)
    P_total = 1200.0
    two = _fuel_flow(0.0, P_total / 2, n_eng=2.0)[0] * 2
    one = _fuel_flow(0.0, P_total, n_eng=1.0)[0]
    assert_near_equal(two - one, _fuel_flow(0.0, 0.0)[0], 1e-10)


def test_turboshaft_fuel_flow_partials():
    p = om.Problem()
    p.model.add_subsystem('comp', TurboshaftFuelFlowComp(num_nodes=3), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('altitude', [0.0, 4000.0, 22000.0])
    p.set_val('T_air', [518.67, 554.67, 450.0])
    p.set_val('P_req', [1500.0, 2600.0, 700.0])
    p.set_val('n_eng', [1.0, 2.0, 2.0])
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# PistonFuelFlowComp -- Willans line, p. 275-276 (C4-5)
# ------------------------------------------------------------------------

def test_piston_fuel_flow_willans_line():
    p = om.Problem()
    p.model.add_subsystem('ff', PistonFuelFlowComp(num_nodes=2), promotes=['*'])
    p.setup()
    p.set_val('P_req', [120.0, 120.0])
    p.set_val('n_eng', [1.0, 2.0])
    p.set_val('P_rated', 145.0)
    p.set_val('bsfc', 0.48)
    p.run_model()
    assert_near_equal(p.get_val('FF'), [0.48 * 120.0] * 2, 1e-12)      # a_w = 0: constant BSFC

    p.set_val('a_w', 0.05)
    p.set_val('k_det', 0.03)
    p.run_model()
    expected = 1.03 * (np.array([1.0, 2.0]) * 0.05 * 145.0 + 0.48 * 120.0)
    assert_near_equal(p.get_val('FF'), expected, 1e-12)


def test_piston_fuel_flow_partials():
    p = om.Problem()
    p.model.add_subsystem('comp', PistonFuelFlowComp(num_nodes=2), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('P_req', [100.0, 150.0])
    p.set_val('n_eng', [1.0, 2.0])
    p.set_val('P_rated', 145.0)
    p.set_val('a_w', 0.04)
    p.set_val('k_det', 0.02)
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# EngineGroup -- G0 assembly, pp. 274-277
# ------------------------------------------------------------------------

def test_atmosphere_vectorized_partials():
    p = om.Problem()
    p.model.add_subsystem('atm', AtmosphereComp(num_nodes=3), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('altitude', [0.0, 9000.0, 25000.0])
    p.set_val('dT', [0.0, 20.0, -10.0])
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def _engine_group(engine_type, day='standard', nn=2, **vals):
    p = om.Problem()
    p.model.add_subsystem('engine', EngineGroup(num_nodes=nn, engine_type=engine_type, day=day,
                                                transmission_limit=True), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('P_trans', 1e5)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_engine_group_turboshaft_example_helicopter():
    """p. 320: 3,920 hp intermediate installed at sea level; node 2 is 4,000 ft, 95 F."""
    p = _engine_group('turboshaft', day='isothermal', altitude=[0.0, 4000.0],
                      T_day=[518.67, 554.67], n_eng=2.0, k_inst=0.02, P_req=[2000.0, 2000.0])
    assert_near_equal(p.get_val('P_avail', units='hp')[0, 1], 3920.0, 0.002)
    assert_near_equal(p.get_val('T_air', units='degF')[1], 95.0, 1e-9)
    assert_near_equal(p.get_val('FF')[1] / 2, _fuel_flow(4000.0, 1000.0, day='isothermal')[0], 1e-10)


def test_engine_group_nodes_match_single_node_runs():
    h, P_req = [2000.0, 14000.0], [150.0, 110.0]
    p = _engine_group('piston_turbo', altitude=h, P_SL=P_SL_PISTON, h_crit=8000.0,
                      P_req=P_req, bsfc=0.48)
    for i in range(2):
        q = _engine_group('piston_turbo', nn=1, altitude=h[i], P_SL=P_SL_PISTON, h_crit=8000.0,
                          P_req=P_req[i], bsfc=0.48)
        assert_near_equal(p.get_val('P_avail')[i], q.get_val('P_avail')[0], 1e-12)
        assert_near_equal(p.get_val('FF')[i], q.get_val('FF')[0], 1e-12)


@pytest.mark.parametrize('engine_type, day', [('piston_na', 'standard'),
                                              ('piston_turbo', 'offset'),
                                              ('turboshaft', 'isothermal')])
def test_engine_group_totals(engine_type, day):
    vals = dict(altitude=[3000.0, 12000.0], n_eng=[2.0, 1.0], P_req=[300.0, 900.0])
    wrt = ['altitude', 'n_eng', 'P_req', 'k_inst', 'k_det']
    if engine_type == 'turboshaft':
        vals['V'] = [100.0, 250.0]
        wrt += ['V']
    else:
        vals['P_SL'] = P_SL_PISTON
        wrt += ['P_SL']
        if engine_type == 'piston_turbo':
            vals['h_crit'] = 12000.0
            wrt += ['h_crit']
    if day == 'offset':
        vals['dT_offset'] = [10.0, -5.0]
        wrt += ['dT_offset']
    if day == 'isothermal':
        wrt += ['T_day']
    p = _engine_group(engine_type, day=day, **vals)
    data = p.check_totals(of=['P_avail', 'FF'], wrt=wrt, method='cs', out_stream=None)
    assert_check_totals(data, atol=1e-6, rtol=1e-6)


# ------------------------------------------------------------------------
# G1 -- GearboxLossComp, p. 277
# ------------------------------------------------------------------------

EXAMPLE_DESIGN = {'P_design_nose': 2000.0, 'P_design_main': 4000.0, 'P_design_tail': 750.0}


def _gearbox(powers, design, gearboxes=None):
    nn = len(next(iter(powers.values())))
    opts = {} if gearboxes is None else {'gearboxes': gearboxes}
    p = om.Problem()
    p.model.add_subsystem('gb', GearboxLossComp(num_nodes=nn, **opts), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in {**powers, **design}.items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_gearbox_loss_example_helicopter_p277():
    """p. 277: 0.0025(4,000 + eng) + 0.00875(4,000 + MR) + 0.005(750 + TR),
    i.e. 49 + 0.0112 MR + 0.0075 TR when the engine power is MR + TR."""
    P_MR, P_TR = np.array([1500.0, 3200.0]), np.array([150.0, 260.0])
    p = _gearbox({'P_MR': P_MR, 'P_TR': P_TR, 'P_req': P_MR + P_TR}, EXAMPLE_DESIGN)
    L = p.get_val('P_loss_gearbox', units='hp')
    assert_near_equal(L, 48.75 + 0.01125 * P_MR + 0.0075 * P_TR, 1e-12)
    assert_near_equal(L, 49.0 + 0.0112 * P_MR + 0.0075 * P_TR, 0.003)


def test_gearbox_loss_custom_layout():
    """A single-engine light helicopter: one spur reduction, one bevel main box, one tail box."""
    layout = {'reduction': dict(shaft='engine', n_spur_bevel=1, n_planetary=0),
              'main': dict(shaft='main_rotor', n_spur_bevel=1, n_planetary=0),
              'tail': dict(shaft='tail_rotor', n_spur_bevel=1, n_planetary=0)}
    p = _gearbox({'P_MR': [140.0], 'P_TR': [15.0], 'P_req': [160.0]},
                 {'P_design_reduction': 180.0, 'P_design_main': 145.0, 'P_design_tail': 25.0},
                 gearboxes=layout)
    expected = 0.0025 * ((180.0 + 160.0) + (145.0 + 140.0) + (25.0 + 15.0))
    assert_near_equal(p.get_val('P_loss_gearbox'), expected, 1e-12)


def test_gearbox_loss_rejects_unknown_shaft():
    with pytest.raises(ValueError, match='shaft'):
        GearboxLossComp(gearboxes={'x': dict(shaft='propeller', n_spur_bevel=1, n_planetary=0)})


def test_gearbox_loss_partials():
    p = _gearbox({'P_MR': [1500.0, 3200.0], 'P_TR': [150.0, 260.0], 'P_req': [1700.0, 3500.0]},
                 EXAMPLE_DESIGN)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G1 -- AccessoryLossComp, p. 278
# ------------------------------------------------------------------------

def _accessories(nn=1, **vals):
    p = om.Problem()
    p.model.add_subsystem('acc', AccessoryLossComp(num_nodes=nn), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_accessory_loss_example_helicopter_p278():
    """p. 278: 2,200 W -> 4 hp, 1.3 gpm at 3,000 psi -> 3 hp; 49 + 4 + 3 = 56 hp."""
    gen = _accessories(load_elec=2200.0).get_val('P_loss_acc', units='hp')[0]
    hyd = _accessories(flow_hyd=1.3, p_hyd=3000.0).get_val('P_loss_acc', units='hp')[0]
    assert_near_equal(gen, 2200.0 / (0.75 * 746.0), 1e-12)
    assert round(gen) == 4 and round(hyd) == 3

    gb = _gearbox({'P_MR': [0.0], 'P_TR': [0.0], 'P_req': [0.0]}, EXAMPLE_DESIGN)
    assert round(gb.get_val('P_loss_gearbox')[0] + gen + hyd) == 56


def test_accessory_loss_book_constants_are_unit_conversions():
    from openmdao.utils.units import convert_units
    assert_near_equal(convert_units(1.0, 'hp', 'W'), 746.0, 0.001)
    assert_near_equal(convert_units(1.0, 'hp', 'psi*galUS/min'), 1714.0, 0.001)


def test_accessory_loss_partials():
    p = _accessories(nn=2, load_elec=[2200.0, 3000.0], flow_hyd=[1.3, 4.0], p_hyd=3000.0,
                     eta_gen=0.7, eta_hyd=0.85, P_other=2.0)
    assert_near_equal(p.get_val('P_loss_acc'),
                      np.array([2200.0, 3000.0]) / (0.7 * 746.0)
                      + 3000.0 * np.array([1.3, 4.0]) / (0.85 * 1714.0) + 2.0, 1e-12)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G1 -- EnginePowerRequiredComp, pp. 277-278, 311
# ------------------------------------------------------------------------

BOOK_ACC = 4.0 + 3.0          # p. 278, generator + hydraulic pump, rounded as in the book


def _engine_power(P_MR, P_TR, P_acc, sigma=None, gearboxes=None, design=EXAMPLE_DESIGN):
    nn = len(P_MR)
    opts = dict(num_nodes=nn, density_scaled_losses=sigma is not None)
    if gearboxes is not None:
        opts['gearboxes'] = gearboxes
    p = om.Problem()
    p.model.add_subsystem('epr', EnginePowerRequiredComp(**opts), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in {'P_MR': P_MR, 'P_TR': P_TR, 'P_loss_acc': P_acc, **design}.items():
        p.set_val(name, val)
    if sigma is not None:
        p.set_val('density_ratio', sigma)
    p.run_model()
    return p


def test_engine_power_closes_the_loss_loop_exactly():
    """C4-9: P_req = P_MR + P_TR + L_gearbox(P_req) + P_acc, nose boxes loaded by P_req."""
    P_MR, P_TR, P_acc = np.array([1500.0, 3200.0]), np.array([150.0, 260.0]), np.array([7.0, 9.0])
    P_req = _engine_power(P_MR, P_TR, P_acc).get_val('P_req')
    L_gb = _gearbox({'P_MR': P_MR, 'P_TR': P_TR, 'P_req': P_req},
                    EXAMPLE_DESIGN).get_val('P_loss_gearbox')
    assert_near_equal(P_req, P_MR + P_TR + L_gb + P_acc, 1e-12)


def test_engine_power_example_helicopter_p311():
    """p. 311: P_req/sigma = 56 + 1.0112 P_MR/sigma + 1.0075 P_TR/sigma."""
    P_MR, P_TR = np.array([1200.0, 2000.0, 3000.0]), np.array([120.0, 200.0, 300.0])
    for sigma in (1.0, 0.7):
        sig = np.full(3, sigma)
        p = _engine_power(P_MR, P_TR, np.full(3, BOOK_ACC), sigma=sig)
        book = sig * (56.0 + 1.0112 * P_MR / sig + 1.0075 * P_TR / sig)
        assert np.all(np.abs(p.get_val('P_req') - book) < 1.0)
        assert_near_equal(p.get_val('P_loss'), p.get_val('P_req') - P_MR - P_TR, 1e-12)


@pytest.mark.parametrize('scaled', [False, True])
def test_engine_power_partials(scaled):
    layout = {'main': dict(shaft='main_rotor', n_spur_bevel=2, n_planetary=0),
              'tail': dict(shaft='tail_rotor', n_spur_bevel=1, n_planetary=0, n_series=2)}
    for gearboxes, design in ((None, EXAMPLE_DESIGN),
                              (layout, {'P_design_main': 180.0, 'P_design_tail': 30.0})):
        p = _engine_power([140.0, 1500.0], [15.0, 150.0], [2.0, 7.0],
                          sigma=[0.9, 0.6] if scaled else None, gearboxes=gearboxes, design=design)
        assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G1 -- PowerLossesGroup, pp. 275-278, 311
# ------------------------------------------------------------------------

def test_power_losses_group_example_helicopter_p311():
    """Accessory loads of p. 278 with the drive system of p. 277 give p. 311."""
    P_MR, P_TR = np.array([1500.0, 3000.0]), np.array([150.0, 300.0])
    p = om.Problem()
    p.model.add_subsystem('losses', PowerLossesGroup(num_nodes=2), promotes=['*'])
    p.setup()
    for name, val in {'P_MR': P_MR, 'P_TR': P_TR, 'load_elec': 2200.0, 'flow_hyd': 1.3,
                      'p_hyd': 3000.0, **EXAMPLE_DESIGN}.items():
        p.set_val(name, val)
    p.run_model()
    book = 56.0 + 1.0112 * P_MR + 1.0075 * P_TR
    assert np.all(np.abs(p.get_val('P_req', units='hp') - book) < 1.0)


def test_power_losses_feed_engine_group_fuel_flow():
    """G0 + G1: atmosphere first, then losses, then engines. P_req connects,
    density_ratio is current on the first pass, P_eng (ratings) does not clash."""
    p = om.Problem()
    p.model.add_subsystem('atm', AtmosphereGroup(num_nodes=2, day='isothermal'), promotes=['*'])
    p.model.add_subsystem('losses', PowerLossesGroup(num_nodes=2, density_scaled_losses=True),
                          promotes=['*'])
    p.model.add_subsystem('engine', EngineGroup(num_nodes=2, engine_type='turboshaft',
                                                atmosphere=False), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in {'altitude': [0.0, 4000.0], 'T_day': [518.67, 554.67], 'n_eng': 2.0,
                      'k_inst': 0.02, 'P_MR': [1500.0, 1800.0], 'P_TR': [150.0, 190.0],
                      'load_elec': 2200.0, 'flow_hyd': 1.3, 'p_hyd': 3000.0,
                      **EXAMPLE_DESIGN}.items():
        p.set_val(name, val)
    p.run_model()
    P_req = p.get_val('P_req', units='hp').copy()
    p.run_model()
    assert_near_equal(p.get_val('P_req', units='hp'), P_req, 1e-14)      # no stale inputs

    FF_alone = np.array([_fuel_flow(h, P / 2, day=day, n_eng=2.0)[0] * 2
                         for h, P, day in ((0.0, P_req[0], 'standard'),
                                           (4000.0, P_req[1], 'isothermal'))])
    assert_near_equal(p.get_val('FF', units='lbm/h'), FF_alone, 1e-10)

    data = p.check_totals(of=['P_req', 'FF'], wrt=['P_MR', 'P_TR', 'altitude', 'load_elec'],
                          method='cs', out_stream=None)
    assert_check_totals(data, atol=1e-6, rtol=1e-6)


# ------------------------------------------------------------------------
# G2 -- WakeDynamicPressureComp, Figure 4.6 p. 281, Table 4.1 p. 282
# ------------------------------------------------------------------------

# Table 4.1: segment r/R, Z/R and Prouty's reading of q/D.L. (-10 deg twist)
TABLE_4_1_Q = [(.7, .2, .95), (.6, .2, .80), (.5, .2, .68), (.4, .2, .55), (.3, .2, .45),
               (.2, .2, .20), (.3, .2, .45), (.4, .22, .57), (.5, .22, .70), (.6, .23, .83),
               (.7, .23, .90), (.8, .24, 1.00), (.9, .24, 0.0), (.97, .25, 0.0),
               (.17, .1, .10), (.22, .1, .25), (.27, .1, .35), (.2, .35, .42), (.3, .4, .47)]
TABLE_4_1_OUTLIERS = {(.8, .24), (.2, .35)}          # C4-11


@pytest.mark.parametrize('r, z, q_book', TABLE_4_1_Q)
def test_wake_q_ratio_reproduces_table_4_1(r, z, q_book):
    """C4-11: the -10 deg curves of Figure 4.6 give Prouty's readings."""
    q = wake_q_ratio([r], [z], table=_Q_MINUS10)[0][0]
    tol = 0.30 if (r, z) in TABLE_4_1_OUTLIERS else 0.06
    assert abs(q - q_book) < tol


def test_wake_q_ratio_measured_curves_fig_4_6():
    """-4 deg, z/R = 0.1: tip peak 1.58 near r/R = 0.82, no wake outboard of 0.87."""
    q = wake_q_ratio([0.82, 0.88, 0.10, 0.65], [0.1, 0.1, 0.3, 0.3])[0]
    assert abs(q[0] - 1.58) < 0.03
    assert_near_equal(q[1:3], [0.0, 0.0], 1e-12)
    assert abs(q[3] - 0.93) < 0.03


def test_wake_dynamic_pressure_twist_correction_and_warning():
    p = om.Problem()
    p.model.add_subsystem('w', WakeDynamicPressureComp(num_segments=2), promotes=['*'])
    p.setup()
    p.set_val('seg_r_R', [0.5, 0.5])
    p.set_val('seg_z_R', [0.3, 0.3])
    p.set_val('vi_ratio', [1.0, 1.1])
    p.run_model()
    q = p.get_val('q_DL')
    assert_near_equal(q[1], 1.21 * q[0], 1e-12)

    p.set_val('seg_z_R', [0.3, 0.7])
    with pytest.warns(UserWarning, match='z/R'):
        p.run_model()


def test_wake_dynamic_pressure_partials():
    p = om.Problem()
    p.model.add_subsystem('w', WakeDynamicPressureComp(num_segments=4), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('seg_r_R', [0.35, 0.62, 0.78, 0.83])
    p.set_val('seg_z_R', [0.15, 0.24, 0.37, 0.45])
    p.set_val('vi_ratio', [1.0, 1.05, 0.97, 1.1])
    p.run_model()
    assert_check_partials(p.check_partials(method='fd', step=1e-7, out_stream=None),
                          atol=2e-4, rtol=2e-4)


# ------------------------------------------------------------------------
# G2 -- InducedVelocityRatioComp, p. 279, Figure 4.6 p. 281
# ------------------------------------------------------------------------

EXAMPLE_MAIN_ROTOR = dict(R=30.0, c_root=2.0, c_tip=2.0, r_1=30.0, r_cutout=4.5,
                          V_tip=650.0, b=4.0, altitude=0.0)           # p. 669, as in Chapter 1


@pytest.fixture(scope='module')
def example_inflows():
    """Chapter 1 inflow of the example rotor trimmed to 20,000 lb, twists -10 and -4 deg."""
    from prouty.hover import HoverRotorGroup
    out = {}
    for twist in (-10.0, -4.0):
        p = om.Problem()
        p.model.add_subsystem('rotor', HoverRotorGroup(mode='trim', theta_0_bounds=(2.0, 25.0)),
                              promotes=['*'])
        p.setup()
        for name, val in EXAMPLE_MAIN_ROTOR.items():
            p.set_val(name, val)
        p.set_val('theta_1', twist)
        p.set_val('T_target', 20000.0)
        p.set_val('theta_0', 14.0)
        p.run_model()
        out[twist] = (p.get_val('r_R').copy(), p.get_val('v1_Or').copy())
    return out


def _vi_ratio(inflows, seg_r, seg_z, twist=-10.0):
    r_R, v1 = inflows[twist]
    p = om.Problem()
    p.model.add_subsystem('vr', InducedVelocityRatioComp(num_segments=len(seg_r),
                          num_stations=len(r_R)), promotes=['*'])
    p.setup()
    for name, val in dict(r_R=r_R, v1_Or=v1, v1_Or_ref=inflows[-4.0][1],
                          seg_r_R=seg_r, seg_z_R=seg_z).items():
        p.set_val(name, val)
    p.run_model()
    return p.get_val('vi_ratio')


def test_induced_velocity_ratio_is_one_for_the_reference_twist(example_inflows):
    ratio = _vi_ratio(example_inflows, [0.3, 0.6, 0.8], [0.1, 0.3, 0.5], twist=-4.0)
    assert_near_equal(ratio, np.ones(3), 1e-12)


def _corrected_rms(inflows, contracted=True):
    """rms of (-4 deg curve x ratio^2) - Prouty's -10 deg curve, per panel of Figure 4.6."""
    r_b, v10 = inflows[-10.0]
    v4 = inflows[-4.0][1]
    out = []
    for panel, z in enumerate((0.1, 0.3, 0.5)):
        inside = (_Q_MINUS4[panel] > 0.2) & (_R < 0.8)
        r = _R[inside]
        if contracted:
            ratio = _vi_ratio(inflows, r, np.full(r.size, z))
        else:
            ratio = np.interp(r, r_b, v10) / np.interp(r, r_b, v4)
        err = _Q_MINUS4[panel][inside] * ratio ** 2 - _Q_MINUS10[panel][inside]
        out.append(np.sqrt(np.mean(err ** 2)))
    return np.array(out)


def test_twist_correction_against_the_minus_10_curves_fig_4_6(example_inflows):
    """C4-12: p. 279 method with the Chapter 1 inflow stays within 0.06 of Prouty's
    -10 deg curves, and mapping through the wake contraction beats the direct mapping."""
    contracted = _corrected_rms(example_inflows)
    assert np.all(contracted < 0.06)
    assert contracted.mean() < _corrected_rms(example_inflows, contracted=False).mean()


def test_induced_velocity_ratio_partials(example_inflows):
    r_R, v1 = example_inflows[-10.0]
    p = om.Problem()
    p.model.add_subsystem('vr', InducedVelocityRatioComp(num_segments=5, num_stations=len(r_R)),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in dict(r_R=r_R, v1_Or=v1, v1_Or_ref=example_inflows[-4.0][1],
                          seg_r_R=[0.12, 0.33, 0.57, 0.71, 0.95],
                          seg_z_R=[0.05, 0.17, 0.3, 0.42, 0.6]).items():
        p.set_val(name, val)
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G2 -- VerticalDragComp, p. 280, Table 4.1 p. 282
# ------------------------------------------------------------------------

# Table 4.1: r/R, Z/R, (q/D.L.)_n, C_Dn, A_n (ft^2); 21 segments of the half airframe
TABLE_4_1 = np.array([
    [.7, .2, .95, .1, 6], [.6, .2, .80, .9, 10], [.5, .2, .68, .9, 12], [.4, .2, .55, .9, 12],
    [.3, .2, .45, .9, 12], [.2, .2, .20, .9, 12], [.2, .2, .20, .9, 12], [.3, .2, .45, .9, 11],
    [.4, .22, .57, .9, 10], [.5, .22, .70, .6, 9], [.6, .23, .83, .5, 8], [.7, .23, .90, .45, 7],
    [.8, .24, 1.00, .40, 6], [.9, .24, 0.0, .35, 5], [.97, .25, 0.0, .35, 2],
    [.17, .1, .10, 2.0, 6], [.22, .1, .25, .6, 14], [.27, .1, .35, .6, 12], [.27, .1, .35, .1, 2],
    [.20, .35, .42, 2.0, 1.5], [.3, .4, .47, .25, 1.5]])
DISC_AREA = 2827.0          # p. 282
GW_EXAMPLE = 20000.0


def _vertical_drag(q_DL, CD=TABLE_4_1[:, 3], S=TABLE_4_1[:, 4], cs=False):
    p = om.Problem()
    p.model.add_subsystem('vd', VerticalDragComp(num_segments=len(S)), promotes=['*'])
    p.setup(force_alloc_complex=cs)
    for name, val in dict(seg_CD=CD, seg_A=S, q_DL=q_DL, A=DISC_AREA, GW=GW_EXAMPLE).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_vertical_drag_table_4_1():
    """p. 282: D_v/GW = 2(59.48)/2,827 = 0.042 and sum A_n / A = 2(171)/2,827 = 0.12."""
    p = _vertical_drag(TABLE_4_1[:, 2])
    assert_near_equal(p.get_val('Dv_GW'), 2 * 59.48 / 2827.0, 1e-4)
    assert_near_equal(p.get_val('A_wake_A'), 2 * 171.0 / 2827.0, 1e-12)
    assert_near_equal(p.get_val('T_target', units='lbf'), GW_EXAMPLE * (1 + 2 * 59.48 / 2827.0), 1e-4)
    assert round(p.get_val('Dv_GW')[0], 3) == 0.042


def test_vertical_drag_chain_with_chapter_1_twist_correction(example_inflows):
    """Table 4.1 segments, q/D.L. from Fig. 4.6 (-4 deg) x Chapter 1 ratio^2: C4-12."""
    r, z = TABLE_4_1[:, 0], TABLE_4_1[:, 1]
    q = wake_q_ratio(r, z)[0] * _vi_ratio(example_inflows, r, z) ** 2
    Dv_GW = _vertical_drag(q).get_val('Dv_GW')[0]
    assert abs(Dv_GW - 0.042) < 0.002


def test_vertical_drag_partials():
    p = _vertical_drag(TABLE_4_1[:, 2], cs=True)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G2 -- PseudoGroundEffectComp, pp. 280-283, with GroundEffectComp (Figure 1.41)
# ------------------------------------------------------------------------

def test_ground_effect_figure_1_41_book_readings():
    """p. 67: 0.75 at z/D = 0.3; p. 281: 0.62 at z/D = 0.12; tends to 1 out of ground effect."""
    from prouty.hover.ground_effect_comp import ground_effect_ratio
    v = ground_effect_ratio([0.3, 0.12, 1.5, 3.0])[0]
    assert abs(v[0] - 0.75) < 0.005
    assert abs(v[1] - 0.62) < 0.005
    assert_near_equal(v[2:], [1.0, 1.0], 1e-12)
    z = np.linspace(0.05, 2.0, 400)
    assert np.all(np.diff(ground_effect_ratio(z)[0]) >= -1e-12)


def _pseudo_ground_effect(CT_sigma, z_D=0.12, A_wake_A=2 * 171.0 / 2827.0, cs=False):
    from prouty.hover import GroundEffectComp
    CT_sigma = np.atleast_1d(CT_sigma)
    p = om.Problem()
    p.model.add_subsystem('ge', GroundEffectComp(), promotes=['*'])
    p.model.add_subsystem('pge', PseudoGroundEffectComp(num_nodes=CT_sigma.size), promotes=['*'])
    p.setup(force_alloc_complex=cs)
    for name, val in dict(z_D=z_D, CT_sigma=CT_sigma, sigma=0.085, A_wake_A=A_wake_A).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_pseudo_ground_effect_example_helicopter_p282_p283():
    """dCQ/sigma = -0.0094 (CT/sigma)^1.5; -0.00024 at CT/sigma = 0.086, i.e. 68 hp (C4-2)."""
    p = _pseudo_ground_effect([1.0, 0.086])
    dCQ = p.get_val('dCQ_sigma')
    assert abs(dCQ[0] + 0.0094) < 0.0002
    assert abs(dCQ[1] + 0.00024) < 0.00001
    rho, A, V_tip, sigma = 0.002377, 2827.0, 650.0, 0.085
    hp = -dCQ[1] * sigma * rho * A * V_tip ** 3 / 550.0
    assert abs(hp - 68.0) < 3.0


def test_pseudo_ground_effect_partials():
    p = _pseudo_ground_effect([0.07, 0.09], z_D=0.2, cs=True)
    assert_check_partials(p.check_partials(method='fd', step=1e-7, out_stream=None),
                          atol=1e-6, rtol=1e-4)


# ------------------------------------------------------------------------
# G2 -- GroundProximityDownloadComp, Figure 4.8 p. 285
# ------------------------------------------------------------------------

def _proximity(z_D, configuration='fuselage'):
    z_D = np.atleast_1d(z_D)
    p = om.Problem()
    p.model.add_subsystem('gp', GroundProximityDownloadComp(num_nodes=z_D.size,
                          configuration=configuration), promotes=['*'])
    p.setup()
    p.set_val('z_D', z_D)
    p.run_model()
    return p.get_val('k_Dv'), p.get_val('k_PGE')


def test_ground_proximity_fig_4_8_shapes():
    """Fuselage alone: download reverses below z/D = 0.5. With wing: download vanishes
    near z/D = 0.2 and the pseudo ground effect reverses, most at z/D = 0.5. OGE: 1."""
    k_dv, k_pge = _proximity([0.45, 0.55, 2.6, 3.5])
    assert k_dv[0] < 0.0 < k_dv[1]
    assert_near_equal(np.r_[k_dv[2:], k_pge[2:]], np.ones(4), 1e-12)

    k_dv, k_pge = _proximity([0.2, 0.5, 1.0], configuration='fuselage_wing')
    assert abs(k_dv[0]) < 0.02
    assert abs(k_pge[1] + 0.41) < 0.02 and k_pge[2] > 0.0


def test_ground_proximity_s76_download_reversal():
    """p. 283, S-76 model: +3 % OGE to -1 % close to the ground, k = -1/3 near z/D 0.3."""
    k_dv = _proximity(np.linspace(0.25, 0.4, 31))[0]
    assert k_dv.min() < -1 / 3 < k_dv.max()


def test_ground_proximity_book_switches():
    assert_near_equal(np.array(_proximity([0.3, 1.0], 'none')), np.ones((2, 2)), 1e-12)
    assert_near_equal(np.array(_proximity([0.3, 1.0], 'removed')), np.zeros((2, 2)), 1e-12)


def test_ground_proximity_factors_scale_the_oge_results():
    p = om.Problem()
    p.model.add_subsystem('vd', VerticalDragComp(num_segments=21, num_nodes=2), promotes=['*'])
    p.model.add_subsystem('pge', PseudoGroundEffectComp(num_nodes=2), promotes=['*'])
    p.setup()
    for name, val in dict(seg_CD=TABLE_4_1[:, 3], seg_A=TABLE_4_1[:, 4], q_DL=TABLE_4_1[:, 2],
                          A=DISC_AREA, GW=GW_EXAMPLE, k_Dv=[1.0, -0.3], k_PGE=[1.0, 0.5],
                          CT_sigma=0.086, vi_IGE_OGE=0.62).items():
        p.set_val(name, val)
    p.run_model()
    Dv, dCQ = p.get_val('Dv_GW'), p.get_val('dCQ_sigma')
    assert_near_equal(Dv[1], -0.3 * Dv[0], 1e-12)
    assert_near_equal(dCQ[1], 0.5 * dCQ[0], 1e-12)


@pytest.mark.parametrize('configuration', ['fuselage', 'fuselage_wing'])
def test_ground_proximity_partials(configuration):
    p = om.Problem()
    p.model.add_subsystem('gp', GroundProximityDownloadComp(num_nodes=4,
                          configuration=configuration), promotes=['*'])
    p.setup()
    p.set_val('z_D', [0.33, 0.62, 1.3, 2.2])
    p.run_model()
    assert_check_partials(p.check_partials(method='fd', step=1e-7, out_stream=None),
                          atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------
# G2 -- VerticalDragGroup, pp. 278-285
# ------------------------------------------------------------------------

def _g2_problem(ground_proximity='none', rotor_height_D=3.0):
    p = om.Problem()
    p.model.add_subsystem('g2', VerticalDragGroup(num_segments=21, theta_0_bounds=(2.0, 25.0),
                                                  ground_proximity=ground_proximity),
                          promotes=['*'])
    p.setup()
    for name, val in dict(**EXAMPLE_MAIN_ROTOR, theta_1=-10.0, GW=GW_EXAMPLE,
                          seg_r_R=TABLE_4_1[:, 0], seg_z_R=TABLE_4_1[:, 1],
                          seg_CD=TABLE_4_1[:, 3], seg_A=TABLE_4_1[:, 4],
                          fuselage_Z_D=0.12, rotor_height_D=rotor_height_D).items():
        p.set_val(name, val)
    p.set_val('theta_0', 14.0)
    p.set_val('rotor_ref.theta_0', 14.0)
    p.run_model()
    return p


@pytest.fixture(scope='module')
def g2_example():
    return _g2_problem()


def test_vertical_drag_group_example_helicopter(g2_example):
    """pp. 281-283: D_v/GW = 0.042, A_wake/A = 0.12, dCQ/sigma = -0.00024 at CT/sigma = 0.086.
    Both rotors trimmed to T_target = GW (1 + D_v/GW); the -4 deg rotor needs less collective."""
    p = g2_example
    Dv_GW = p.get_val('Dv_GW')[0]
    assert abs(Dv_GW - 0.042) < 0.002                                     # C4-12
    assert abs(p.get_val('A_wake_A')[0] - 0.12) < 0.001
    assert abs(p.get_val('CT_sigma')[0] - 0.086) < 0.001
    assert abs(p.get_val('dCQ_sigma')[0] + 0.00024) < 0.00001             # C4-2
    T_target = p.get_val('T_target', units='lbf')[0]
    assert_near_equal(T_target, GW_EXAMPLE * (1 + Dv_GW), 1e-9)
    assert_near_equal([p.get_val('T')[0], p.get_val('rotor_ref.T')[0]], [T_target] * 2, 1e-8)
    assert p.get_val('rotor_ref.theta_0')[0] < p.get_val('theta_0')[0]


def test_vertical_drag_group_is_converged(g2_example):
    T_target = g2_example.get_val('T_target').copy()
    g2_example.run_model()
    assert_near_equal(g2_example.get_val('T_target'), T_target, 1e-10)


def test_vertical_drag_group_book_hover_in_ground_effect():
    """p. 309: no vertical drag and no pseudo ground effect in ground effect."""
    p = _g2_problem(ground_proximity='removed', rotor_height_D=0.3)
    assert_near_equal(p.get_val('T_target', units='lbf'), [GW_EXAMPLE], 1e-10)
    assert_near_equal(p.get_val('dCQ_sigma'), [0.0], 1e-12)


def test_vertical_drag_group_totals_match_retrimmed_differences(g2_example):
    """Analytic totals through the Gauss-Seidel loop and both Newton trims (see HoverRotorGroup)."""
    p = g2_example
    J = p.compute_totals(of=['T_target', 'dCQ_sigma'], wrt=['GW', 'fuselage_Z_D'])

    def run(**kw):
        for name, val in kw.items():
            p.set_val(name, val)
        p.run_model()
        return np.array([p.get_val('T_target')[0], p.get_val('dCQ_sigma')[0]])

    dGW = (run(GW=GW_EXAMPLE + 1.0) - run(GW=GW_EXAMPLE - 1.0)) / 2.0
    dZ = (run(fuselage_Z_D=0.13) - run(fuselage_Z_D=0.11)) / 0.02
    run(GW=GW_EXAMPLE, fuselage_Z_D=0.12)
    assert_near_equal([J['T_target', 'GW'][0, 0], J['dCQ_sigma', 'GW'][0, 0]], dGW, 1e-4)
    assert_near_equal(J['dCQ_sigma', 'fuselage_Z_D'][0, 0], dZ[1], 2e-3)


# ------------------------------------------------------------------------
# G3 -- FinInterferenceRatioComp, Figure 4.9 p. 286
# ------------------------------------------------------------------------

def _fin_ratio(S_A, x_R, installation='pusher'):
    p = om.Problem()
    p.model.add_subsystem('fin', FinInterferenceRatioComp(installation=installation), promotes=['*'])
    p.setup()
    p.set_val('S_A', S_A)
    p.set_val('x_R', x_R)
    p.run_model()
    return p.get_val('F_T')[0]


def test_fin_interference_example_helicopter_p286():
    """p. 286: pusher, S/A = 0.25, x/R = 0.3 gives F/T = 0.125."""
    assert abs(_fin_ratio(0.25, 0.3) - 0.125) < 0.003


# Figure 4.9 end points read on the figure: (installation, S/A, x/R, F/T)
FIG_4_9_POINTS = [('tractor', 0.25, 0.2, 0.222), ('tractor', 0.25, 1.0, 0.167),
                  ('tractor', 0.10, 1.0, 0.051), ('tractor', 0.10, 0.2, 0.082),
                  ('pusher', 0.25, 0.2, 0.160), ('pusher', 0.10, 0.2, 0.068),
                  ('pusher', 0.20, 1.0, 0.0)]


@pytest.mark.parametrize('installation, S_A, x_R, F_T', FIG_4_9_POINTS)
def test_fin_interference_fig_4_9_end_points(installation, S_A, x_R, F_T):
    assert abs(_fin_ratio(S_A, x_R, installation) - F_T) < 0.004


def test_fin_interference_trends_and_edges():
    """F/T grows with S/A and falls with x/R; outside the figure the edge is held."""
    x = np.linspace(0.2, 0.9, 8)
    for installation in ('tractor', 'pusher'):
        F = np.array([[_fin_ratio(s, xi, installation) for xi in x] for s in (0.1, 0.17, 0.25)])
        assert np.all(np.diff(F, axis=0) > 0.0)
        assert np.all(np.diff(F, axis=1) <= 1e-12)
    with pytest.warns(UserWarning, match='Figure 4.9'):
        assert_near_equal(_fin_ratio(0.30, 0.3), _fin_ratio(0.25, 0.3), 1e-12)


@pytest.mark.parametrize('installation', ['tractor', 'pusher'])
def test_fin_interference_partials(installation):
    p = om.Problem()
    p.model.add_subsystem('fin', FinInterferenceRatioComp(installation=installation), promotes=['*'])
    p.setup()
    p.set_val('S_A', 0.18)
    p.set_val('x_R', 0.43)
    p.run_model()
    assert_check_partials(p.check_partials(method='fd', step=1e-7, out_stream=None),
                          atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------
# G3 -- TailRotorGrossThrustComp, p. 283
# ------------------------------------------------------------------------

def test_tail_rotor_gross_thrust_p283_and_example():
    """T_net = T_gross (1 - F/T) exactly; the book's 1.125 T_net is 1.6 % low (C4-1)."""
    p = om.Problem()
    p.model.add_subsystem('tg', TailRotorGrossThrustComp(num_nodes=2), promotes=['*'])
    p.setup(force_alloc_complex=True)
    T_req = np.array([0.69 * 1990.0, 0.69 * 2500.0])        # p. 309: T_net = 0.69 hp_M
    p.set_val('T_req', T_req)
    F_T = 0.125                                              # example, p. 286
    p.set_val('F_T', F_T)
    p.run_model()
    T_gross = p.get_val('T_gross', units='lbf')
    assert_near_equal(T_gross * (1 - F_T), T_req, 1e-12)
    assert_near_equal(T_gross / T_req, [1 / 0.875] * 2, 1e-12)
    assert abs(T_gross[0] / (1.125 * T_req[0]) - 1.016) < 0.001
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G3 -- TailRotorFinPowerComp, p. 286
# ------------------------------------------------------------------------

def test_tail_rotor_fin_power_p286_and_fig_4_10():
    """p. 286: factor 0.94 for F/T = 0.125; Figure 4.10: about 94 % for F/T = 0.13."""
    p = om.Problem()
    p.model.add_subsystem('tp', TailRotorFinPowerComp(num_nodes=2), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('P_TR_iso', [200.0, 250.0])
    for F_T in (0.125, 0.13):
        p.set_val('F_T', F_T)
        p.run_model()
        ratio = p.get_val('P_TR', units='hp') / np.array([200.0, 250.0])
        assert_near_equal(ratio, [1 - F_T / 2] * 2, 1e-12)
        assert abs(ratio[0] - 0.94) < 0.006
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G3 -- TailRotorFinInterferenceGroup, pp. 283-287, 309
# ------------------------------------------------------------------------

EXAMPLE_TAIL_ROTOR = dict(tr_R=6.5, tr_c_root=1.0, tr_c_tip=1.0, tr_r_1=6.5, tr_r_cutout=0.975,
                          tr_V_tip=650.0, tr_b=3.0, tr_theta_1=-5.0)   # p. 669, as in Chapter 1
T_REQ_EXAMPLE = 0.69 * 2000.0                                          # p. 309, hp_M about 2,000


def _g3_problem(installation='pusher', S_A=0.25, x_R=0.3, T_req=T_REQ_EXAMPLE):
    p = om.Problem()
    p.model.add_subsystem('g3', TailRotorFinInterferenceGroup(installation=installation,
                                                              theta_0_bounds=(1.0, 30.0)),
                          promotes=['*'])
    p.setup()
    for name, val in dict(**EXAMPLE_TAIL_ROTOR, altitude=0.0, S_A=S_A, x_R=x_R,
                          T_req=T_req).items():
        p.set_val(name, val)
    p.set_val('tr_theta_0', 12.0)
    p.run_model()
    return p


@pytest.fixture(scope='module')
def g3_example():
    return _g3_problem()


def test_tail_rotor_fin_group_example_helicopter(g3_example):
    """pp. 286, 309: pusher, S/A 0.25, x/R 0.3; tail rotor trimmed to T_req/(1 - F/T),
    installed power (1 - (F/T)/2) of the isolated power at that thrust."""
    p = g3_example
    F_T = p.get_val('F_T')[0]
    T_gross = p.get_val('T_gross', units='lbf')[0]
    assert abs(F_T - 0.125) < 0.003
    assert_near_equal(T_gross, T_REQ_EXAMPLE / (1 - F_T), 1e-12)
    assert_near_equal(p.get_val('tail_rotor.T')[0], T_gross, 1e-8)
    assert_near_equal(p.get_val('P_TR')[0], (1 - F_T / 2) * p.get_val('P_TR_iso')[0], 1e-12)
    assert 0.06 < p.get_val('tr_CT_sigma')[0] < 0.12                     # Figure 4.31 range
    assert 100.0 < p.get_val('P_TR')[0] < 400.0


def test_tail_rotor_fin_group_installations():
    """Same geometry: a tractor fin takes more thrust, so needs more power, than a pusher."""
    pusher = _g3_problem('pusher')
    tractor = _g3_problem('tractor')
    assert tractor.get_val('T_gross')[0] > pusher.get_val('T_gross')[0]
    assert tractor.get_val('P_TR')[0] > pusher.get_val('P_TR')[0]


def test_tail_rotor_fin_group_totals(g3_example):
    """Totals through the tail rotor trim match retrimmed central differences."""
    p = g3_example
    J = p.compute_totals(of=['P_TR'], wrt=['T_req', 'S_A'])

    def run(**kw):
        for name, val in kw.items():
            p.set_val(name, val)
        p.run_model()
        return p.get_val('P_TR')[0]

    dT = (run(T_req=T_REQ_EXAMPLE + 1.0) - run(T_req=T_REQ_EXAMPLE - 1.0)) / 2.0
    dS = (run(T_req=T_REQ_EXAMPLE, S_A=0.18 + 1e-4) - run(S_A=0.18 - 1e-4)) / 2e-4   # between curves
    run(S_A=0.25)
    assert_near_equal(J['P_TR', 'T_req'][0, 0], dT, 1e-4)
    p.set_val('S_A', 0.18)
    p.run_model()
    J = p.compute_totals(of=['P_TR'], wrt=['S_A'])
    run(S_A=0.25)
    assert_near_equal(J['P_TR', 'S_A'][0, 0], dS, 1e-3)


def test_g2_and_g3_promote_side_by_side():
    """Main rotor (G2) and tail rotor (G3) in one model: no name clash, shared altitude."""
    p = om.Problem()
    p.model.add_subsystem('g2', VerticalDragGroup(num_segments=21), promotes=['*'])
    p.model.add_subsystem('g3', TailRotorFinInterferenceGroup(), promotes=['*'])
    p.setup()
    p.set_val('altitude', 4000.0)
    p.final_setup()
    targets = [meta['prom_name'] for _, meta in
               p.model.list_inputs(prom_name=True, out_stream=None) if meta['prom_name'] == 'altitude']
    assert len(targets) >= 2                                  # both rotor atmospheres share it
    assert_near_equal(p.get_val('altitude'), [4000.0], 1e-12)


# ------------------------------------------------------------------------
# G4 -- FuselageDragComp, Figure 4.17 p. 294, example p. 306
# ------------------------------------------------------------------------

def _fuselage(l_d=7.0, A_F=74.0, **opts):
    p = om.Problem()
    p.model.add_subsystem('fus', FuselageDragComp(**opts), promotes=['*'])
    p.setup()
    p.set_val('A_F', A_F)
    if opts.get('mode', 'reference') == 'reference':
        p.set_val('l_d', l_d)
    p.run_model()
    return p


def test_fuselage_drag_example_helicopter_p306():
    """A_F = 74 ft^2, l/d = 7: C_DF = 0.078, f_F = 5.8 ft^2 (C4-16)."""
    for reference in ('L-286', 'P-51'):
        p = _fuselage(reference=reference)
        assert abs(p.get_val('C_DF')[0] - 0.078) < 0.003
        assert abs(p.get_val('f_F', units='ft**2')[0] - 5.8) < 0.25
    p = _fuselage(mode='input')
    p.set_val('C_DF', 0.078)
    p.run_model()
    assert_near_equal(p.get_val('f_F'), [74.0 * 0.078], 1e-12)


def test_fuselage_drag_figure_4_17():
    """Minimum curve: 0.036 near l/d = 2.6; each reference aircraft is reproduced at its own l/d."""
    from prouty.performance.fuselage_drag_comp import REFERENCE_AIRCRAFT, minimum_fuselage_drag
    assert abs(minimum_fuselage_drag(2.6)[0] - 0.036) < 0.001
    for name, (l_d, cd) in REFERENCE_AIRCRAFT.items():
        assert_near_equal(_fuselage(l_d=l_d, reference=name).get_val('C_DF'), [cd], 1e-10)


def test_fuselage_drag_partials():
    p = _fuselage(l_d=5.3)
    assert_check_partials(p.check_partials(method='fd', step=1e-6, out_stream=None),
                          atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------
# G4 -- NacelleDragComp, Figure 4.19 p. 296, example p. 306
# ------------------------------------------------------------------------

def _nacelles(D_N=2.8, y_N=1.4, num_nacelles=2):
    p = om.Problem()
    p.model.add_subsystem('nac', NacelleDragComp(num_nacelles=num_nacelles), promotes=['*'])
    p.setup()
    p.set_val('D_N', D_N)
    p.set_val('y_N', y_N)
    p.run_model()
    return p


def test_nacelle_drag_example_helicopter_p306():
    """Two nacelles, D_N = 2.8 ft, y/D_N = 0.5: A_N = 12 ft^2, C_DN = 0.09, f_N = 1.1 ft^2 (C4-17)."""
    p = _nacelles()
    assert_near_equal(p.get_val('y_D_N'), [0.5], 1e-12)
    assert abs(p.get_val('A_N', units='ft**2')[0] - 12.0) < 0.35
    assert abs(p.get_val('C_DN')[0] - 0.09) < 0.003
    assert abs(p.get_val('f_N', units='ft**2')[0] - 1.1) < 0.05


def test_nacelle_drag_figure_4_19():
    """0.18 against the fuselage, falling to about 0.075 beyond one diameter; held past 0.9."""
    from prouty.performance.nacelle_drag_comp import nacelle_drag_coefficient
    assert abs(nacelle_drag_coefficient(0.0)[0] - 0.18) < 0.005
    for y_D, cd in ((0.302, 0.106), (0.504, 0.088), (0.771, 0.078)):     # measured points
        assert abs(nacelle_drag_coefficient(y_D)[0] - cd) < 0.003
    y = np.linspace(0.0, 0.9, 50)
    assert np.all(np.diff([nacelle_drag_coefficient(v)[0] for v in y]) <= 1e-12)
    with pytest.warns(UserWarning, match='Figure 4.19'):
        _nacelles(D_N=1.0, y_N=1.5)


def test_nacelle_drag_partials():
    p = _nacelles(D_N=2.5, y_N=0.9)
    assert_check_partials(p.check_partials(method='fd', step=1e-7, out_stream=None),
                          atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------
# G4 -- RotorHubDragComp, Table 4.2 p. 298, Figure 4.22 p. 299, example pp. 306-307
# ------------------------------------------------------------------------

def _hub(A_hub, C_D0=1.1, alpha_s=0.0, rpm_pct=100.0, fairing='unfaired'):
    p = om.Problem()
    p.model.add_subsystem('hub', RotorHubDragComp(fairing=fairing), promotes=['*'])
    p.setup()
    for name, val in dict(A_hub=A_hub, C_D0=C_D0, alpha_s=alpha_s, rpm_pct=rpm_pct).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_rotor_hub_drag_example_helicopter_p306():
    """Main hub 5 ft^2 and tail hub 0.6 ft^2, C_D0 = 1.1, alpha 0, 100 % rpm:
    DR = 1.00/0.95 = 1.05, C_D = 1.16, f_MH = 5.8 ft^2, f_T = 0.7 ft^2 (C4-18)."""
    main, tail = _hub(5.0), _hub(0.6)
    assert abs(main.get_val('DR')[0] - 1.05) < 0.006
    assert abs(main.get_val('C_D')[0] - 1.16) < 0.006
    assert abs(main.get_val('f_hub', units='ft**2')[0] - 5.8) < 0.05
    assert abs(tail.get_val('f_hub', units='ft**2')[0] - 0.7) < 0.01


def test_rotor_hub_drag_figure_4_22_trends():
    """Stopped hub: DR = 1 (Table 4.2 condition). Nose-down shaft: the faired hub loses its
    advantage (1.56 at -10 deg against 1.10 unfaired); spinning costs the faired hub more."""
    from prouty.performance.rotor_hub_drag_comp import TABLE_4_2, hub_drag_ratio
    for fairing in ('unfaired', 'faired'):
        assert_near_equal(hub_drag_ratio(0.0, 0.0, fairing)[0], 1.0, 1e-12)
    assert hub_drag_ratio(-10.0, 0.0, 'faired')[0] > 1.5 > hub_drag_ratio(-10.0, 0.0, 'unfaired')[0]
    assert hub_drag_ratio(0.0, 100.0, 'faired')[0] > hub_drag_ratio(0.0, 100.0, 'unfaired')[0]
    assert TABLE_4_2['OH-6A'][3:] == (1.13, 0.80)


def test_rotor_hub_drag_partials():
    for fairing in ('unfaired', 'faired'):
        p = _hub(4.2, C_D0=0.9, alpha_s=-3.3, rpm_pct=73.0, fairing=fairing)
        assert_check_partials(p.check_partials(method='fd', step=1e-6, out_stream=None),
                              atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------
# G4 -- RotorShaftDragComp, Figure 4.23 p. 300, example p. 306
# ------------------------------------------------------------------------

def _shaft(V_kt=115.0, D_s=0.5, l_s=2.0, altitude=0.0):
    p = om.Problem()
    p.model.add_subsystem('atm', AtmosphereComp(), promotes=['*'])
    p.model.add_subsystem('shaft', RotorShaftDragComp(), promotes=['*'])
    p.setup()
    for name, val in dict(altitude=altitude, V=V_kt * KT, D_s=D_s, l_s=l_s).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_rotor_shaft_drag_example_helicopter_p306():
    """D_s = 0.5 ft at 115 kt: R.N. = 6,400(115)(1.69)(0.5) = 0.6e6, C_D = 0.3,
    f_MS = 0.3 ft^2 for 1 ft^2 (C4-19)."""
    p = _shaft()
    assert abs(p.get_val('RN')[0] / 0.6e6 - 1.0) < 0.05
    assert_near_equal(p.get_val('RN')[0], 6400 * 115 * 1.69 * 0.5, 0.01)
    assert abs(p.get_val('C_D')[0] - 0.3) < 0.035
    assert abs(p.get_val('f_s', units='ft**2')[0] - 0.3) < 0.035


def test_cylinder_drag_figure_4_23():
    """Subcritical 1.2, drag crisis between 1e5 and 6e5, supercritical about 0.3; the
    150 kt diameter scale: 2.3 in at 3e5, 23 in at 3e6."""
    from prouty.performance.rotor_shaft_drag_comp import cylinder_drag_coefficient
    assert_near_equal(cylinder_drag_coefficient(3e4)[0], 1.2, 1e-3)
    assert cylinder_drag_coefficient(2e5)[0] > 1.0 > 0.4 > cylinder_drag_coefficient(4e5)[0]
    assert abs(cylinder_drag_coefficient(3e7)[0] - 0.31) < 0.01
    for d_in, RN in ((2.3, 3e5), (23.0, 3e6)):
        assert abs(_shaft(V_kt=150.0, D_s=d_in / 12).get_val('RN')[0] / RN - 1) < 0.05


def test_rotor_shaft_drag_partials():
    p = om.Problem()
    p.model.add_subsystem('shaft', RotorShaftDragComp(), promotes=['*'])
    p.setup()
    for name, val in dict(rho=0.0021, T_air=500.0, V=180.0, D_s=0.4, l_s=1.8).items():
        p.set_val(name, val)
    p.run_model()
    assert_check_partials(p.check_partials(method='fd', step=1e-7, form='central',
                                           out_stream=None), atol=1e-4, rtol=2e-4)


# ------------------------------------------------------------------------
# G4 -- HubPylonInterferenceComp, Figure 4.24 p. 301, example p. 306
# ------------------------------------------------------------------------

def _hub_pylon(Z=2.8, W_p=9.0, alpha_F=-5.0, f_hub=5.8, f_shaft=0.3):
    p = om.Problem()
    p.model.add_subsystem('int', HubPylonInterferenceComp(), promotes=['*'])
    p.setup()
    for name, val in dict(Z=Z, W_p=W_p, alpha_F=alpha_F, f_hub=f_hub, f_shaft=f_shaft).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_hub_pylon_interference_example_helicopter_p306():
    """Z/W_p = 2.8/9 = 0.3, alpha_F = -5 deg: the book reads K_i = 0.15 and f_M = 7.0 ft^2;
    the digitized figure gives 0.19 and 7.2 ft^2 (C4-20)."""
    with pytest.warns(UserWarning, match='Figure 4.24'):
        p = _hub_pylon()
    assert_near_equal(p.get_val('Z_Wp')[0], 2.8 / 9.0, 1e-12)
    assert abs(p.get_val('K_i')[0] - 0.19) < 0.02
    assert abs(p.get_val('f_M', units='ft**2')[0] - 7.2) < 0.15
    assert_near_equal(p.get_val('f_M')[0], (1 + p.get_val('K_i')[0]) * 6.1, 1e-12)


def test_hub_pylon_interference_figure_4_24():
    """Curves of Figure 4.24: K_i falls steeply with the gap and grows with angle of attack."""
    from prouty.performance.hub_pylon_interference_comp import hub_pylon_factor
    at_03 = [hub_pylon_factor(0.3, a)[0] for a in (-3.0, 0.0, 3.0, 6.0, 9.0)]
    assert_near_equal(at_03, [0.224, 0.281, 0.314, 0.368, 0.442], 1e-3)
    assert abs(hub_pylon_factor(0.0, 9.0)[0] - 1.56) < 0.03
    assert abs(hub_pylon_factor(0.0, -3.0)[0] - 0.78) < 0.03
    z = np.linspace(0.0, 0.7, 40)
    assert np.all(np.diff([hub_pylon_factor(v, 0.0)[0] for v in z]) < 0.0)


def test_hub_pylon_interference_partials():
    p = _hub_pylon(Z=2.2, W_p=7.5, alpha_F=1.7)
    assert_check_partials(p.check_partials(method='fd', step=1e-6, out_stream=None),
                          atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------
# G4 -- LandingGearDragComp, Figure 4.26 p. 303, example p. 307
# ------------------------------------------------------------------------

def _gear(A_gear, C_D=None, e=None, d=None, strut='round'):
    mode = 'catalog' if C_D is not None else 'nose_wheel'
    p = om.Problem()
    p.model.add_subsystem('gear', LandingGearDragComp(mode=mode, strut=strut), promotes=['*'])
    p.setup()
    p.set_val('A_gear', A_gear)
    if C_D is not None:
        p.set_val('C_D', C_D)
    else:
        p.set_val('e', e)
        p.set_val('d', d)
    p.run_model()
    return p


def test_landing_gear_drag_example_helicopter_p307():
    """Main gear 4 ft^2 at C_D = 0.3: 1.2 ft^2. Nose gear 1.5 ft^2, e/d = 3.4/2 = 1.7:
    the book reads C_D = 0.56, the digitized curve 0.545, f = 0.8 ft^2 (C4-21)."""
    assert_near_equal(_gear(4.0, C_D=0.3).get_val('f_gear'), [1.2], 1e-12)
    nose = _gear(1.5, e=3.4, d=2.0)
    assert abs(nose.get_val('C_D')[0] - 0.56) < 0.02
    assert abs(nose.get_val('f_gear', units='ft**2')[0] - 0.8) < 0.05


def test_landing_gear_figure_4_26():
    """Catalogue values, and the e/d curve: shielded wheel below e/d = 1, strut emerging above."""
    from prouty.performance.landing_gear_drag_comp import (LANDING_GEAR_CD,
                                                           nose_wheel_drag_coefficient)
    assert LANDING_GEAR_CD['skid, tubular'] == 1.01 and LANDING_GEAR_CD['skid, faired'] == 0.40
    assert abs(nose_wheel_drag_coefficient(1.0)[0] - 0.46) < 0.02
    assert abs(nose_wheel_drag_coefficient(3.0)[0] - 0.81) < 0.02
    round_, faired = (nose_wheel_drag_coefficient(2.0, s)[0] for s in ('round', 'faired'))
    assert round_ > faired                                    # the fairing pays above e/d = 1.3
    assert_near_equal(nose_wheel_drag_coefficient(0.75, 'round')[0],
                      nose_wheel_drag_coefficient(0.75, 'faired')[0], 1e-12)


@pytest.mark.parametrize('strut', ['round', 'faired'])
def test_landing_gear_drag_partials(strut):
    p = _gear(1.8, e=2.6, d=1.7, strut=strut)
    assert_check_partials(p.check_partials(method='fd', step=1e-6, out_stream=None),
                          atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------
# G4 -- StabilizerDragComp, Figures 4.12 and 4.21, example p. 307
# ------------------------------------------------------------------------

def _stab(A, b, MAC, t_c=0.12, C_L=0.0, q_ratio=0.75, V_kt=115.0, num_junctions=2):
    p = om.Problem()
    p.model.add_subsystem('atm', AtmosphereComp(), promotes=['*'])
    p.model.add_subsystem('stab', StabilizerDragComp(num_junctions=num_junctions), promotes=['*'])
    p.setup()
    for name, val in dict(altitude=0.0, A=A, b=b, MAC=MAC, t_c=t_c, C_L=C_L,
                          q_ratio=q_ratio, V=V_kt * KT).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_stabilizer_drag_example_helicopter_p307():
    """Horizontal: A = 18, b = 9, MAC = 2, t/c = .12, C_L = -.3, q/q = .75 -> R.N. = 2e6,
    C_d0 = .010, C_Di = .008, junction .001, f_H = 0.2 ft^2. Vertical: A = 24, MAC = 3,
    no lift, no junction -> f_V = 0.2 ft^2 (C4-22)."""
    h = _stab(18.0, 9.0, 2.0, C_L=-0.3)
    assert_near_equal(h.get_val('RN')[0], 6400 * 115 * 1.69 * 2.0, 0.01)
    assert abs(h.get_val('C_D')[0] - 0.019) < 0.002
    assert abs(h.get_val('f_stab', units='ft**2')[0] - 0.2565) < 0.05

    v = _stab(24.0, 8.0, 3.0, num_junctions=0)
    assert abs(v.get_val('C_D')[0] - 0.010) < 0.002
    assert abs(v.get_val('f_stab', units='ft**2')[0] - 0.18) < 0.05


def test_skin_friction_and_junction_charts():
    """Figure 4.12: forced turbulent 0.0040 at 2e6, close to 0.455/(log R.N.)^2.58;
    natural transition dips to 0.0023 near 3e5. Figure 4.21: C_dJ = 0.076 at t/c = 0.12."""
    from prouty.performance.stabilizer_drag_comp import junction_drag_coefficient, skin_friction
    cf = skin_friction(2e6)[0]
    assert abs(cf - 0.0040) < 0.0003
    assert abs(cf - 0.455 / np.log10(2e6) ** 2.58) < 0.0004
    assert abs(skin_friction(3e5, 'natural')[0] - 0.0023) < 0.0004
    assert abs(junction_drag_coefficient(0.12)[0] - 0.072) < 0.006
    assert_near_equal(junction_drag_coefficient(0.05)[0], 0.0, 1e-12)


def test_stabilizer_drag_partials():
    p = _stab(16.0, 8.0, 2.2, t_c=0.15, C_L=-0.25)
    assert_check_partials(p.check_partials(method='fd', step=1e-6, form='central',
                                           out_stream=None), atol=2e-4, rtol=2e-3)


# ------------------------------------------------------------------------
# G4 -- RotorFuselageInterferenceComp, Figure 4.25 p. 302, example p. 308
# ------------------------------------------------------------------------

def _interference(A_F=74.0, alpha_F=0.0):
    p = om.Problem()
    p.model.add_subsystem('int', RotorFuselageInterferenceComp(), promotes=['*'])
    p.setup()
    p.set_val('A_F', A_F)
    p.set_val('alpha_F', alpha_F)
    p.run_model()
    return p


def test_rotor_fuselage_interference_example_helicopter_p308():
    """p. 308: dC_D read at alpha_F = 0 gives 0.018, f_int = 0.018(74) = 1.3 ft^2 (C4-23)."""
    p = _interference()
    assert abs(p.get_val('dC_D')[0] - 0.018) < 0.0005
    assert abs(p.get_val('f_int', units='ft**2')[0] - 1.3) < 0.05


def test_rotor_fuselage_interference_figure_4_25():
    """Figure 4.25: 0.006 at -10 deg rising to 0.023 at 8 deg, flattening at the top."""
    from prouty.performance.rotor_fuselage_interference_comp import rotor_fuselage_interference
    assert abs(rotor_fuselage_interference(-10.0)[0] - 0.0058) < 0.0005
    assert abs(rotor_fuselage_interference(8.0)[0] - 0.0229) < 0.0005
    a = np.linspace(-10.0, 8.0, 40)
    slopes = np.diff([rotor_fuselage_interference(v)[0] for v in a])
    assert np.all(slopes > 0) and slopes[0] > 2 * slopes[-1]
    with pytest.warns(UserWarning, match='Figure 4.25'):
        _interference(alpha_F=-14.0)


def test_rotor_fuselage_interference_partials():
    p = _interference(alpha_F=-3.5)
    assert_check_partials(p.check_partials(method='fd', step=1e-6, out_stream=None),
                          atol=1e-6, rtol=1e-4)


# ------------------------------------------------------------------------
# G4 -- ExhaustDragComp, p. 304, example p. 308
# ------------------------------------------------------------------------

def _exhaust(mode='thrust', V_kt=115.0, **vals):
    p = om.Problem()
    p.model.add_subsystem('atm', AtmosphereComp(), promotes=['*'])
    p.model.add_subsystem('ex', ExhaustDragComp(mode=mode), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('altitude', 0.0)
    p.set_val('V', V_kt * KT)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_exhaust_drag_example_helicopter_p308():
    """T_net = 2(-11) = -22 lb at 115 kt, q = 45 psf: f_ex = 0.5 ft^2."""
    p = _exhaust(T_res=-11.0, n_eng=2.0)
    assert abs(p.get_val('q', units='lbf/ft**2')[0] - 45.0) < 0.5
    assert abs(p.get_val('f_ex', units='ft**2')[0] - 0.5) < 0.02


def test_exhaust_drag_momentum_form_p304():
    """D_ex = m_dot (V - V_ex cos chi): thrust while V_ex cos chi exceeds V, drag beyond;
    canting the stack away from rearward adds drag."""
    fast = _exhaust('momentum', V_kt=115.0, m_dot=0.6, V_ex=400.0, chi=0.0)
    assert fast.get_val('f_ex')[0] < 0.0                       # residual thrust
    slow = _exhaust('momentum', V_kt=300.0, m_dot=0.6, V_ex=400.0, chi=0.0)
    assert slow.get_val('f_ex')[0] > 0.0                       # residual drag
    canted = _exhaust('momentum', V_kt=115.0, m_dot=0.6, V_ex=400.0, chi=np.radians(30.0))
    assert canted.get_val('f_ex')[0] > fast.get_val('f_ex')[0]


@pytest.mark.parametrize('mode', ['thrust', 'momentum'])
def test_exhaust_drag_partials(mode):
    vals = dict(T_res=-11.0, n_eng=2.0) if mode == 'thrust' else \
        dict(m_dot=0.6, V_ex=400.0, chi=0.3)
    p = _exhaust(mode, **vals)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G4 -- TotalParasiteDragComp, p. 304 and example p. 308
# ------------------------------------------------------------------------

# p. 308: fuselage, nacelles, main hub group, tail hub, main and nose gear,
# stabilizers, rotor-fuselage interference, exhaust, miscellaneous
EXAMPLE_F = dict(f_F=5.8, f_N=1.1, f_M=7.0, f_T=0.7, f_MLG=1.2, f_NLG=0.8, f_H=0.2, f_V=0.2,
                 f_int=1.3, f_ex=0.5, f_misc=0.5)


def test_total_parasite_drag_example_helicopter_p308():
    """The eleven items of p. 308 add to 19.3 ft^2; p. 304 advises at least 20 % more."""
    p = om.Problem()
    p.model.add_subsystem('tot', TotalParasiteDragComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in EXAMPLE_F.items():
        p.set_val(name, val)
    p.run_model()
    assert_near_equal(p.get_val('f_total', units='ft**2'), [19.3], 1e-10)
    assert_near_equal(p.get_val('f_design'), p.get_val('f_total'), 1e-12)

    p.set_val('margin', 0.20)
    p.run_model()
    assert_near_equal(p.get_val('f_design', units='ft**2'), [19.3 * 1.2], 1e-10)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def test_total_parasite_drag_custom_items():
    """A light helicopter with skids and no nacelles: any breakdown can be summed."""
    p = om.Problem()
    p.model.add_subsystem('tot', TotalParasiteDragComp(items=('F', 'M', 'skids', 'V')),
                          promotes=['*'])
    p.setup()
    for name, val in dict(f_F=2.1, f_M=1.4, f_skids=0.6, f_V=0.1, f_misc=0.3, margin=0.2).items():
        p.set_val(name, val)
    p.run_model()
    assert_near_equal(p.get_val('f_total'), [4.5], 1e-10)
    assert_near_equal(p.get_val('f_design'), [5.4], 1e-10)


# ------------------------------------------------------------------------
# G4 -- ParasiteDragGroup, procedure pp. 306-308
# ------------------------------------------------------------------------

EXAMPLE_AIRFRAME = dict(
    A_F=74.0, l_d=7.0, D_N=2.8, y_N=1.4,
    A_hub_M=5.0, C_D0_M=1.1, alpha_s=0.0, rpm_pct=100.0, D_s=0.5, l_s=2.0,
    Z=2.8, W_p=9.0, alpha_F=-5.0, A_hub_T=0.6, C_D0_T=1.1, alpha_s_T=0.0, rpm_pct_T=100.0,
    A_MLG=4.0, C_D_MLG=0.3, A_NLG=1.5, e_NLG=3.4, d_NLG=2.0,
    A_H=18.0, b_H=9.0, MAC_H=2.0, t_c_H=0.12, C_L_H=-0.3, q_ratio_H=0.75,
    A_V=24.0, b_V=8.0, MAC_V=3.0, t_c_V=0.12, C_L_V=0.0, q_ratio_V=0.75,
    T_res=-11.0, n_eng=2.0, f_misc=0.5)                       # pp. 306-308, 115 kt


@pytest.fixture(scope='module')
def g4_example():
    p = om.Problem()
    p.model.add_subsystem('atm', AtmosphereComp(), promotes=['*'])
    p.model.add_subsystem('g4', ParasiteDragGroup(), promotes=['*'])
    p.setup()
    p.set_val('altitude', 0.0)
    p.set_val('V', 115.0 * KT)
    for name, val in EXAMPLE_AIRFRAME.items():
        p.set_val(name, val)
    p.run_model()
    return p


@pytest.mark.parametrize('item, book', [('f_F', 5.8), ('f_N', 1.1), ('f_M', 7.0), ('f_T', 0.7),
                                        ('f_MLG', 1.2), ('f_NLG', 0.8), ('f_H', 0.2),
                                        ('f_V', 0.2), ('f_ex', 0.5)])
def test_parasite_drag_group_items_p306_p308(g4_example, item, book):
    assert abs(g4_example.get_val(item, units='ft**2')[0] - book) < 0.16


def test_parasite_drag_group_total_p308(g4_example):
    """The eleven items of the procedure: 19.3 ft^2 in the book. The interference term is
    read at zero angle of attack (alpha_F_int = 0, p. 308), the hub-pylon factor at -5 deg."""
    p = g4_example
    assert abs(p.get_val('f_int', units='ft**2')[0] - 1.3) < 0.05
    assert abs(p.get_val('f_total', units='ft**2')[0] - 19.3) < 0.5
    assert_near_equal(p.get_val('f_design'), p.get_val('f_total'), 1e-12)


def test_parasite_drag_group_totals(g4_example):
    """Totals of the whole chain: frontal area, hub gap and stabilizer area."""
    J = g4_example.compute_totals(of=['f_total'], wrt=['A_F', 'Z', 'A_H'])
    assert J['f_total', 'A_F'][0, 0] > 0.0                    # fuselage and interference
    assert J['f_total', 'Z'][0, 0] < 0.0                      # a wider gap unloads the pylon
    assert J['f_total', 'A_H'][0, 0] > 0.0


# ------------------------------------------------------------------------
# G5 -- TailRotorThrustComp, p. 309
# ------------------------------------------------------------------------

def test_tail_rotor_thrust_example_helicopter_p309():
    """p. 309: T_T_net = 550 hp_M R_M / ((Omega R)_M l_T) = 0.69 hp_M for the example."""
    p = om.Problem()
    p.model.add_subsystem('yaw', TailRotorThrustComp(num_nodes=2), promotes=['*'])
    p.setup(force_alloc_complex=True)
    P_MR = np.array([1500.0, 2000.0])
    for name, val in dict(P_MR=P_MR, R_M=30.0, V_tip_M=650.0, l_T=36.8).items():
        p.set_val(name, val)
    p.run_model()
    T = p.get_val('T_req', units='lbf')
    assert_near_equal(T, 550 * P_MR * 30.0 / (650.0 * 36.8), 1e-12)
    assert np.all(np.abs(T / P_MR - 0.69) < 0.005)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G5 -- RotorLoadingMatchComp, Figures 4.28 and 4.31, discussion p. 310
# ------------------------------------------------------------------------

def _loading(CT_M, CT_T, max_M=0.167, max_T=0.155):
    CT_M, CT_T = np.atleast_1d(CT_M), np.atleast_1d(CT_T)
    p = om.Problem()
    p.model.add_subsystem('match', RotorLoadingMatchComp(num_nodes=CT_M.size), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in dict(CT_sigma_M=CT_M, CT_sigma_T=CT_T,
                          CT_sigma_max_M=max_M, CT_sigma_max_T=max_T).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_rotor_loading_match_example_helicopter_p310():
    """Figure 4.31: at design gross weight both rotors are near C_T/sigma = 0.086 and 0.10,
    but against their own maxima (0.167 and 0.155, Figure 4.28) the tail rotor is the more
    loaded, so it limits the high gross weight performance (p. 310)."""
    p = _loading(0.086, 0.100)
    assert_near_equal(p.get_val('util_M')[0], 0.086 / 0.167, 1e-12)
    assert_near_equal(p.get_val('margin_T')[0], 1 - 0.100 / 0.155, 1e-12)
    assert p.get_val('mismatch')[0] > 1.0
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def test_rotor_loading_match_is_a_usable_constraint():
    """margin crosses zero at the rotor maximum; a bigger tail rotor cures the mismatch."""
    p = _loading([0.167, 0.150], [0.100, 0.100])
    assert_near_equal(p.get_val('margin_M'), [0.0, 1 - 0.150 / 0.167], 1e-12)
    cured = _loading(0.086, 0.100 * (0.146 / 0.20))          # more tail rotor solidity
    assert cured.get_val('mismatch')[0] < 1.0


# ------------------------------------------------------------------------
# G5 -- HoverCeilingBalance, pp. 311-312
# ------------------------------------------------------------------------

def _ceiling_problem(GW, num_nodes=1, day='standard'):
    """Power required rising with altitude against a rating falling with density."""
    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', HoverCeilingBalance(num_nodes=num_nodes), promotes=['*'])
    model.add_subsystem('atm', AtmosphereGroup(num_nodes=num_nodes, day=day), promotes=['*'])
    model.add_subsystem('power', om.ExecComp(
        ['P_req = 0.0007 * GW ** 1.5 / density_ratio ** 0.5',
         'P_avail = 2400.0 * density_ratio ** 0.9'],
        P_req={'units': 'hp', 'shape': num_nodes}, P_avail={'units': 'hp', 'shape': num_nodes},
        GW={'units': 'lbf', 'shape': num_nodes}, density_ratio={'shape': num_nodes}),
        promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=0)
    model.linear_solver = om.DirectSolver()
    p.setup(force_alloc_complex=True)
    p.set_val('GW', GW)
    p.run_model()
    return p


def test_hover_ceiling_balance_closes_the_power_balance():
    """The altitude comes out where P_req meets P_avail, and heavier means lower."""
    p = _ceiling_problem(20000.0)
    assert_near_equal(p.get_val('P_req', units='hp'), p.get_val('P_avail', units='hp'), 1e-8)
    light, heavy = (_ceiling_problem(w).get_val('altitude', units='ft')[0]
                    for w in (18000.0, 22000.0))
    assert light > heavy > 0.0

    hot = _ceiling_problem(20000.0, day='isothermal').get_val('altitude', units='ft')[0]
    assert hot < p.get_val('altitude', units='ft')[0]        # a hot day lowers the ceiling


def test_hover_ceiling_balance_vectorized_and_differentiable():
    p = _ceiling_problem(np.array([16000.0, 19000.0, 22000.0]), num_nodes=3)
    h = p.get_val('altitude', units='ft')
    assert np.all(np.diff(h) < 0.0)

    J = p.compute_totals(of=['altitude'], wrt=['GW'])
    dGW = 100.0
    up = _ceiling_problem(np.array([16000.0, 19000.0, 22000.0]) + dGW, num_nodes=3)
    down = _ceiling_problem(np.array([16000.0, 19000.0, 22000.0]) - dGW, num_nodes=3)
    fd = (up.get_val('altitude', units='ft') - down.get_val('altitude', units='ft')) / (2 * dGW)
    assert_near_equal(np.diag(J['altitude', 'GW']), fd, 1e-4)


# ------------------------------------------------------------------------
# G5 -- GroundEffectPowerComp (Chapter 1 pp. 67-68), used by hover performance
# ------------------------------------------------------------------------

def _ge_power(CT_sigma=0.085, sigma=0.085, vi=0.75, Dv_GW=None):
    from prouty.hover import GroundEffectPowerComp
    mode = 'none' if Dv_GW is None else 'included'
    p = om.Problem()
    p.model.add_subsystem('ge', GroundEffectPowerComp(vertical_drag=mode), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in dict(CT_sigma=CT_sigma, sigma=sigma, vi_IGE_OGE=vi).items():
        p.set_val(name, val)
    if Dv_GW is not None:
        p.set_val('Dv_GW', Dv_GW)
    p.run_model()
    return p


def test_ground_effect_power_example_p67_p68():
    """pp. 67-68: C_T/sigma = 0.085, D_v/GW = 0.04, v_IGE/v_OGE = 0.75 saves 407 hp out of
    2,000 on the example rotor. The printed 0.000143 is out by ten (C4-26)."""
    p = _ge_power(Dv_GW=0.04)
    dCQ_sigma = p.get_val('dCQ_sigma')[0]
    assert abs(dCQ_sigma + 0.00151) < 0.00005
    rho, A, V_tip, sigma = 0.002377, 2827.0, 650.0, 0.085
    hp = -dCQ_sigma * sigma * rho * A * V_tip ** 3 / 550.0
    assert abs(hp - 407.0) < 25.0
    assert abs(dCQ_sigma / 0.000143) > 5.0                    # the printed value is ten times low
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def test_ground_effect_power_vertical_drag_and_thrust_ratio():
    """Losing the download in ground effect adds to the saving; out of ground effect nothing
    changes; at constant power the thrust grows as (v_IGE/v_OGE)^(-2/3), p. 68."""
    plain = _ge_power().get_val('dCQ_sigma')[0]
    with_drag = _ge_power(Dv_GW=0.04).get_val('dCQ_sigma')[0]
    assert with_drag < plain < 0.0
    assert_near_equal(_ge_power(Dv_GW=0.0).get_val('dCQ_sigma')[0], plain, 1e-12)
    assert_near_equal(_ge_power(vi=1.0).get_val('dCQ_sigma')[0], 0.0, 1e-12)
    assert_near_equal(_ge_power(vi=0.75).get_val('T_ratio_IGE')[0], 0.75 ** (-2 / 3), 1e-12)


# ------------------------------------------------------------------------
# G5 -- HoverPerformanceGroup, pp. 308-312
# ------------------------------------------------------------------------

EXAMPLE_HOVER = dict(
    **EXAMPLE_MAIN_ROTOR, theta_1=-10.0, GW=GW_EXAMPLE,
    seg_r_R=TABLE_4_1[:, 0], seg_z_R=TABLE_4_1[:, 1], seg_CD=TABLE_4_1[:, 3],
    seg_A=TABLE_4_1[:, 4], fuselage_Z_D=0.12, rotor_height_D=3.0,
    l_T=36.8, S_A=0.25, x_R=0.3, **EXAMPLE_TAIL_ROTOR,
    **{k: v for k, v in EXAMPLE_DESIGN.items()},
    load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0, n_eng=2.0, k_inst=0.02)


def _hover_problem(mode='performance', day='standard', GW=GW_EXAMPLE, altitude=0.0):
    p = om.Problem()
    p.model.add_subsystem('g5', HoverPerformanceGroup(num_segments=21, mode=mode, day=day),
                          promotes=['*'])
    p.setup()
    for name, val in EXAMPLE_HOVER.items():
        if name != 'altitude':
            p.set_val(name, val)
    p.set_val('GW', GW)
    p.set_val('altitude', altitude if mode == 'performance' else 5000.0)   # ceiling: first guess
    p.set_val('theta_0', 17.0)
    p.set_val('rotor_ref.theta_0', 13.0)
    p.set_val('tr_theta_0', 15.0)
    p.run_model()
    return p


@pytest.fixture(scope='module')
def g5_example():
    return _hover_problem()


def test_hover_performance_group_engine_power_p311(g5_example):
    """p. 311: h.p._eng = 56 + 1.0112 h.p._M + 1.0075 h.p._T. The whole chain, from the
    download of G2 to the drive losses of G1, must reproduce it."""
    p = g5_example
    P_MR, P_TR, P_req = (p.get_val(k, units='hp')[0] for k in ('P_MR', 'P_TR', 'P_req'))
    assert_near_equal(P_req, 56.0 + 1.0112 * P_MR + 1.0075 * P_TR, 2e-3)
    assert abs(P_MR - 2000.0) < 100.0                    # "approximately 2,000 h.p.", p. 68
    assert 150.0 < P_TR < 250.0
    assert abs(p.get_val('P_rating', units='hp')[0] - 2 * 0.98 * 2080.0) < 1.0


def test_hover_performance_group_sea_level_capability_p312():
    """p. 312: the example can hover out of ground effect at sea level, standard day,
    at 27,800 lb with takeoff power."""
    heavy = _hover_problem(GW=27800.0)
    assert abs(heavy.get_val('P_margin', units='hp')[0]) < 100.0
    lighter = _hover_problem(GW=26000.0)
    assert lighter.get_val('P_margin')[0] > heavy.get_val('P_margin')[0] > -100.0


@pytest.mark.parametrize('day, ceiling', [('standard', 11000.0), ('isothermal', 7000.0)])
def test_hover_performance_group_ceilings_fig_4_35(day, ceiling):
    """Figure 4.35, out of ground effect with takeoff power at 20,000 lb: about 11,000 ft
    on a standard day and 7,000 ft on a 95 F day."""
    p = _hover_problem(mode='ceiling', day=day)
    assert abs(p.get_val('altitude', units='ft')[0] - ceiling) < 1000.0
    assert_near_equal(p.get_val('P_req', units='hp'), p.get_val('P_rating', units='hp'), 1e-5)


def test_hover_performance_group_tail_rotor_limits_at_altitude(g5_example):
    """p. 310: the tail rotor is the more loaded of the two against its own maximum, and it
    is the first to run out, here needing 26 deg of collective at 11,000 ft."""
    assert g5_example.get_val('mismatch')[0] > 1.0
    high = _hover_problem(altitude=11000.0)
    assert high.get_val('tr_CT_sigma')[0] > 0.13
    assert high.get_val('margin_T')[0] < high.get_val('margin_M')[0]
    assert high.get_val('tr_theta_0', units='deg')[0] > 25.0


# ------------------------------------------------------------------------
# G6 -- ClimbInducedVelocityComp and VerticalClimbPowerComp, pp. 313-315
# ------------------------------------------------------------------------

def _climb_velocity(T=20860.0, A=2827.0, V_c=(0.0, 50.0), rho=0.002377):
    V_c = np.atleast_1d(V_c)
    p = om.Problem()
    p.model.add_subsystem('vi', ClimbInducedVelocityComp(num_nodes=V_c.size), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in dict(T=np.full(V_c.size, T), rho=rho, A=A, V_c=V_c).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_climb_induced_velocity_p315():
    """v_1hov = sqrt(T/2 rho A) and v_1c + V_c = V_c/2 + sqrt((V_c/2)^2 + v_1hov^2)."""
    p = _climb_velocity()
    v_hov = np.sqrt(20860.0 / (2 * 0.002377 * 2827.0))
    assert_near_equal(p.get_val('v_hov')[0], v_hov, 1e-12)
    assert_near_equal(p.get_val('v_sum')[0], v_hov, 1e-12)               # hover: v_sum = v_hov
    assert_near_equal(p.get_val('v_sum')[1], 25.0 + np.sqrt(625.0 + v_hov ** 2), 1e-12)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def _climb_power(V_c=(0.0, 30.0), Dv_GW=0.042, dAz_CD=0.0, v_hov_T=40.0):
    V_c = np.atleast_1d(V_c)
    GW, A, rho = 20000.0, 2827.0, 0.002377
    vel = _climb_velocity(T=GW * (1 + Dv_GW), A=A, V_c=V_c, rho=rho)
    p = om.Problem()
    p.model.add_subsystem('dp', VerticalClimbPowerComp(num_nodes=V_c.size), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in dict(GW=np.full(V_c.size, GW), v_hov=vel.get_val('v_hov'),
                          v_sum=vel.get_val('v_sum'), V_c=V_c,
                          Dv_GW=np.full(V_c.size, Dv_GW), rho=rho, A_M=A, dAz_CD=dAz_CD,
                          v_hov_T=v_hov_T, R_M=30.0, V_tip_M=650.0, l_T=36.8).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_vertical_climb_power_is_zero_in_hover_c4_3():
    """C4-3: with the tail rotor factor on the whole difference, dP vanishes at V_c = 0;
    the printed form, with the factor on the hover bracket alone, would not."""
    p = _climb_power(V_c=0.0)
    assert_near_equal(p.get_val('dP')[0], 0.0, 1e-10)
    k_T = p.get_val('k_T')[0]
    assert abs(k_T - (1 + 40.0 * 30.0 / (650.0 * 36.8))) < 1e-12
    assert 1.0 < k_T < 1.1


def test_vertical_climb_power_terms_p314():
    """The download term is D_v v at each speed, and the airframe outside the wake adds
    (dA_z C_D)(rho/2) V_c^3."""
    GW, rho, A, Dv_GW = 20000.0, 0.002377, 2827.0, 0.042
    p = _climb_power(V_c=(0.0, 30.0))
    vel = _climb_velocity(T=GW * (1 + Dv_GW), V_c=(0.0, 30.0))
    v_h, v_s = vel.get_val('v_hov')[1], vel.get_val('v_sum')[1]
    k_T = p.get_val('k_T')[0]
    expected = ((GW * (v_s - v_h) + 4 * Dv_GW * (rho / 2) * A * (v_s ** 3 - v_h ** 3))
                * k_T / 550.0)
    assert_near_equal(p.get_val('dP')[1], expected, 1e-10)

    with_body = _climb_power(V_c=30.0, dAz_CD=8.0).get_val('dP')[0]
    assert_near_equal(with_body - _climb_power(V_c=30.0).get_val('dP')[0],
                      8.0 * (rho / 2) * 30.0 ** 3 * k_T / 550.0, 1e-10)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G6 -- VerticalClimbBalance, pp. 313-316
# ------------------------------------------------------------------------

def _climb_balance(P_excess, GW=20000.0, Dv_GW=0.042, rho=0.002377, num_nodes=1):
    """Chain V_c -> induced velocity -> climb power, closed on the excess power."""
    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', VerticalClimbBalance(num_nodes=num_nodes), promotes=['*'])
    model.add_subsystem('vi', ClimbInducedVelocityComp(num_nodes=num_nodes), promotes=['*'])
    model.add_subsystem('power', VerticalClimbPowerComp(num_nodes=num_nodes), promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=0)
    model.linear_solver = om.DirectSolver()
    p.setup(force_alloc_complex=True)
    nn = num_nodes
    for name, val in dict(T=np.full(nn, GW * (1 + Dv_GW)), A=2827.0, rho=rho,
                          GW=np.full(nn, GW), Dv_GW=np.full(nn, Dv_GW), A_M=2827.0,
                          v_hov_T=40.0, R_M=30.0, V_tip_M=650.0, l_T=36.8,
                          P_excess=P_excess).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_vertical_climb_balance_absorbs_the_excess_power():
    """The rate of climb comes out where dP equals the excess power, and grows with it."""
    p = _climb_balance(1000.0)
    assert_near_equal(p.get_val('dP', units='hp'), p.get_val('P_excess', units='hp'), 1e-8)
    fpm = p.get_val('V_c', units='ft/min')[0]
    assert 500.0 < fpm < 5000.0
    assert _climb_balance(1500.0).get_val('V_c')[0] > p.get_val('V_c')[0]
    assert_near_equal(_climb_balance(0.0).get_val('V_c')[0], 0.0, 1e-8)   # at the ceiling


def test_vertical_climb_balance_vectorized_and_differentiable():
    P_excess = np.array([400.0, 1000.0, 1600.0])
    p = _climb_balance(P_excess, num_nodes=3)
    assert np.all(np.diff(p.get_val('V_c')) > 0.0)

    J = p.compute_totals(of=['V_c'], wrt=['P_excess'])
    step = 10.0
    up = _climb_balance(P_excess + step, num_nodes=3).get_val('V_c')
    down = _climb_balance(P_excess - step, num_nodes=3).get_val('V_c')
    assert_near_equal(np.diag(J['V_c', 'P_excess']), (up - down) / (2 * step), 1e-5)


# ------------------------------------------------------------------------
# G6 -- VerticalClimbGroup, pp. 313-317, Figure 4.36
# ------------------------------------------------------------------------

def _climb_problem(day='standard', GW=GW_EXAMPLE, altitude=0.0):
    p = om.Problem()
    p.model.add_subsystem('g6', VerticalClimbGroup(num_segments=21,
                                                   hover_options=dict(day=day)), promotes=['*'])
    p.setup()
    for name, val in EXAMPLE_HOVER.items():
        if name != 'altitude':
            p.set_val(name, val)
    p.set_val('GW', GW)
    p.set_val('altitude', altitude)
    p.set_val('theta_0', 17.0)
    p.set_val('rotor_ref.theta_0', 13.0)
    p.set_val('tr_theta_0', 15.0)
    p.run_model()
    return p


# Figure 4.36 p. 317, 20,000 lb with takeoff power: (day, altitude, rate of climb ft/min)
FIG_4_36 = [('standard', 0.0, 3270.0), ('standard', 5000.0, 2600.0),
            ('isothermal', 0.0, 2430.0), ('isothermal', 5000.0, 1100.0)]


@pytest.mark.parametrize('day, altitude, fpm', FIG_4_36)
def test_vertical_climb_group_fig_4_36(day, altitude, fpm):
    p = _climb_problem(day=day, altitude=altitude)
    assert abs(p.get_val('V_c', units='ft/min')[0] - fpm) < 250.0
    assert_near_equal(p.get_val('dP', units='hp'), p.get_val('P_excess', units='hp'), 1e-6)


def test_vertical_climb_group_falls_to_zero_at_the_ceiling():
    """Figure 4.36: the rate of climb decays to zero near the hover ceiling, 11,200 ft on a
    standard day for this helicopter (G5)."""
    rates = [_climb_problem(altitude=h).get_val('V_c', units='ft/min')[0]
             for h in (0.0, 5000.0, 11000.0)]
    assert rates[0] > rates[1] > rates[2] > 0.0
    assert rates[2] < 600.0


# ------------------------------------------------------------------------
# G7 -- ForwardFlightPowerGroup, pp. 317-319, Figures 4.38 and 4.48
# ------------------------------------------------------------------------

# Chapter 3 example helicopter, Appendix A pp. 669-670 (as in tests/forward_flight)
REF_ROTOR = dict(V_tip=650.0, rho=0.002377, A_b=240.0, sigma=0.084883,
                 theta_1=np.deg2rad(-10.0), a=6.0, gamma=8.05033, R=30.0, GW=20000.0,
                 i_s=0.0, a1s=0.0, l_T_R=1.23, cd_bar=0.0100,
                 delta_3=np.deg2rad(-30.0), V_son=1116.0)

# Figure 4.48 p. 333 (level flight, sea level, 20,000 lb), digitised: speed kt -> engine hp
FIG_4_48_LEVEL = {60: 1131, 80: 1059, 100: 1189, 120: 1577, 140: 2337, 160: 3182}


def _ff_power(V_kt, **overrides):
    p = om.Problem()
    p.model.add_subsystem('ff', ForwardFlightPowerGroup(), promotes=['*'])
    p.setup()
    for name, val in {**REF_ROTOR, **overrides}.items():
        p.set_val(name, val)
    for name, val in EXAMPLE_DESIGN.items():
        p.set_val(name, val)
    for name, val in dict(load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0).items():
        p.set_val(name, val)
    p.set_val('V', V_kt * KT)
    p.set_val('alpha_F', np.deg2rad(-6.0))
    p.set_val('CT_sigma', 0.085)
    p.run_model()
    return p


# Chart method rebuilt at mu = 0.30 (115.5 kt), 20,000 lb, sea level: the isolated rotor
# charts give C_Q/sigma = 0.0035 at theta_0 = 12.3, i.e. 997 hp at the main rotor and
# 1,091 hp at the engine (see docs/diagnostics_chapter3_trim.md)
CHART_METHOD = {115.5: 1091.0}


@pytest.mark.parametrize('V_kt, chart', CHART_METHOD.items())
def test_forward_flight_power_matches_the_chart_method(V_kt, chart):
    """C4-29: the reference is the isolated rotor chart, not Figure 4.48. The chain reads
    a few per cent above the charts, the same excess measured curve by curve on chart 2."""
    P_req = _ff_power(V_kt).get_val('P_req', units='hp')[0]
    assert abs(P_req / chart - 1.0) < 0.12
    assert P_req > chart                                      # +3 to +12 %, never below


@pytest.mark.parametrize('V_kt', [60, 80, 100, 120, 140])
def test_forward_flight_power_below_figure_4_48_c4_29(V_kt):
    """A documented fact about the book's figure, not a tolerance on the model: Figures
    4.38 and 4.48 stand about 37 % above the charts they are said to come from, so the
    chain reads below them, and by more as the speed grows."""
    P_req = _ff_power(V_kt).get_val('P_req', units='hp')[0]
    assert P_req < FIG_4_48_LEVEL[V_kt]


def test_forward_flight_power_compressibility_penalty_p319():
    """p. 319: the additional compressibility loss at 20,000 lb and 140 kt is about 20 hp."""
    p = _ff_power(140)
    assert 0.0 < p.get_val('hp_comp', units='hp')[0] < 40.0


# ------------------------------------------------------------------------
# G7 -- MaxSpeedBalance, p. 320, Figure 4.39
# ------------------------------------------------------------------------

def _max_speed_toy(P_avail, num_nodes=1):
    """A bucket-shaped power required curve, in the shape of Figure 4.48."""
    P_avail = np.atleast_1d(P_avail)
    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', MaxSpeedBalance(num_nodes=num_nodes), promotes=['*'])
    model.add_subsystem('power', om.ExecComp(
        'P_req = 1050 + 0.00052 * (V / KT - 80) ** 3.2',
        P_req={'units': 'hp', 'shape': num_nodes}, V={'units': 'ft/s', 'shape': num_nodes},
        KT=KT), promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40, iprint=0)
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector',
                                                             maxiter=8, iprint=0)
    model.linear_solver = om.DirectSolver()
    p.setup(force_alloc_complex=True)
    p.set_val('P_avail', P_avail)
    p.run_model()
    return p


def test_max_speed_balance_closes_on_the_power_available():
    """V_max sits where the two curves cross, and grows with the power available."""
    p = _max_speed_toy(2350.0)
    assert_near_equal(p.get_val('P_req', units='hp'), p.get_val('P_avail', units='hp'), 1e-8)
    V_max = p.get_val('V', units='kn')[0]
    assert 120.0 < V_max < 180.0
    assert _max_speed_toy(3000.0).get_val('V')[0] > p.get_val('V')[0]


def test_max_speed_balance_trade_off_derivative_p320():
    """p. 320: dV_max/dhp is about 0.009 kn/hp for the example helicopter near its
    intermediate rating; the order of magnitude falls out of the balance itself."""
    P_avail = np.array([2350.0, 3000.0, 3920.0])
    p = _max_speed_toy(P_avail, num_nodes=3)
    J = p.compute_totals(of=['V'], wrt=['P_avail'])
    slope_kn_per_hp = np.diag(J['V', 'P_avail']) / KT
    assert np.all(slope_kn_per_hp > 0.0)
    assert 0.002 < slope_kn_per_hp[-1] < 0.05

    step = 5.0
    up = _max_speed_toy(P_avail + step, num_nodes=3).get_val('V')
    down = _max_speed_toy(P_avail - step, num_nodes=3).get_val('V')
    assert_near_equal(np.diag(J['V', 'P_avail']), (up - down) / (2 * step), 1e-5)


# ------------------------------------------------------------------------
# G7 -- EquivalentRotorLDComp, p. 322, Figure 4.40
# ------------------------------------------------------------------------

# Figure 4.40 p. 323, 20,000 lb sea level: speed kt -> equivalent L/D
FIG_4_40 = {60: 2.7, 80: 4.4, 100: 6.05, 120: 5.3, 140: 4.0}


def _equivalent_ld(V_kt, f_hub_mast=7.0):
    """The Chapter 3 trim of this chapter, charged as a wing."""
    ff = _ff_power(V_kt)
    p = om.Problem()
    p.model.add_subsystem('ld', EquivalentRotorLDComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name in ('T', 'alpha_TPP', 'hp_M', 'q'):
        p.set_val(name, ff.get_val(name))
    p.set_val('V', V_kt * KT)
    p.set_val('f_rest', ff.get_val('f')[0] - f_hub_mast)
    p.run_model()
    return p


def test_equivalent_rotor_ld_definition_p322():
    """L_e is the vertical component of the thrust, D_e the power turned into drag less
    the parasite drag of everything but the hub and mast."""
    p = om.Problem()
    p.model.add_subsystem('ld', EquivalentRotorLDComp(num_nodes=2), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in dict(T=[20000.0, 20500.0], alpha_TPP=np.deg2rad([-2.0, -4.0]),
                          hp_M=[1000.0, 1400.0], V=[135.0, 200.0], q=[21.7, 47.6],
                          f_rest=13.0).items():
        p.set_val(name, val)
    p.run_model()
    assert_near_equal(p.get_val('L_e')[0], 20000.0 * np.cos(np.deg2rad(-2.0)), 1e-12)
    assert_near_equal(p.get_val('D_e')[0], 550 * 1000.0 / 135.0 - 21.7 * 13.0, 1e-12)
    assert_near_equal(p.get_val('L_D_e'), p.get_val('L_e') / p.get_val('D_e'), 1e-12)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


@pytest.mark.parametrize('V_kt', [60, 80, 100])
def test_equivalent_rotor_ld_above_figure_4_40_c4_29(V_kt):
    """Figure 4.40 peaks at 6.0 near 100 kt, but it is built on the power required of
    Figure 4.38, which stands 37 % above the book's own charts (C4-29). A power that high
    makes the rotor look worse, so the figure sits below this chain."""
    ratio = _equivalent_ld(V_kt).get_val('L_D_e')[0]
    assert ratio > FIG_4_40[V_kt]
    assert ratio < 2.0 * FIG_4_40[V_kt]


# ------------------------------------------------------------------------
# G7 -- SpecificRangeComp, p. 323
# ------------------------------------------------------------------------

def _specific_range(V_kt, FF, V_wind=0.0):
    V_kt, FF = np.atleast_1d(V_kt), np.atleast_1d(FF)
    p = om.Problem()
    p.model.add_subsystem('sr', SpecificRangeComp(num_nodes=V_kt.size), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in dict(V=V_kt, FF=FF, V_wind=V_wind).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_specific_range_definition_p323():
    """S.R. = ground speed / fuel flow, in n.mi. per lb; a headwind cuts it."""
    p = _specific_range(120.0, 600.0)
    assert_near_equal(p.get_val('SR', units='NM/lbm')[0], 0.2, 1e-12)
    head = _specific_range(120.0, 600.0, V_wind=40.0)
    tail = _specific_range(120.0, 600.0, V_wind=-40.0)
    assert_near_equal(head.get_val('SR')[0], 80.0 / 600.0, 1e-12)
    assert_near_equal(tail.get_val('SR')[0], 160.0 / 600.0, 1e-12)
    assert_check_partials(head.check_partials(method='cs', out_stream=None))


def test_specific_range_best_speed_moves_with_wind_p323():
    """p. 323: on the example the best speed goes from 114 kt in still air to 127 kt into a
    40 kt headwind and 104 kt with a 40 kt tailwind. On any bucket-shaped fuel flow curve
    the tangent from the origin moves the same way."""
    V = np.linspace(60.0, 170.0, 111)
    FF = 420.0 + 0.00060 * (V - 80.0) ** 3                    # rises steeply at high speed
    best = [V[np.argmax(_specific_range(V, FF, V_wind=w).get_val('SR'))]
            for w in (40.0, 0.0, -40.0)]
    assert best[0] > best[1] > best[2]


# ------------------------------------------------------------------------
# G7 -- BestRangeSpeedBalance, pp. 323-325
# ------------------------------------------------------------------------

# a bucket-shaped fuel flow curve and its analytic slope, lb/hr against knots
FF_A, FF_B, FF_V0 = 420.0, 6.0e-4, 80.0


def _best_range_speed(V_wind=0.0, guess=110.0):
    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', BestRangeSpeedBalance(guess=guess), promotes=['*'])
    model.add_subsystem('fuel', om.ExecComp(
        ['FF = FF_A + FF_B * (V - FF_V0) ** 3',
         'dFF_dV = 3 * FF_B * (V - FF_V0) ** 2'],
        FF={'units': 'lbm/h'}, dFF_dV={'units': 'lbm/h/kn'}, V={'units': 'kn'},
        FF_A=FF_A, FF_B=FF_B, FF_V0=FF_V0), promotes=['*'])
    model.add_subsystem('tangency', TangencyProductComp(), promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40, iprint=0)
    model.linear_solver = om.DirectSolver()
    p.setup(force_alloc_complex=True)
    p.set_val('V_wind', V_wind)
    p.run_model()
    return p


def test_best_range_speed_is_the_tangent_from_the_origin():
    """The balance lands on the maximum of the specific range itself."""
    p = _best_range_speed()
    V_best = p.get_val('V', units='kn')[0]
    assert_near_equal(p.get_val('FF', units='lbm/h')[0],
                      V_best * p.get_val('dFF_dV')[0], 1e-8)

    V = np.linspace(60.0, 170.0, 2201)
    FF = FF_A + FF_B * (V - FF_V0) ** 3
    assert abs(V[np.argmax(V / FF)] - V_best) < 0.1


def test_best_range_speed_moves_with_wind_p323():
    """p. 323: 114 kt in still air, 127 kt into a 40 kt headwind, 104 kt with 40 kt behind.
    The tangent starts from V = V_wind, so the speed moves the same way here."""
    still = _best_range_speed().get_val('V', units='kn')[0]
    head = _best_range_speed(V_wind=40.0, guess=130.0).get_val('V', units='kn')[0]
    tail = _best_range_speed(V_wind=-40.0, guess=100.0).get_val('V', units='kn')[0]
    assert head > still > tail
    assert abs((head - still) / (still - tail) - 1.0) < 0.6      # roughly symmetric

    V = np.linspace(60.0, 170.0, 2201)
    FF = FF_A + FF_B * (V - FF_V0) ** 3
    for V_wind, V_best in ((40.0, head), (-40.0, tail)):
        assert abs(V[np.argmax((V - V_wind) / FF)] - V_best) < 0.2


# ------------------------------------------------------------------------
# G7 -- FuelFlowSlopeComp, pp. 323-325
# ------------------------------------------------------------------------

def _fuel_sub_problem():
    """A stand-in cruise chain: fuel flow against speed and gross weight."""
    p = om.Problem()
    p.model.add_subsystem('fuel', om.ExecComp(
        'FF = FF_A * GW / 20000 + FF_B * (V / KT - FF_V0) ** 3',
        FF={'units': 'lbm/h'}, V={'units': 'ft/s'}, GW={'units': 'lbf', 'val': 20000.0},
        FF_A=FF_A, FF_B=FF_B, FF_V0=FF_V0, KT=KT), promotes=['*'])
    return p


def test_fuel_flow_slope_is_the_analytic_derivative():
    """The slope comes from compute_totals through the sub-problem, not from a difference."""
    p = om.Problem()
    p.model.add_subsystem('slope', FuelFlowSlopeComp(problem=_fuel_sub_problem(),
                                                     passthrough={'GW': 'lbf'}), promotes=['*'])
    p.setup()
    for V_kt in (100.0, 120.0, 140.0):
        p.set_val('V', V_kt)
        p.run_model()
        assert_near_equal(p.get_val('FF', units='lbm/h')[0],
                          FF_A + FF_B * (V_kt - FF_V0) ** 3, 1e-5)    # knots -> ft/s round trip
        assert_near_equal(p.get_val('dFF_dV')[0], 3 * FF_B * (V_kt - FF_V0) ** 2, 1e-5)

    p.set_val('GW', 24000.0)                                  # pass-through reaches the sub-model
    p.run_model()
    assert p.get_val('FF')[0] > FF_A + FF_B * (140.0 - FF_V0) ** 3


def test_fuel_flow_slope_drives_the_best_range_balance():
    """Slope component and tangency balance together find the maximum specific range."""
    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', BestRangeSpeedBalance(), promotes=['*'])
    model.add_subsystem('slope', FuelFlowSlopeComp(problem=_fuel_sub_problem()), promotes=['*'])
    model.add_subsystem('tangency', TangencyProductComp(), promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40, iprint=0)
    model.linear_solver = om.DirectSolver()
    p.setup()
    p.run_model()

    V_best = p.get_val('V', units='kn')[0]
    V = np.linspace(60.0, 170.0, 2201)
    FF = FF_A + FF_B * (V - FF_V0) ** 3
    assert abs(V[np.argmax(V / FF)] - V_best) < 0.2


# ------------------------------------------------------------------------
# G7 -- CruiseSpeedBalance, p. 325
# ------------------------------------------------------------------------

def _cruise_speed(fraction=0.99, V_wind=0.0):
    """Peak from the tangency balance, then the 99 % speed past it, on the toy fuel curve."""
    best = _best_range_speed(V_wind=V_wind,
                             guess=110.0 + 0.5 * V_wind)
    V_best = best.get_val('V', units='kn')[0]
    SR_max = (V_best - V_wind) / best.get_val('FF', units='lbm/h')[0]

    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', CruiseSpeedBalance(), promotes=['*'])
    model.add_subsystem('speed', CruiseSpeedComp(), promotes=['*'])
    model.add_subsystem('fuel', om.ExecComp(
        'FF = FF_A + FF_B * (V_cruise - FF_V0) ** 3', FF={'units': 'lbm/h'},
        V_cruise={'units': 'kn'}, FF_A=FF_A, FF_B=FF_B, FF_V0=FF_V0), promotes=['*'])
    model.add_subsystem('sr', om.ExecComp(
        'SR = (V_cruise - V_wind) / FF', SR={'units': 'NM/lbm'}, V_cruise={'units': 'kn'},
        V_wind={'units': 'kn'}, FF={'units': 'lbm/h'}), promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40, iprint=0)
    model.linear_solver = om.DirectSolver()
    p.setup(force_alloc_complex=True)
    for name, val in dict(V_best=V_best, SR_max=SR_max, fraction=fraction,
                          V_wind=V_wind).items():
        p.set_val(name, val)
    p.run_model()
    return p, V_best, SR_max


def test_cruise_speed_is_past_the_peak_p325():
    """p. 325: the speed used is the one to the right of the peak where the specific range
    is 99 % of its maximum."""
    p, V_best, SR_max = _cruise_speed()
    V_cruise = p.get_val('V_cruise', units='kn')[0]
    assert V_cruise > V_best
    assert_near_equal(p.get_val('SR', units='NM/lbm')[0], 0.99 * SR_max, 1e-8)
    assert 0.0 < V_cruise - V_best < 20.0                     # a few knots, little economy lost


def test_cruise_speed_fraction_and_wind():
    """A lower fraction buys more speed, and a headwind pushes the whole pair up."""
    slower = _cruise_speed(fraction=0.995)[0].get_val('V_cruise', units='kn')[0]
    faster = _cruise_speed(fraction=0.95)[0].get_val('V_cruise', units='kn')[0]
    assert faster > slower
    head, V_best_head, _ = _cruise_speed(V_wind=40.0)
    assert head.get_val('V_cruise', units='kn')[0] > V_best_head
    assert V_best_head > _cruise_speed()[1]


# ------------------------------------------------------------------------
# G7 -- SpecificEnduranceComp and BestEnduranceSpeedBalance, pp. 330-331
# ------------------------------------------------------------------------

def test_specific_endurance_definition_p330():
    """S.E. = 1 / fuel flow, in hr/lb; Figure 4.47 runs from 0.0022 to 0.0012 hr/lb."""
    p = om.Problem()
    p.model.add_subsystem('se', SpecificEnduranceComp(num_nodes=2), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('FF', [455.0, 833.0])
    p.run_model()
    assert_near_equal(p.get_val('SE', units='h/lbm'), 1.0 / np.array([455.0, 833.0]), 1e-12)
    assert abs(p.get_val('SE')[0] - 0.0022) < 0.0001
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def _loiter_sub_problem():
    """A fuel flow curve with a real bottom, unlike the cubic used for range."""
    p = om.Problem()
    p.model.add_subsystem('fuel', om.ExecComp(
        'FF = 420 + 0.02 * (V / KT - 70) ** 2', FF={'units': 'lbm/h'},
        V={'units': 'ft/s'}, KT=KT), promotes=['*'])
    return p


def test_best_endurance_speed_is_the_bottom_of_the_fuel_curve_p330():
    """The loiter speed is where dFF/dV = 0, and unlike the best range speed it does not
    move with wind: the wind changes the distance covered, not the fuel burnt per hour."""
    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', BestEnduranceSpeedBalance(), promotes=['*'])
    model.add_subsystem('slope', FuelFlowSlopeComp(problem=_loiter_sub_problem()),
                        promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40, iprint=0)
    model.linear_solver = om.DirectSolver()
    p.setup()
    p.run_model()

    V_loiter = p.get_val('V', units='kn')[0]
    V = np.linspace(40.0, 160.0, 2401)
    assert abs(V[np.argmin(420 + 0.02 * (V - 70) ** 2)] - V_loiter) < 0.1
    assert V_loiter < _best_range_speed().get_val('V', units='kn')[0]


# ------------------------------------------------------------------------
# G7 -- MissionIntegralComp, pp. 325-331
# ------------------------------------------------------------------------

def _integral(GW, y, kind='range', rule='trapezoid'):
    GW, y = np.asarray(GW, dtype=float), np.asarray(y, dtype=float)
    name = 'SR' if kind == 'range' else 'SE'
    p = om.Problem()
    p.model.add_subsystem('int', MissionIntegralComp(num_nodes=GW.size, kind=kind, rule=rule),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('GW', GW)
    p.set_val(name, y)
    p.run_model()
    return p


def test_mission_integral_constant_specific_range():
    """Burning 4,000 lb at a constant 0.114 n.mi./lb covers 456 n.mi. (p. 325)."""
    GW = np.linspace(16000.0, 20000.0, 5)
    p = _integral(GW, np.full(5, 0.114))
    assert_near_equal(p.get_val('range', units='NM')[0], 0.114 * 4000.0, 1e-12)


def test_mission_integral_rules_and_partials():
    """Simpson is exact on a quadratic specific range curve, the trapezoid is not, and the
    derivatives with respect to the weights are carried."""
    GW = np.linspace(14000.0, 22000.0, 9)
    SR = 0.05 + 1.2e-5 * (22000.0 - GW) - 3.0e-10 * (22000.0 - GW) ** 2
    exact = (0.05 * 8000.0 + 1.2e-5 * 8000.0 ** 2 / 2 - 3.0e-10 * 8000.0 ** 3 / 3)

    simpson = _integral(GW, SR, rule='simpson')
    assert_near_equal(simpson.get_val('range')[0], exact, 1e-10)
    trapezoid = _integral(GW, SR)
    assert abs(trapezoid.get_val('range')[0] - exact) > 1e-8
    assert abs(trapezoid.get_val('range')[0] / exact - 1.0) < 1e-3

    assert_check_partials(simpson.check_partials(method='cs', out_stream=None))
    assert_check_partials(trapezoid.check_partials(method='cs', out_stream=None))


def test_mission_integral_endurance():
    """The same integral in hours, p. 331: 4,000 lb at 0.0015 hr/lb is 6 hours."""
    GW = np.linspace(16000.0, 20000.0, 3)
    p = _integral(GW, np.full(3, 0.0015), kind='endurance')
    assert_near_equal(p.get_val('endurance', units='h')[0], 6.0, 1e-12)
    with pytest.raises(ValueError, match='Simpson'):
        _integral(np.linspace(1.0, 2.0, 4), np.ones(4), rule='simpson')


# ------------------------------------------------------------------------
# G7 -- PayloadRangeComp, pp. 326-328, Figure 4.45
# ------------------------------------------------------------------------

def _payload_range(**vals):
    p = om.Problem()
    p.model.add_subsystem('pr', PayloadRangeComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in vals.items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_payload_range_weights_p326():
    """p. 326: GW_ldng = GW_TO - (expended + WUTO); payload = (GW_TO - GW_min_OP) minus
    the fuel and the auxiliary tank. p. 326 and p. 329: 20,000 lb takeoff, 11,261 lb
    minimum operating weight, 6,600 lb of payload."""
    p = _payload_range(GW_TO=20000.0, GW_min_OP=11261.0, fuel_expended=1900.0,
                       fuel_WUTO=56.0, fuel_reserves=183.0)
    assert_near_equal(p.get_val('GW_ldng', units='lbf')[0], 20000.0 - 1956.0, 1e-12)
    assert abs(p.get_val('payload', units='lbf')[0] - 6600.0) < 10.0
    assert_near_equal(p.get_val('fuel_total')[0], 2139.0, 1e-12)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def test_payload_range_trades_payload_for_fuel_p327():
    """p. 327: offloading payload and replacing it with fuel in auxiliary tanks extends the
    range; the tank itself costs payload too."""
    base = _payload_range(GW_TO=20000.0, GW_min_OP=11261.0, fuel_expended=1900.0,
                          fuel_WUTO=56.0, fuel_reserves=183.0)
    more_fuel = _payload_range(GW_TO=20000.0, GW_min_OP=11261.0, fuel_expended=2900.0,
                               fuel_WUTO=56.0, fuel_reserves=183.0, w_aux_tank=100.0)
    assert_near_equal(base.get_val('payload')[0] - more_fuel.get_val('payload')[0],
                      1100.0, 1e-12)
    assert more_fuel.get_val('GW_ldng')[0] < base.get_val('GW_ldng')[0]


# ------------------------------------------------------------------------
# G7 -- FerryReservesComp, pp. 328-330, Figure 4.46
# ------------------------------------------------------------------------

FERRY = dict(GW_min_OP=11261.0, fuel_total=15961.0, fuel_first_period=4400.0,
             fuel_WUTO=56.0, w_tanks=2 * 409.0)                # p. 329


def _ferry(**overrides):
    p = om.Problem()
    p.model.add_subsystem('ferry', FerryReservesComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in {**FERRY, **overrides}.items():
        p.set_val(name, val)
    p.set_val('FF_reserve', overrides.get('FF_reserve', 560.0))    # 420 lb in 45 min
    p.run_model()
    return p


def test_ferry_reserves_example_p329():
    """p. 329: reserve no. 1 = 420 lb, reserve no. 2 = (0.1/1.1)[15,961 - (4,400 + 56 + 420)]
    = 1,008 lb, landing weight 12,689 lb."""
    p = _ferry()
    assert_near_equal(p.get_val('reserve_1', units='lbf')[0], 420.0, 1e-12)
    assert abs(p.get_val('reserve_2', units='lbf')[0] - 1008.0) < 1.0
    assert abs(p.get_val('GW_ldng', units='lbf')[0] - 12689.0) < 1.0
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def test_ferry_start_weight_and_the_book_takeoff_weight_c4_32():
    """The start weight is the takeoff weight less the WUTO fuel. Adding the minimum
    operating weight, the two external tanks and all the usable fuel gives 28,040 lb, where
    the book states 27,944 (C4-32)."""
    p = _ferry()
    GW_TO = p.get_val('GW_TO', units='lbf')[0]
    assert_near_equal(GW_TO, 11261.0 + 818.0 + 15961.0, 1e-12)
    assert_near_equal(p.get_val('GW_start')[0], GW_TO - 56.0, 1e-12)
    assert abs(GW_TO - 27944.0) < 120.0


def test_ferry_reserve_fraction_and_time():
    """A longer reserve time or a larger percentage eats into the fuel available to cruise."""
    base = _ferry().get_val('reserves')[0]
    assert _ferry(t_reserve=1.0).get_val('reserves')[0] > base
    assert _ferry(reserve_fraction=0.15).get_val('reserves')[0] > base


# ------------------------------------------------------------------------
# G7 -- CruisePerformanceGroup, pp. 317-331
# ------------------------------------------------------------------------

CRUISE_INPUTS = {k: v for k, v in REF_ROTOR.items() if k not in ('rho', 'V_son')}
CRUISE_INPUTS.update(EXAMPLE_DESIGN, load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0,
                     n_eng=2.0, k_inst=0.02, altitude=0.0,
                     alpha_F=np.deg2rad(-6.0), CT_sigma=0.085)


def _cruise_point(V_kt, **overrides):
    p = om.Problem()
    p.model.add_subsystem('cruise', CruisePerformanceGroup(), promotes=['*'])
    p.setup()
    for name, val in {**CRUISE_INPUTS, **overrides}.items():
        p.set_val(name, val)
    p.set_val('V', V_kt, units='kn')
    p.run_model()
    return p


def test_cruise_group_specific_range_p323():
    """The whole chain at 20,000 lb and sea level: trim, compressibility, drive losses,
    engine fuel flow, specific range. Figure 4.42 reads 0.114 n.mi./lb at 114 kt; the chain
    reads higher because that figure follows the power of Figure 4.38, which stands 37 %
    above the book's own charts (C4-29)."""
    p = _cruise_point(114.0)
    SR = p.get_val('SR', units='NM/lbm')[0]
    assert 0.114 < SR < 0.14
    assert_near_equal(SR, 114.0 / p.get_val('FF', units='lbm/h')[0], 1e-10)
    assert_near_equal(p.get_val('SE', units='h/lbm')[0],
                      1.0 / p.get_val('FF', units='lbm/h')[0], 1e-12)


def test_cruise_group_wind_moves_the_specific_range():
    """Only the ground speed changes with wind, not the fuel flow."""
    still = _cruise_point(114.0)
    head = _cruise_point(114.0, V_wind=40.0)
    assert_near_equal(head.get_val('FF'), still.get_val('FF'), 1e-10)
    assert_near_equal(head.get_val('SR')[0],
                      74.0 / head.get_val('FF', units='lbm/h')[0], 1e-10)


def test_cruise_group_best_range_speed_with_the_real_chain_c4_29():
    """Tangency balance driven by FuelFlowSlopeComp on a sub-problem of this same group.
    Figure 4.42 gives 114 kt and 0.114 n.mi./lb; the chain finds about 148 kt and 0.135.
    A power curve rising too steeply with speed, as Figure 4.38 does against the charts,
    pulls the tangency back towards low speed, which is why the book's speed is lower
    (C4-29)."""
    from prouty.performance import cruise_problem
    sub = cruise_problem(inputs=CRUISE_INPUTS)

    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', BestRangeSpeedBalance(V_bounds=(60.0, 165.0), guess=130.0),
                        promotes=['*'])
    model.add_subsystem('slope', FuelFlowSlopeComp(
        problem=sub, speed_units='kn', passthrough={'GW': 'lbf'},
        reset={'alpha_F': np.deg2rad(-6.0), 'CT_sigma': 0.085}), promotes=['*'])
    model.add_subsystem('tangency', TangencyProductComp(), promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=25, iprint=0)
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
        bound_enforcement='vector', maxiter=6, iprint=0, retry_on_analysis_error=True)
    model.linear_solver = om.DirectSolver()
    p.setup()
    p.set_val('GW', 20000.0)
    p.run_model()

    V_best = p.get_val('V', units='kn')[0]
    assert_near_equal(p.get_val('FF', units='lbm/h')[0],
                      V_best * p.get_val('dFF_dV')[0], 1e-6)         # tangency closed
    assert 114.0 < V_best < 165.0


# ------------------------------------------------------------------------
# G8 -- ClimbFlatPlateComp, p. 333
# ------------------------------------------------------------------------

def _climb_flat_plate(V_kt, fpm, f=20.0, GW=20000.0, convention='horizontal'):
    V = V_kt * KT
    q = 0.5 * 0.002377 * V ** 2
    p = om.Problem()
    p.model.add_subsystem('fc', ClimbFlatPlateComp(angle_convention=convention), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in dict(f=f, GW=GW, q=q, V=V, V_c=fpm / 60.0).items():
        p.set_val(name, val)
    p.run_model()
    return p, q


def test_climb_flat_plate_p333():
    """f_climb = f + GW sin(gamma)/q, so a climb is a drag increment in the level chain."""
    p, q = _climb_flat_plate(100.0, 1500.0)
    sin_gamma = p.get_val('sin_gamma')[0]
    assert_near_equal(p.get_val('f_climb', units='ft**2')[0],
                      20.0 + 20000.0 * sin_gamma / q, 1e-12)
    assert p.get_val('f_climb')[0] > 20.0
    assert_near_equal(_climb_flat_plate(100.0, 0.0)[0].get_val('f_climb')[0], 20.0, 1e-12)
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def test_climb_angle_convention_c4_34():
    """p. 333: one engine, 1,100 ft/min at 48 kt, 'which gives a climb angle of 12.7
    degrees' — that is the arc tangent, not the arc sine."""
    horizontal = _climb_flat_plate(48.0, 1100.0)[0].get_val('gamma', units='deg')[0]
    path = _climb_flat_plate(48.0, 1100.0, convention='path')[0].get_val('gamma',
                                                                        units='deg')[0]
    assert abs(horizontal - 12.7) < 0.1
    assert path > horizontal
    assert abs(path - 13.1) < 0.1


# ------------------------------------------------------------------------
# G8 -- ForwardClimbBalance, pp. 332-335, Figure 4.48
# ------------------------------------------------------------------------

def _forward_climb(P_avail, V_kt=60.0, f=20.0, GW=20000.0, num_nodes=1):
    """Level power in the shape of Figure 4.48, plus the climb drag of p. 333."""
    P_avail = np.atleast_1d(P_avail)
    q = 0.5 * 0.002377 * (V_kt * KT) ** 2
    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', ForwardClimbBalance(num_nodes=num_nodes), promotes=['*'])
    model.add_subsystem('flat_plate', ClimbFlatPlateComp(num_nodes=num_nodes), promotes=['*'])
    model.add_subsystem('power', om.ExecComp(
        'P_req = 1050 + 26.0 * (f_climb - 20.0)',
        P_req={'units': 'hp', 'shape': num_nodes},
        f_climb={'units': 'ft**2', 'shape': num_nodes}), promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40, iprint=0)
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector',
                                                             maxiter=6, iprint=0)
    model.linear_solver = om.DirectSolver()
    p.setup(force_alloc_complex=True)
    for name, val in dict(P_avail=P_avail, f=f, GW=GW, q=q, V=V_kt * KT).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_forward_climb_balance_closes_on_the_power_available():
    """The rate of climb comes out where the climb power meets the power available."""
    p = _forward_climb(3000.0)
    assert_near_equal(p.get_val('P_req', units='hp'), p.get_val('P_avail', units='hp'), 1e-8)
    fpm = p.get_val('V_c', units='ft/min')[0]
    assert 100.0 < fpm < 5000.0
    assert _forward_climb(3500.0).get_val('V_c')[0] > p.get_val('V_c')[0]
    assert_near_equal(_forward_climb(1050.0).get_val('V_c')[0], 0.0, 1e-6)   # level flight


def test_forward_climb_balance_descends_above_the_ceiling_p334():
    """Less power than level flight needs gives a negative rate, which is how the ceilings
    of Figure 4.51 are found: absolute where the rate reaches zero, service at 100 ft/min."""
    p = _forward_climb(900.0)
    assert p.get_val('V_c', units='ft/min')[0] < 0.0

    P_avail = np.array([1050.0, 2000.0, 3000.0])
    vec = _forward_climb(P_avail, num_nodes=3)
    assert np.all(np.diff(vec.get_val('V_c')) > 0.0)
    J = vec.compute_totals(of=['V_c'], wrt=['P_avail'])
    step = 10.0
    up = _forward_climb(P_avail + step, num_nodes=3).get_val('V_c')
    down = _forward_climb(P_avail - step, num_nodes=3).get_val('V_c')
    assert_near_equal(np.diag(J['V_c', 'P_avail']), (up - down) / (2 * step), 1e-5)


# ------------------------------------------------------------------------
# G8 -- ClimbTimeDistanceComp, p. 334, Figure 4.50
# ------------------------------------------------------------------------

def _climb_time(h, V_c, V):
    h, V_c, V = np.asarray(h, float), np.asarray(V_c, float), np.asarray(V, float)
    p = om.Problem()
    p.model.add_subsystem('td', ClimbTimeDistanceComp(num_nodes=h.size), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in dict(altitude=h, V_c=V_c, V=V).items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_climb_time_and_distance_constant_rate():
    """At a constant 2,000 ft/min and 100 kt, 20,000 ft takes 10 min and 16.7 n.mi."""
    h = np.linspace(0.0, 20000.0, 21)
    p = _climb_time(h, np.full(21, 2000.0), np.full(21, 100.0))
    assert_near_equal(p.get_val('time_total', units='min')[0], 10.0, 1e-12)
    assert_near_equal(p.get_val('distance_total', units='NM')[0], 100.0 * 10.0 / 60.0, 1e-12)
    assert_near_equal(p.get_val('time', units='min')[10], 5.0, 1e-12)      # cumulative
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


def test_climb_time_with_a_decaying_rate_fig_4_50():
    """Figure 4.50, intermediate power, 20,000 lb: about 10 min and 14 n.mi. to 20,000 ft.
    With the rate falling linearly from 2,900 to 900 ft/min it comes out close."""
    h = np.linspace(0.0, 20000.0, 21)
    V_c = np.linspace(2900.0, 900.0, 21)
    p = _climb_time(h, V_c, np.full(21, 72.0))
    assert abs(p.get_val('time_total', units='min')[0] - 10.0) < 2.0
    assert abs(p.get_val('distance_total', units='NM')[0] - 14.0) < 3.0
    assert np.all(np.diff(p.get_val('time')) > 0.0)


# ------------------------------------------------------------------------
# G8 -- ClimbCeilingBalance, p. 335, Figures 4.49 and 4.51
# ------------------------------------------------------------------------

def _climb_ceiling(GW, ceiling='absolute', num_nodes=1):
    """Maximum rate of climb falling with altitude, in the shape of Figure 4.49."""
    GW = np.atleast_1d(GW)
    p = om.Problem()
    model = p.model
    model.add_subsystem('balance', ClimbCeilingBalance(num_nodes=num_nodes, ceiling=ceiling),
                        promotes=['*'])
    model.add_subsystem('atm', AtmosphereGroup(num_nodes=num_nodes), promotes=['*'])
    model.add_subsystem('climb', om.ExecComp(
        'V_c = 3000 * density_ratio - 0.09 * GW',
        V_c={'units': 'ft/min', 'shape': num_nodes}, density_ratio={'shape': num_nodes},
        GW={'units': 'lbf', 'shape': num_nodes}), promotes=['*'])
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40, iprint=0)
    model.linear_solver = om.DirectSolver()
    p.setup(force_alloc_complex=True)
    p.set_val('GW', GW)
    p.run_model()
    return p


def test_climb_ceiling_definitions_p335():
    """p. 335: absolute ceiling where the rate of climb is zero, service ceiling at
    100 ft/min, so the service ceiling is the lower of the two."""
    absolute = _climb_ceiling(20000.0)
    service = _climb_ceiling(20000.0, ceiling='service')
    assert_near_equal(absolute.get_val('V_c', units='ft/min')[0], 0.0, 1e-6)
    assert_near_equal(service.get_val('V_c', units='ft/min')[0], 100.0, 1e-6)
    assert service.get_val('altitude')[0] < absolute.get_val('altitude')[0]
    with pytest.raises(ValueError, match='ceiling'):
        ClimbCeilingBalance(ceiling='cruise')


def test_climb_ceiling_falls_with_gross_weight_fig_4_51():
    """Figure 4.51 plots both ceilings against gross weight; heavier is lower."""
    GW = np.array([16000.0, 20000.0, 24000.0])
    p = _climb_ceiling(GW, num_nodes=3)
    h = p.get_val('altitude', units='ft')
    assert np.all(np.diff(h) < 0.0)

    J = p.compute_totals(of=['altitude'], wrt=['GW'])
    step = 100.0
    up = _climb_ceiling(GW + step, num_nodes=3).get_val('altitude')
    down = _climb_ceiling(GW - step, num_nodes=3).get_val('altitude')
    assert_near_equal(np.diag(J['altitude', 'GW']), (up - down) / (2 * step), 1e-4)


# ------------------------------------------------------------------------
# G9 -- MilitaryMissionGroup, pp. 336-337, Table 4.4
# ------------------------------------------------------------------------

# Table 4.4 in flight order: 1 WUTO, 2 climb, 3 cruise, 4 loiter, 5 dash,
# 6 hover OGE (payload dropped), 7 WUTO, 8 climb, 9 cruise
TABLE_4_4_MODES = ('time', 'time', 'time', 'endurance', 'time', 'time', 'time', 'time',
                   'distance')
TABLE_4_4 = dict(
    FF=[1680.0, 1900.0, 1570.0, 0.0, 2080.0, 1180.0, 1680.0, 1800.0, 0.0],
    time=[0.033, 0.028, 0.667, 0.333, 0.060, 0.167, 0.033, 0.033, 0.0],
    distance=[0.0] * 8 + [96.0],
    SR=[1.0] * 8 + [0.188],
    SE=[1.0] * 3 + [0.001218] + [1.0] * 5,
    payload_drop=[0.0] * 5 + [6600.0] + [0.0] * 3,
    GW_ldng=10430.0 + 300.0,                       # minimum operating weight plus the estimate
    GW_min_OP=10430.0, payload=6600.0, reserve_fraction=0.10)
TABLE_4_4_FUEL = [56.0, 53.0, 1047.0, 275.0, 125.0, 194.0, 56.0, 59.0, 510.0]
TABLE_4_4_GW_END = [19649.0, 19596.0, 18549.0, 18274.0, 18149.0, 17955.0, 11299.0, 11240.0,
                    10730.0]


def _mission():
    p = om.Problem()
    p.model.add_subsystem('mission', MilitaryMissionGroup(modes=TABLE_4_4_MODES),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for name, val in TABLE_4_4.items():
        p.set_val(name, val)
    p.run_model()
    return p


def test_military_mission_segment_fuel_table_4_4():
    """The three ways a segment burns fuel: flow times time, distance over specific range,
    time over specific endurance."""
    fuel = _mission().get_val('fuel', units='lbf')
    assert np.all(np.abs(fuel - TABLE_4_4_FUEL) < 4.0)


def test_military_mission_backward_sweep_table_4_4():
    """p. 336: the mission is analysed in reverse from the landing weight, and the payload
    dropped at the hover reappears as a step in the sweep."""
    p = _mission()
    GW_end = p.get_val('GW_end', units='lbf')
    assert np.all(np.abs(GW_end - TABLE_4_4_GW_END) < 15.0)
    assert GW_end[5] - GW_end[6] > 6600.0                     # the payload step
    assert abs(p.get_val('GW_TO_est', units='lbf')[0] - 19705.0) < 20.0


def test_military_mission_summary_p337():
    """p. 337: mission fuel 2,375 lb, total fuel 2,375/0.9 = 2,639 lb, reserve 264 lb and
    T.O.G.W. = 10,430 + 6,600 + 2,639 = 19,669 lb. The swept takeoff weight is 36 lb higher,
    'due to initial reserve estimate'."""
    p = _mission()
    assert abs(p.get_val('fuel_mission', units='lbf')[0] - 2375.0) < 15.0
    assert abs(p.get_val('fuel_total', units='lbf')[0] - 2639.0) < 20.0
    assert abs(p.get_val('reserve', units='lbf')[0] - 264.0) < 5.0
    assert abs(p.get_val('GW_TO', units='lbf')[0] - 19669.0) < 20.0
    gap = p.get_val('GW_TO_est')[0] - p.get_val('GW_TO')[0]
    assert abs(gap - (300.0 - p.get_val('reserve')[0])) < 5.0
    assert_check_partials(p.check_partials(method='cs', out_stream=None))


# ------------------------------------------------------------------------
# G8 -- ForwardClimbGroup, pp. 332-335 (unblocked by the Chapter 3 trim fix)
# ------------------------------------------------------------------------

def _forward_climb_group(V_kt, rating='max_continuous'):
    p = om.Problem()
    p.model.add_subsystem('g8', ForwardClimbGroup(rating=rating), promotes=['*'])
    p.setup()
    for name, val in CRUISE_INPUTS.items():
        p.set_val(name, val)
    p.set_val('V', V_kt, units='kn')
    p.run_model()
    return p


def test_forward_climb_group_closes_cold_c4_35():
    """The group used to inherit the false root of the climb trim; with subsystem solves on
    it converges from a cold start in under a second, and the power balance closes."""
    p = _forward_climb_group(80.0)
    assert_near_equal(p.get_val('P_req', units='hp'), p.get_val('P_rating', units='hp'), 1e-6)
    assert 1000.0 < p.get_val('R_C', units='ft/min')[0] < 5000.0
    assert 5.0 < p.get_val('gamma', units='deg')[0] < 40.0


def test_forward_climb_group_more_power_climbs_faster():
    """Takeoff power against maximum continuous, at the same speed (Figure 4.48). The rates
    come out above the book because the power required is low at speed (C4-29)."""
    continuous = _forward_climb_group(80.0).get_val('R_C', units='ft/min')[0]
    takeoff = _forward_climb_group(80.0, rating='takeoff').get_val('R_C', units='ft/min')[0]
    assert takeoff > continuous
    assert continuous > 2500.0                                # Figure 4.48 peaks at 2,650


def _ff_power_book(V_kt):
    """Chain plus the C5-6 stall increment, charts read as the book reads them
    (no p. 230 twist displacement of the stall limits)."""
    p = om.Problem()
    p.model.add_subsystem('ff', ForwardFlightPowerGroup(
        stall=True, stall_options={'twist_shift': 'book'}), promotes=['*'])
    p.setup()
    for name, val in {**REF_ROTOR, **EXAMPLE_DESIGN,
                      **dict(load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0)}.items():
        p.set_val(name, val)
    p.set_val('V', V_kt * KT)
    p.set_val('alpha_F', np.deg2rad(-6.0))
    p.set_val('CT_sigma', 0.085)
    p.run_model()
    return p


@pytest.mark.parametrize('V_kt', [60, 80, 100, 120, 140])
def test_c4_29_book_reading_reproduces_figure_4_48(V_kt):
    """C4-29 explained: with the stall torque read on the Chapter 3 charts as they are
    (theta_1 = -5 deg, no p. 230 shift; C5-6), the chain meets Figure 4.48 within 5 %
    from 60 to 140 kt (1,120/1,059/1,152/1,522/2,244 hp against 1,131/1,059/1,189/
    1,577/2,337). At 160 kt (mu = 0.415, beyond the last plate) it is 13 % low."""
    P = _ff_power_book(V_kt).get_val('P_req', units='hp')[0]
    assert_near_equal(P, FIG_4_48_LEVEL[V_kt], 0.05)
