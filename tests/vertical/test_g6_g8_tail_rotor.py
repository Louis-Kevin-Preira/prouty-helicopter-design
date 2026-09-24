"""Chapter 2, G6 -- TailRotorVortexRingGroup, pp. 107-109; G8 -- TailRotorDriveTorqueComp, p. 116."""

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.vertical import ClimbPowerGroup, TailRotorDriveTorqueComp, TailRotorVortexRingGroup


def _run_g6(V_y_T, r_yaw=0.0, v_hov_T=48.0, l_T=37.0, **opts):
    V = np.atleast_1d(np.asarray(V_y_T, dtype=float))
    nn = V.size
    p = om.Problem()
    p.model.add_subsystem('g6', TailRotorVortexRingGroup(num_nodes=nn, **opts), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('V_y_T', V, units='ft/s')
    p.set_val('r_yaw', np.broadcast_to(r_yaw, (nn,)).astype(float), units='rad/s')
    p.set_val('v_hov_T', np.full(nn, v_hov_T), units='ft/s')
    p.set_val('l_T', l_T, units='ft')
    p.run_model()
    return p


def test_uh1_maximum_vortex_ring_at_20_kt_p108():
    """v_1hov_T = 48 ft/s: 70 % of it is about 20 kt."""
    p = _run_g6(0.0)
    assert_near_equal(p.get_val('V_D_max_instability_T', units='kn')[0], 20.0, 0.01)


def test_unstable_band_p108():
    """Figure 2.7 is unstable between velocity ratios 0.4 and 0.8."""
    p = _run_g6([0.4 * 48.0, 0.6 * 48.0, 0.8 * 48.0, 60.0])
    assert_near_equal(p.get_val('vrs_margin_T')[:3], [0.0, 1.0, 0.0], 1e-12)
    assert p.get_val('vrs_margin_T')[3] < 0.0


def test_hover_turn_is_equivalent_to_sideward_flight():
    """A yaw rate r moves the tail rotor at r l_T (turn over a spot, p. 107)."""
    turn, side = _run_g6(0.0, r_yaw=0.5), _run_g6(0.5 * 37.0)
    assert_near_equal(turn.get_val('V_D_bar_T'), side.get_val('V_D_bar_T'), 1e-14)


def test_linked_to_the_g2_tail_rotor():
    """v_hov_T comes from G2 (tail_rotor='hover_power') by promotion."""
    p = om.Problem()
    p.model.add_subsystem('g2', ClimbPowerGroup(tail_rotor='hover_power'), promotes=['*'])
    p.model.add_subsystem('g6', TailRotorVortexRingGroup(), promotes=['*'])
    p.setup()
    p.set_val('P_MR', 1600.0, units='hp')
    p.set_val('R', 30.0)
    p.set_val('l_T', 36.8)
    p.set_val('tr_R', 6.5)
    p.set_val('V_y_T', 20.0)
    p.run_model()
    v_hov_T = p.get_val('v_hov_T')[0]
    assert 35.0 < v_hov_T < 50.0
    assert_near_equal(p.get_val('V_D_bar_T')[0], 20.0 / v_hov_T, 1e-12)


def test_drive_design_torque_p116():
    p = om.Problem()
    p.model.add_subsystem('g8', TailRotorDriveTorqueComp(num_nodes=2), promotes=['*'])
    p.model.add_subsystem('g8b', TailRotorDriveTorqueComp(k_design=3.0),
                          promotes_inputs=[('Q_T_hov_max', 'Q_b')])
    p.setup(force_alloc_complex=True)
    p.set_val('Q_T_hov_max', [1000.0, 1500.0])
    p.set_val('Q_b', 1000.0)
    p.run_model()
    assert_near_equal(p.get_val('Q_T_design'), [2000.0, 3000.0], 1e-14)
    assert_near_equal(p.get_val('g8b.Q_T_design'), [3000.0], 1e-14)
    assert_check_partials(p.check_partials(method='cs', out_stream=None), atol=1e-12, rtol=1e-12)


def test_partials_g6():
    p = _run_g6([5.0, 25.0, 40.0], r_yaw=[0.1, -0.2, 0.3])
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-12, rtol=1e-12)
