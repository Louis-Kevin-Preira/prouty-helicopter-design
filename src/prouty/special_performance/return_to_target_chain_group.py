"""
ReturnToTargetChainGroup -- G6, return-to-target maneuver from the rotor models.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Return-to-Target Maneuver" pp. 368-371, Figure 5.17.

    decel     TurnDecelerationGroup   n_turn(V), V_dot(V) on V_grid (Fig. 5.2 ceiling)
    accel     MaxAccelerationGroup    acc_max(V) on V_grid (G3)
    limit     AutorotationLimitComp   V_sw where n_turn = 1
    powered   PoweredTurnGroup        n_p at V_sw (powered_turn='exact')
    maneuver  ReturnToTargetGroup     t1, t2, trajectory

powered_turn='fixed' drops PoweredTurnGroup; n_p is then an input.
The rotor design inputs shared by decel and accel are promoted (SHARED);
powered.hover.* and powered.power.* are set through their paths.
"""

import numpy as np
import openmdao.api as om

from prouty.special_performance.autorotation_limit_comp import AutorotationLimitComp
from prouty.special_performance.max_acceleration_group import MaxAccelerationGroup
from prouty.special_performance.powered_turn_group import PoweredTurnGroup
from prouty.special_performance.return_to_target_group import ReturnToTargetGroup
from prouty.special_performance.turn_deceleration_group import TurnDecelerationGroup

SHARED = ['sigma', 'cd_bar', 'R', 'theta_1', 'a', 'gamma', 'rho', 'A_b', 'V_tip', 'GW', 'f']


class ReturnToTargetChainGroup(om.Group):
    """Tables, autorotative limit, powered turn and the maneuver, pp. 368-371."""

    def initialize(self):
        self.options.declare('V_grid', types=np.ndarray, desc='table speeds, ft/s')
        self.options.declare('num_steps', types=int, default=40)
        self.options.declare('powered_turn', default='exact', values=('exact', 'fixed'))
        self.options.declare('boundary', default='transient',
                             values=('transient', 'steady_turn', 'level'))

    def setup(self):
        Vg = self.options['V_grid']
        M = len(Vg)
        self.add_subsystem('decel', TurnDecelerationGroup(num_nodes=M,
                                                          boundary=self.options['boundary']),
                           promotes_inputs=SHARED + [('V', 'V_table'), 'band_fraction'])
        self.add_subsystem('accel', MaxAccelerationGroup(num_nodes=M),
                           promotes_inputs=SHARED + [('V', 'V_table'), 'T_max', 'P_MR_avail'])
        self.add_subsystem('limit', AutorotationLimitComp(V_grid=Vg))
        exact = self.options['powered_turn'] == 'exact'
        if exact:
            self.add_subsystem('powered', PoweredTurnGroup(), promotes_inputs=['GW'])
        self.add_subsystem('maneuver', ReturnToTargetGroup(
            V_grid=Vg, num_steps=self.options['num_steps']),
            promotes_inputs=['V_0'] + ([] if exact else ['n_p']),
            promotes_outputs=['t1', 't2', 't_total', 'V_min', 'V_end', 'x', 'y'])

        self.connect('decel.n_turn', ['limit.n_tab', 'maneuver.n_tab'])
        self.connect('decel.V_dot', 'maneuver.Vdot_tab')
        self.connect('accel.acc_max', 'maneuver.acc_tab')
        self.connect('limit.V_sw', ['maneuver.V_sw'] + (['powered.V_sw'] if exact else []))
        if exact:
            self.connect('powered.n_p', 'maneuver.n_p')
        self.set_input_defaults('V_table', val=Vg, units='ft/s')
        self.set_input_defaults('V_tip', val=650.0, units='ft/s')
        self.set_input_defaults('gamma', val=8.1 * np.ones(M))
        self.set_input_defaults('GW', val=20000.0, units='lbf')
