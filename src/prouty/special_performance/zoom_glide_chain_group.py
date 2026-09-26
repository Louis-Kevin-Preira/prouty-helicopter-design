"""
ZoomGlideChainGroup -- G2c connected to G2b and Chapter 4.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Glide Distance" pp. 351-352, Figure 5.6; Chapter 4 level flight
power pp. 317-319.

The autorotation trim (G2b) and the level flight power (Chapter 4 G7) are
both run on the N = nn + 1 speeds [V_1 ..., V_0], so that every per-node
design input has the same shape in the two chains:

    speeds    SpeedNodesComp             V_nodes = [V_1, V_0]
    descent   AutorotationDescentGroup   R/D at V_nodes
    power     ForwardFlightPowerGroup    P_req at V_nodes (stall increment C5-6 by default)
    split     ChainSplitComp             RD, P_1 at V_1;  P_0 at V_0
    zoom      ZoomGlideGroup             Delta_h, Delta_d, glide distance

Only the design inputs listed in SHARED, DESCENT_ONLY and POWER_ONLY are
promoted; the two trims keep their own states.
"""

import numpy as np
import openmdao.api as om

from prouty.performance import ForwardFlightPowerGroup
from prouty.special_performance.autorotation_descent_group import AutorotationDescentGroup
from prouty.special_performance.chain_split_comp import ChainSplitComp
from prouty.special_performance.speed_nodes_comp import SpeedNodesComp
from prouty.special_performance.zoom_glide_group import ZoomGlideGroup

SHARED = ['V_tip', 'rho', 'A_b', 'sigma', 'theta_1', 'a', 'gamma', 'R', 'i_s', 'a1s',
          'l_T_R', 'cd_bar', 'delta_3']
DESCENT_ONLY = ['hp_T0', 'hp_trans', 'hp_acc']
POWER_ONLY = ['V_son', 'P_design_nose', 'P_design_main', 'P_design_tail',
              'load_elec', 'flow_hyd', 'p_hyd']


class ZoomGlideChainGroup(om.Group):
    """Zoom and glide with R/D from G2b and level powers from Chapter 4, p. 352."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1,
                             desc='number of autorotation speeds V_1')
        self.options.declare('stall', types=bool, default=True,
                             desc='stall torque increment (C5-6) in the level power')

    def setup(self):
        nn = self.options['num_nodes']
        N = nn + 1
        self.add_subsystem('speeds', SpeedNodesComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('descent', AutorotationDescentGroup(num_nodes=N),
                           promotes_inputs=SHARED + DESCENT_ONLY + ['GW'])
        self.add_subsystem('power', ForwardFlightPowerGroup(num_nodes=N,
                                                            stall=self.options['stall']),
                           promotes_inputs=SHARED + POWER_ONLY + ['GW'])
        self.connect('V_nodes', ['descent.V', 'power.V'])
        self.add_subsystem('split', ChainSplitComp(num_nodes=nn),
                           promotes_outputs=['RD', 'P_1', 'P_0'])
        self.connect('descent.RD', 'split.RD_nodes')
        self.connect('power.P_req', 'split.P_nodes')
        self.add_subsystem('zoom', ZoomGlideGroup(num_nodes=nn), promotes=['*'])
