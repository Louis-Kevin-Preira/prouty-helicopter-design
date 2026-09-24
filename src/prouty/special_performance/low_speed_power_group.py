"""
LowSpeedPowerGroup -- G5, level flight power from hover to the join speed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5 p. 368; Chapter 4 pp. 308-319; Chapter 3 trim (induced='exact').

    stencil   JoinStencilComp          V_b -> 3 speeds
    forward   ForwardFlightPowerGroup  level power at the 3 speeds (exact induced velocity)
    slope     JoinSlopeComp            P_b, dP_b
    join      LowSpeedPowerComp        P_level(V), Hermite from P_hover to P_b

P_hover (engine power, OGE) is an input: connect HoverPerformanceGroup.P_req
(Chapter 4). V_b = 40 kt by default, the lowest speed where the level trim
converges at high gross weight.
"""

import openmdao.api as om

from prouty.performance import ForwardFlightPowerGroup
from prouty.special_performance.join_slope_comp import JoinSlopeComp
from prouty.special_performance.join_stencil_comp import JoinStencilComp
from prouty.special_performance.low_speed_power_comp import LowSpeedPowerComp


class LowSpeedPowerGroup(om.Group):
    """P_level(V) for 0 <= V <= V_b."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('stencil', JoinStencilComp(), promotes=['V_b'])
        self.add_subsystem('forward', ForwardFlightPowerGroup(
            num_nodes=3, trim_options=dict(induced='exact')),
            promotes_inputs=[('V', 'V_join'), '*'])
        self.connect('stencil.V_nodes', 'V_join')
        self.add_subsystem('slope', JoinSlopeComp(), promotes=['P_b', 'dP_b'])
        self.connect('forward.P_req', 'slope.P_nodes')
        self.add_subsystem('join', LowSpeedPowerComp(num_nodes=nn), promotes=['*'])
