"""
HeightVelocityGroup -- G2d, height-velocity diagram (Deadman's curve).

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Generating the Deadman's Curve" pp. 352-358, Figures 5.8-5.10.

single engine:
    low_hover     LowHoverHeightComp            h_lo                 p. 354
    min_power     MinPowerSpeedComp             V_min                pp. 356-357
    critical      CriticalSpeedComp             V_CR (Fig. 5.9 top)  pp. 355-357
    high_hover    HighHoverHeightComp           h_hi, h_CR           p. 357
    boundary      HVBoundaryComp                Figure 5.8           p. 355
multiengine (one engine out):
    low_hover     LowHoverHeightComp            h_lo                 p. 357
    critical      MultiEngineCriticalSpeedComp  V_CR, h_CR           pp. 357-358
    high_hover    HighHoverHeightComp           h_hi (assumption: Fig. 5.9 bottom)
    boundary      HVBoundaryComp

C_W/sigma is used as C_T/sigma in the C_L/sigma parameter of Figure 5.9.
Multiengine: V_sink (and P_req for RD) come from the Chapter 4 power curve.
"""

import openmdao.api as om

from prouty.special_performance.critical_speed_comp import CriticalSpeedComp
from prouty.special_performance.high_hover_height_comp import HighHoverHeightComp
from prouty.special_performance.hv_boundary_comp import HVBoundaryComp
from prouty.special_performance.low_hover_height_comp import LowHoverHeightComp
from prouty.special_performance.min_power_speed_comp import MinPowerSpeedComp
from prouty.special_performance.multi_engine_critical_speed_comp import \
    MultiEngineCriticalSpeedComp


class HeightVelocityGroup(om.Group):
    """H-V diagram parameters and boundary, pp. 352-358."""

    def initialize(self):
        self.options.declare('engines', default='single', values=('single', 'multi'))
        self.options.declare('time_delay', default='faa', values=('faa', 'military'))
        self.options.declare('book', types=bool, default=False,
                             desc='multi-engine h_lo as printed (C5-3)')
        self.options.declare('num_points', types=int, default=21,
                             desc='points along each boundary branch')

    def setup(self):
        opt = self.options
        td = opt['time_delay']
        self.add_subsystem('low_hover', LowHoverHeightComp(engines=opt['engines'],
                                                           book=opt['book']),
                           promotes=['*'])
        if opt['engines'] == 'single':
            self.add_subsystem('min_power', MinPowerSpeedComp(), promotes=['*'])
            self.add_subsystem('critical', CriticalSpeedComp(time_delay=td),
                               promotes_inputs=['V_min', ('CT_sigma', 'CW_sigma'), 'V_tip'],
                               promotes_outputs=['CL_sigma', 'V_CR'])
            self.add_subsystem('high_hover', HighHoverHeightComp(time_delay=td), promotes=['*'])
        else:
            self.add_subsystem('critical', MultiEngineCriticalSpeedComp(time_delay=td),
                               promotes=['*'])
            self.add_subsystem('high_hover', HighHoverHeightComp(time_delay=td),
                               promotes_inputs=['V_CR'], promotes_outputs=['h_hi'])
            self.set_input_defaults('P_avail', val=2000.0, units='hp')
        self.add_subsystem('boundary', HVBoundaryComp(num_nodes=opt['num_points']),
                           promotes=['*'])
