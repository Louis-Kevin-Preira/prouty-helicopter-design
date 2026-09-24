"""
MinTouchdownSpeedGroup -- G2e, minimum touchdown speed after an autorotative flare.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Minimum Touchdown Speed" pp. 358-363, Figures 5.11 and 5.12.

    pitch_rate    FlarePitchRateComp       theta_dot_max = gamma Omega Delta_B1/16
    flare_time    FlareTimeComp            Delta_t
    flare_angle   FlareAngleComp           alpha_TPP = min(theta_dot Delta_t, 45 deg)
    autorotation  FlareAutorotationGroup   mu_auto (Chapter 3 closed-form rotor)
    touchdown     TouchdownSpeedComp       V_TD

P_OGE and (C_T/sigma)_max come from Chapters 1 and 4 (inputs here).
"""

import numpy as np
import openmdao.api as om

from prouty.special_performance.flare_angle_comp import FlareAngleComp
from prouty.special_performance.flare_autorotation_group import FlareAutorotationGroup
from prouty.special_performance.flare_pitch_rate_comp import FlarePitchRateComp
from prouty.special_performance.flare_time_comp import FlareTimeComp
from prouty.special_performance.touchdown_speed_comp import TouchdownSpeedComp


class MinTouchdownSpeedGroup(om.Group):
    """Ideal flare: pitch rate, time, angle, autorotation speed, touchdown speed."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('pitch_rate', FlarePitchRateComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('flare_time', FlareTimeComp(num_nodes=nn),
                           promotes=[('Omega_0', 'Omega'), '*'])
        self.add_subsystem('flare_angle', FlareAngleComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('autorotation', FlareAutorotationGroup(num_nodes=nn), promotes=['*'])
        self.add_subsystem('touchdown', TouchdownSpeedComp(num_nodes=nn), promotes=['*'])
        self.set_input_defaults('gamma', val=8.1 * np.ones(nn))
