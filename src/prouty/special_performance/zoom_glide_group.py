"""
ZoomGlideGroup -- G2c, zoom maneuver and glide distance.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Glide Distance" pp. 351-352, Figure 5.6.

    climb_angle     ZoomClimbAngleComp     gamma_c
    altitude_gain   ZoomAltitudeGainComp   Delta_h(V_1)
    glide           GlideDistanceComp      Delta_d(V_1), d(V_1)

Inputs to be connected later: RD(V_1) from G2b (autorotation descent),
P_0 and P_1(V_1) from the Chapter 4 level flight power.
"""

import openmdao.api as om

from prouty.special_performance.glide_distance_comp import GlideDistanceComp
from prouty.special_performance.zoom_altitude_gain_comp import ZoomAltitudeGainComp
from prouty.special_performance.zoom_climb_angle_comp import ZoomClimbAngleComp


class ZoomGlideGroup(om.Group):
    """Zoom altitude gain and glide distances over the speeds V_1, p. 352."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1,
                             desc='number of autorotation speeds V_1')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('climb_angle', ZoomClimbAngleComp(), promotes=['*'])
        self.add_subsystem('altitude_gain', ZoomAltitudeGainComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('glide', GlideDistanceComp(num_nodes=nn), promotes=['*'])
