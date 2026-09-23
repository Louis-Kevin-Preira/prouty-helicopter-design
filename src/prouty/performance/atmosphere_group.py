"""
AtmosphereGroup -- reference day and atmosphere shared by the Chapter 4 groups.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Appendix C, Figures C.1-C.2 pp. 703-704; hot day of Chapter 4 (C4-4).

    altitude, [dT_offset | T_day]
        DayTemperatureComp    dT                  day = standard | offset | isothermal
        AtmosphereComp        rho, density_ratio, T_air, V_son

Every later group reads the atmosphere (engine ratings, density-scaled losses,
rotor powers), so in a combined model it must run first. EngineGroup embeds
one by default for standalone use; pass atmosphere=False when this group is
placed upstream. Without it OpenMDAO runs the groups in order and a group
placed before the embedded atmosphere reads a stale density ratio.
"""

import openmdao.api as om

from prouty.hover.atmosphere_comp import AtmosphereComp
from prouty.performance.day_temperature_comp import DayTemperatureComp

DAYS = ('standard', 'offset', 'isothermal')


class AtmosphereGroup(om.Group):
    """Reference day temperature and standard atmosphere."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('day', default='standard', values=DAYS)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('day', DayTemperatureComp(num_nodes=nn, day=self.options['day']),
                           promotes=['*'])
        self.add_subsystem('atmosphere', AtmosphereComp(num_nodes=nn), promotes=['*'])
