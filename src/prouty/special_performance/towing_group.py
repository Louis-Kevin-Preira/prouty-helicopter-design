"""
TowingGroup -- G7, towing capability.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Towing" pp. 371-372.

    towline   TowlineTensionComp   tension(gamma)

T_max is an input: the maximum hover gross weight OGE of the Chapter 4
hover analysis (27,800 lb at sea level for the example, p. 312).
"""

import openmdao.api as om

from prouty.special_performance.towline_tension_comp import TowlineTensionComp


class TowingGroup(om.Group):
    """Towline tension against towline angle, pp. 371-372."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1,
                             desc='number of towline angles')

    def setup(self):
        self.add_subsystem('towline', TowlineTensionComp(num_nodes=self.options['num_nodes']),
                           promotes=['*'])
