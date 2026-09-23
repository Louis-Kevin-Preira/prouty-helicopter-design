"""
RotorPerformanceGroup -- step 21 of the combined momentum and blade element
method, plus the standard hover efficiency measures.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 21, p. 72; figure of merit p. 9 and p. 22.

    step 21  DimensionalPerfComp   T, power_hp, Q
             FigureOfMeritComp     FM, power_loading

Inputs  : rho, A, V_tip, R, CT, CQ
Outputs : T, power_hp, Q, FM, power_loading
"""

import openmdao.api as om

from prouty.hover.dimensional_perf_comp import DimensionalPerfComp
from prouty.hover.figure_of_merit_comp import FigureOfMeritComp


class RotorPerformanceGroup(om.Group):
    """Dimensional rotor performance and efficiency measures."""

    def setup(self):
        self.add_subsystem('dimensional', DimensionalPerfComp(), promotes=['*'])
        self.add_subsystem('efficiency', FigureOfMeritComp(), promotes=['*'])
