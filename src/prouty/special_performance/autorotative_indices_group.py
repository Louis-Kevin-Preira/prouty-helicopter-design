"""
AutorotativeIndicesGroup -- G2f, autorotative indices.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Autorotative Indices" pp. 363-364.

    t_equiv   EquivalentHoverTimeComp   t/k, entry and flare (Figure 5.13)
    index     AutorotativeIndexComp     AI, landing flare

J is an input: connect it to RotorSpeedDecayGroup.J (G2a).
"""

import openmdao.api as om

from prouty.special_performance.autorotative_index_comp import AutorotativeIndexComp
from prouty.special_performance.equivalent_hover_time_comp import EquivalentHoverTimeComp


class AutorotativeIndicesGroup(om.Group):
    """Equivalent hover time and autorotative index, p. 363."""

    def setup(self):
        self.add_subsystem('t_equiv', EquivalentHoverTimeComp(), promotes=['*'])
        self.add_subsystem('index', AutorotativeIndexComp(), promotes=['*'])
