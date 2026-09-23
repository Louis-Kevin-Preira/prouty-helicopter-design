"""
TotalParasiteDragComp -- total equivalent flat plate area of the helicopter.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Total Parasite Drag" p. 304, Figure 4.27 and Table 4.3 p. 305,
procedure and example p. 308.

    f_total = sum of the items + f_misc
    f_design = (1 + margin) f_total

f_misc covers antennas, door handles, lights, steps, skin gaps, cooling
leakage and ventilation (0.5 ft^2 in the example). p. 304 warns that the
method gives minimum estimates and recommends adding at least 20 % for the
items that appear or grow during development; margin is 0 by default and set
to 0.20 to follow that advice.

The items are an option, so any breakdown can be summed. The example of
p. 308 gives 5.8 + 1.1 + 7.0 + 0.7 + 1.2 + 0.8 + 0.2 + 0.2 + 1.3 + 0.5 + 0.5
= 19.3 ft^2 with EXAMPLE_ITEMS.

    f_<item> ..., f_misc, margin --> f_total, f_design
"""

import numpy as np
import openmdao.api as om

EXAMPLE_ITEMS = ('F', 'N', 'M', 'T', 'MLG', 'NLG', 'H', 'V', 'int', 'ex')


class TotalParasiteDragComp(om.ExplicitComponent):
    """Sum of the parasite drag items, p. 308."""

    def initialize(self):
        self.options.declare('items', types=tuple, default=EXAMPLE_ITEMS,
                             desc='names of the drag items, summed as f_<name>')

    def setup(self):
        self._names = [f'f_{name}' for name in self.options['items']] + ['f_misc']
        for name in self._names:
            self.add_input(name, val=0.0, units='ft**2')
        self.add_input('margin', val=0.0, desc='development margin, 0.20 recommended p. 304')

        self.add_output('f_total', val=0.0, units='ft**2', desc='sum of the items')
        self.add_output('f_design', val=0.0, units='ft**2', desc='sum with the margin')

        self.declare_partials('f_total', self._names, val=1.0)
        self.declare_partials('f_design', self._names)
        self.declare_partials('f_design', 'margin')

    def compute(self, inputs, outputs):
        total = sum(inputs[name][0] for name in self._names)
        outputs['f_total'] = total
        outputs['f_design'] = (1.0 + inputs['margin'][0]) * total

    def compute_partials(self, inputs, partials):
        total = sum(inputs[name][0] for name in self._names)
        for name in self._names:
            partials['f_design', name] = 1.0 + inputs['margin'][0]
        partials['f_design', 'margin'] = total
