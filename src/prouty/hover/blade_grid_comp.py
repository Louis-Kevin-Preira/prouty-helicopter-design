"""
BladeGridComp -- radial station grid for the combined momentum / blade element
hover method.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, steps 2-3 of the combined momentum and blade element method, p. 69.

Prouty tabulates every quantity at the BOUNDARIES between blade elements, so
`num_elements` elements give `num_elements + 1` stations. All downstream
vectorised components therefore run with num_nodes = num_elements + 1.

    x0 --> BladeGridComp --> r_R        (num_elements + 1,)

The grid spans the root cutout x0 to the tip (r/R = 1). Integration is later
truncated at the tip loss factor B by the integration weights, not by the grid,
so the grid itself stays fixed.
"""

import warnings

import numpy as np
import openmdao.api as om


class BladeGridComp(om.ExplicitComponent):
    """Radial stations r/R, from the root cutout to the tip."""

    def initialize(self):
        self.options.declare('num_elements', types=int, default=10,
                             desc='number of blade elements; p. 69 recommends 5 to 15')
        self.options.declare('distribution', values=('uniform', 'cosine'),
                             default='uniform',
                             desc="'cosine' clusters stations at root and tip")

    def setup(self):
        ne = self.options['num_elements']
        if not 5 <= ne <= 15:
            warnings.warn(f'num_elements = {ne} lies outside the 5 to 15 range '
                          'recommended on p. 69.')

        # Fixed unit-interval abscissae; only the mapping to [x0, 1] varies.
        s = np.linspace(0.0, 1.0, ne + 1)
        if self.options['distribution'] == 'cosine':
            s = 0.5 * (1.0 - np.cos(np.pi * s))
        self._s = s

        self.add_input('x0', val=0.15, desc='root cutout, r/R')
        self.add_output('r_R', val=s, desc='radial stations, r/R')

        # r_R = x0 + (1 - x0) s  ->  d r_R / d x0 = 1 - s, a constant.
        self.declare_partials('r_R', 'x0', val=1.0 - s)

    def compute(self, inputs, outputs):
        outputs['r_R'] = inputs['x0'] + (1.0 - inputs['x0']) * self._s
