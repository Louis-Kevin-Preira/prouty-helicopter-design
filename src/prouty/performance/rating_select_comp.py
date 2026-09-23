"""
RatingSelectComp -- pick one engine rating out of the installed ratings.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Engine Performance" p. 274 (the three ratings), hover ceilings
p. 312 (the installed ratings of Figures 4.33 and 4.34).

InstalledPowerComp gives P_avail as (nn, 3) for takeoff, intermediate and
maximum continuous; a ceiling or a power margin is computed against one of
them.

    P_avail (nn, 3) --> P_rating (nn,)
"""

import numpy as np
import openmdao.api as om

from prouty.performance.piston_power_lapse_comp import RATINGS


class RatingSelectComp(om.ExplicitComponent):
    """One column of the installed ratings."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('rating', default='takeoff', values=RATINGS)

    def setup(self):
        nn = self.options['num_nodes']
        nr = len(RATINGS)
        self._k = RATINGS.index(self.options['rating'])

        self.add_input('P_avail', val=np.zeros((nn, nr)), units='hp')
        self.add_output('P_rating', val=np.zeros(nn), units='hp',
                        desc=f'installed {self.options["rating"]} power')
        self.declare_partials('P_rating', 'P_avail', val=1.0,
                              rows=np.arange(nn), cols=np.arange(nn) * nr + self._k)

    def compute(self, inputs, outputs):
        outputs['P_rating'] = inputs['P_avail'][:, self._k]
