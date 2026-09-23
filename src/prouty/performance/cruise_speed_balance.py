"""
CruiseSpeedBalance -- speed at 99 % of the maximum specific range.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Cruising Flight" p. 325, Figures 4.42 to 4.44.

"The speed for maximum range corresponds to the maximum specific range, but it
is common to use the speed to the right of the peak where the specific range is
99 % of maximum, the rationale being that the mission time can be shortened
with little sacrifice of economy by flying at this speed." Figure 4.44, and
with it the payload-range curves, are built on that speed.

The peak itself comes from BestRangeSpeedBalance. Here the unknown is the
speed increment past it, which keeps the solution on the fast side of the peak
without needing a bound that depends on the solution:

    V_cruise = V_best + dV_cruise,   dV_cruise >= 0
    residual: S.R.(V_cruise) - fraction * S.R._max = 0

Example helicopter, p. 325 and Figure 4.42 (20,000 lb, sea level): the peak is
0.114 n.mi./lb at 114 kt in still air, 0.076 at 127 kt into a 40 kt headwind
and 0.153 at 104 kt with 40 kt behind; the cruise speeds follow a few knots
above each peak.

    SR, SR_max, fraction, V_best (nn,) --> dV_cruise, V_cruise (nn,)
"""

import numpy as np
import openmdao.api as om


class CruiseSpeedBalance(om.BalanceComp):
    """BalanceComp driving the speed increment past the peak."""

    def __init__(self, num_nodes=1, max_increment=60.0, guess=8.0, **kwargs):
        super().__init__(**kwargs)
        self.add_balance('dV_cruise', val=np.full(num_nodes, guess), units='kn',
                         lhs_name='SR', rhs_name='SR_target', eq_units='NM/lbm',
                         lower=0.0, upper=max_increment, ref=10.0, normalize=True,
                         desc='speed past the best range speed')


class CruiseSpeedComp(om.ExplicitComponent):
    """Cruise speed and the specific range it is asked to reach, p. 325."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('V_best', val=np.full(nn, 110.0), units='kn',
                       desc='speed for maximum specific range')
        self.add_input('dV_cruise', val=np.zeros(nn), units='kn', desc='increment past the peak')
        self.add_input('SR_max', val=np.ones(nn), units='NM/lbm',
                       desc='maximum specific range')
        self.add_input('fraction', val=0.99, desc='fraction of the maximum kept, p. 325')

        self.add_output('V_cruise', val=np.full(nn, 110.0), units='kn')
        self.add_output('SR_target', val=np.ones(nn), units='NM/lbm')

        self.declare_partials('V_cruise', ['V_best', 'dV_cruise'], rows=ar, cols=ar, val=1.0)
        self.declare_partials('SR_target', 'SR_max', rows=ar, cols=ar)
        self.declare_partials('SR_target', 'fraction')

    def compute(self, inputs, outputs):
        outputs['V_cruise'] = inputs['V_best'] + inputs['dV_cruise']
        outputs['SR_target'] = inputs['fraction'][0] * inputs['SR_max']

    def compute_partials(self, inputs, partials):
        partials['SR_target', 'SR_max'] = np.full(self.options['num_nodes'],
                                                  inputs['fraction'][0])
        partials['SR_target', 'fraction'] = inputs['SR_max'].reshape(-1, 1)
