"""
SpecificEnduranceComp -- hours flown per pound of fuel, and the loiter speed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Loiter Flight" pp. 330-331, Figure 4.47.

"For these mission legs, the optimum speed is at the bottom of the fuel flow
curve of Figure 4.42, and the endurance at any speed is a function of the
specific endurance, S.E., whose units are hr/lb":

    S.E. = 1 / fuel flow

The loiter speed is therefore the bottom of the fuel flow curve, which is
BestEnduranceSpeedBalance: the residual is simply dFF/dV = 0, the slope coming
from FuelFlowSlopeComp. Unlike the best range speed, it does not move with
wind: the wind changes the distance covered, not the fuel burnt per hour.

Figure 4.47 gives about 0.0022 hr/lb at 11,000 lb and 0.0012 at 23,000 lb for
the example, and shows again the single engine advantage of a turbine, which
the Willans line of G0 carries through its zero-power term.

    FF (nn,) --> SE (nn,)
"""

import numpy as np
import openmdao.api as om


class SpecificEnduranceComp(om.ExplicitComponent):
    """Specific endurance, p. 330."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('FF', val=np.ones(nn), units='lbm/h', desc='fuel flow, all engines')
        self.add_output('SE', val=np.zeros(nn), units='h/lbm', desc='specific endurance')
        self.declare_partials('SE', 'FF', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['SE'] = 1.0 / inputs['FF']

    def compute_partials(self, inputs, partials):
        partials['SE', 'FF'] = -1.0 / inputs['FF'] ** 2


class BestEnduranceSpeedBalance(om.BalanceComp):
    """BalanceComp driving the speed to the bottom of the fuel flow curve, p. 330."""

    def __init__(self, num_nodes=1, V_bounds=(30.0, 160.0), guess=70.0, **kwargs):
        super().__init__(**kwargs)
        lower, upper = V_bounds
        self.add_balance('V', val=np.full(num_nodes, guess), units='kn',
                         lhs_name='dFF_dV', rhs_name='zero', eq_units='lbm/h/kn',
                         lower=lower, upper=upper, ref=50.0, normalize=False,
                         desc='loiter speed, at minimum fuel flow')
