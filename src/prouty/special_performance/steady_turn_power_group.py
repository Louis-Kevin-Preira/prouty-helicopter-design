"""
SteadyTurnPowerGroup -- G1, engine power required in a steady turn.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Turns and Pullups" p. 343; Chapter 4 ForwardFlightPowerGroup
pp. 317-319.

    effective_weight  EffectiveWeightComp       GW_eff = n GW
    power             ForwardFlightPowerGroup   level power at GW_eff (Chapter 4 G7)

The Chapter 4 group sees GW_eff as its gross weight; every other input
(V, rotor, fuselage, losses) is promoted unchanged. stall=True (option, default False)
adds the chart-calibrated stall torque increment of C5-6 (Chapter 3 charts
pp. 258-266, twist shift p. 230): in the turn the rotor works at n C_T/sigma,
where the closed-form trim alone has no stall.
"""

import openmdao.api as om

from prouty.performance import ForwardFlightPowerGroup
from prouty.special_performance.effective_weight_comp import EffectiveWeightComp


class SteadyTurnPowerGroup(om.Group):
    """Level flight power of Chapter 4 at the effective weight n*GW, p. 343."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('stall', types=bool, default=False,
                             desc='stall torque increment (C5-6) in the Chapter 4 power')
        self.options.declare('power_options', types=dict, default={},
                             desc='passed to ForwardFlightPowerGroup')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('effective_weight', EffectiveWeightComp(num_nodes=nn),
                           promotes=['*'])
        power_options = dict(stall=self.options['stall'])
        power_options.update(self.options['power_options'])
        self.add_subsystem('power', ForwardFlightPowerGroup(num_nodes=nn, **power_options),
                           promotes_inputs=[('GW', 'GW_eff'), '*'], promotes_outputs=['*'])
