"""
ClimbCeilingBalance -- absolute and service ceilings in forward flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Forward Flight Ceilings" p. 335, Figures 4.49 and 4.51.

"The altitude at which the rate of climb is zero is the absolute ceiling, and
the altitude at which the rate of climb is 100 ft/min is the service ceiling."
Both are read off the maximum rate of climb against altitude of Figure 4.49,
and repeating the analysis at other gross weights gives the ceilings against
gross weight of Figure 4.51, drawn there at maximum continuous power.

    residual: V_c(altitude) - V_c_target = 0

altitude is the implicit output and V_c comes from ForwardClimbBalance at the
speed for best climb, so the two balances stack: the inner one finds the rate
at an altitude, the outer one moves the altitude until that rate is the target.
A single Newton solve on the group handles both.

ceiling = 'absolute' sets the target to zero, 'service' to 100 ft/min, and
any other value can be given on the V_c_target input.

    V_c, V_c_target (nn,) --> altitude (nn,)
"""

import numpy as np
import openmdao.api as om

TARGETS = {'absolute': 0.0, 'service': 100.0}      # ft/min, p. 335


class ClimbCeilingBalance(om.BalanceComp):
    """BalanceComp driving the altitude to the ceiling definition."""

    def __init__(self, num_nodes=1, ceiling='absolute', altitude_bounds=(-2000.0, 36089.0),
                 guess=10000.0, **kwargs):
        super().__init__(**kwargs)
        if ceiling not in TARGETS:
            raise ValueError(f"ceiling must be one of {tuple(TARGETS)}")
        lower, upper = altitude_bounds
        self.add_balance('altitude', val=np.full(num_nodes, guess), units='ft',
                         lhs_name='V_c', rhs_name='V_c_target', eq_units='ft/min',
                         lower=lower, upper=upper, ref=10000.0, normalize=True,
                         rhs_val=np.full(num_nodes, TARGETS[ceiling]),
                         desc=f'{ceiling} ceiling in forward flight')
