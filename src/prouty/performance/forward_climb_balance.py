"""
ForwardClimbBalance -- rate of climb in forward flight at a given speed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Climb in Forward Flight" pp. 332-335, Figures 4.48, 4.49 and 4.51.

Figure 4.48 plots the power required for rates of climb from zero to
3,000 ft/min and reads the rate off the crossing with each rating; the cross
plot below it gives the rate against forward speed. Here the crossing is a
residual, with the climb entering the level flight chain through the flat
plate area of ClimbFlatPlateComp:

    residual: P_req(f_climb(V_c)) - P_avail = 0

V_c is the implicit output, so the flat plate area, the trim, the losses and
the ratings all sit downstream and the group carrying this balance needs a
Newton solver with a line search.

Ceilings, p. 334: the absolute ceiling is where the maximum rate of climb
reaches zero and the service ceiling where it reaches 100 ft/min, both of
which are this balance closed on altitude instead, or read off the rate of
climb against altitude of Figure 4.49.

Negative rates are allowed by default: above the ceiling the helicopter
descends, and the same equation still holds as long as the descent stays
shallow enough for the rotor to keep an ordinary wake.

    P_req, P_avail (nn,) --> V_c (nn,)
"""

import numpy as np
import openmdao.api as om


class ForwardClimbBalance(om.BalanceComp):
    """BalanceComp driving the rate of climb to absorb the excess power."""

    def __init__(self, num_nodes=1, V_c_bounds=(-3000.0, 6000.0), guess=1000.0, **kwargs):
        super().__init__(**kwargs)
        lower, upper = V_c_bounds
        self.add_balance('V_c', val=np.full(num_nodes, guess), units='ft/min',
                         lhs_name='P_req', rhs_name='P_avail', eq_units='hp',
                         lower=lower, upper=upper, ref=1000.0, normalize=True,
                         desc='rate of climb in forward flight')
