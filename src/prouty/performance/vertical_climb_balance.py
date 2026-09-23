"""
VerticalClimbBalance -- rate of climb that uses up the excess power.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Climb" pp. 313-316: "This equation has been evaluated for
the example helicopter by equating it to the excess power available and the
resultant vertical rate of climb as a function of altitude is shown on
Figure 4.36."

    residual: dP(V_c) - P_excess = 0            P_excess = P_rating - P_req_hover

V_c is the implicit output; the induced velocity, the climb power and, in
the group, the hover power required are all downstream of it, so the group
holding this balance needs a Newton solver and its own DirectSolver.

The momentum relation of p. 315 is the climb branch. At the hover ceiling the
excess power vanishes and V_c comes out zero. Below the ceiling the bounds
allow a small negative V_c so the balance still has a solution and stays
differentiable, but such a value only measures the power deficit: the descent
branch of momentum theory, and the vortex ring state it leads to, are not
modelled here.

    dP, P_excess (nn,) --> V_c (nn,)
"""

import numpy as np
import openmdao.api as om


class VerticalClimbBalance(om.BalanceComp):
    """BalanceComp driving the rate of climb to absorb the excess power."""

    def __init__(self, num_nodes=1, V_c_bounds=(-50.0, 100.0), guess=20.0, **kwargs):
        super().__init__(**kwargs)
        lower, upper = V_c_bounds
        self.add_balance('V_c', val=np.full(num_nodes, guess), units='ft/s',
                         lhs_name='dP', rhs_name='P_excess', eq_units='hp',
                         lower=lower, upper=upper, ref=20.0, normalize=True,
                         desc='vertical rate of climb')
