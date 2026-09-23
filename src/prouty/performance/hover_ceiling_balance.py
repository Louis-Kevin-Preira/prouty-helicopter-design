"""
HoverCeilingBalance -- altitude where the power required equals the power available.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Hover Performance" pp. 311-312, Figures 4.33 to 4.35.

The book cross-plots the power required and the installed ratings against
density ratio (Figures 4.33 and 4.34) and reads the hover ceiling off the
crossing (Figure 4.35): the example hovers out of ground effect at sea level
up to 27,800 lb, and has a 7,000 ft ceiling at 20,000 lb on a 95 deg F day.

Here the crossing is solved directly:

    residual: P_req(altitude) - P_avail(altitude) = 0

altitude is the implicit output; the atmosphere, the rotors, the losses and
the engine ratings must all be downstream of it, so the group holding this
balance needs a Newton solver (and its own DirectSolver), for instance

    group.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    group.linear_solver = om.DirectSolver()

In ground effect the same balance is used with the in-ground-effect power
required; out of ground effect, with the vertical drag and pseudo ground
effect of G2. The rating (takeoff, intermediate, maximum continuous) is chosen
upstream by picking the column of P_avail.

    P_req, P_avail (nn,) --> altitude (nn,)
"""

import numpy as np
import openmdao.api as om

TROPOPAUSE = 36089.0


class HoverCeilingBalance(om.BalanceComp):
    """BalanceComp driving the altitude to close P_req = P_avail."""

    def __init__(self, num_nodes=1, altitude_bounds=(-2000.0, TROPOPAUSE), guess=5000.0, **kwargs):
        super().__init__(**kwargs)
        lower, upper = altitude_bounds
        self.add_balance('altitude', val=np.full(num_nodes, guess), units='ft',
                         lhs_name='P_req', rhs_name='P_avail',
                         eq_units='hp', lower=lower, upper=upper, ref=10000.0,
                         normalize=True,
                         desc='hover ceiling, where the power required meets the power available')
