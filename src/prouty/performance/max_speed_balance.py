"""
MaxSpeedBalance -- speed where the power required meets the power available.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Maximum Speed" p. 320, Figure 4.39 p. 321.

The book matches the power required curves of Figure 4.38 against the engine
curves of Figures 4.1 and 4.2, iterating because the compressibility loss
itself depends on the speed found. Here the match is a residual:

    residual: P_req(V) - P_avail(V) = 0

V is the implicit output, so the trim, the compressibility penalty, the drive
losses and the ram effect on the ratings are all downstream of it and the
iteration on compressibility disappears into the Newton solve. The group
carrying this balance needs a Newton solver with a line search: the Chapter 3
trim stops converging a few knots beyond the maximum speed, so unbounded steps
walk straight out of the model.

The transmission torque limit of Figure 4.1 is already in the engine ratings,
which is what gives the takeoff power lines of Figure 4.39 their kink at low
altitude.

    P_req, P_avail (nn,) --> V (nn,)
"""

import numpy as np
import openmdao.api as om

KNOT = 1.68781      # ft/s


class MaxSpeedBalance(om.BalanceComp):
    """BalanceComp driving the speed to close P_req = P_avail."""

    def __init__(self, num_nodes=1, V_bounds=(40.0 * KNOT, 220.0 * KNOT),
                 guess=140.0 * KNOT, **kwargs):
        super().__init__(**kwargs)
        lower, upper = V_bounds
        self.add_balance('V', val=np.full(num_nodes, guess), units='ft/s',
                         lhs_name='P_req', rhs_name='P_avail', eq_units='hp',
                         lower=lower, upper=upper, ref=100.0 * KNOT, normalize=True,
                         desc='maximum level flight speed')
