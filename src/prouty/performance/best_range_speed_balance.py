"""
BestRangeSpeedBalance -- speed for the best specific range.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Cruising Flight" pp. 323-325, Figures 4.42 and 4.43.

The best range speed is where a line from the origin is tangent to the fuel
flow curve, the origin being displaced by the wind. Writing S.R. =
(V - V_wind)/FF and setting its derivative to zero gives

    d(S.R.)/dV = [FF - (V - V_wind) dFF/dV] / FF^2 = 0

so the residual is taken in its multiplied form, which has no pole where the
fuel flow vanishes:

    residual: FF - (V - V_wind) dFF/dV = 0

V is the implicit output and dFF/dV an input: it is the slope of the fuel flow
against speed at that same point, which FuelFlowSlopeComp supplies from the
power chain. With a Willans line, FF = n a P_rated + b P_req, the slope is
b dP_req/dV, so the condition is really the tangency of the power required
curve seen through the engine.

In wind the tangent starts from V = V_wind rather than from the origin, which
is why the best speed rises into a headwind and falls with a tailwind: 114 kt
in still air for the example, 127 kt into 40 kt and 104 kt with 40 kt behind
(p. 323).

The cruise speed the book actually uses is the one at 99 % of the maximum
specific range, on the fast side of the peak (p. 325); it is a second balance
on S.R. = 0.99 S.R._max, not this one.

    FF, dFF_dV, V_wind (nn,) --> V (nn,)
"""

import numpy as np
import openmdao.api as om

KNOT = 1.68781      # ft/s


class BestRangeSpeedBalance(om.BalanceComp):
    """BalanceComp driving the speed onto the tangent of the fuel flow curve."""

    def __init__(self, num_nodes=1, V_bounds=(40.0, 200.0), guess=110.0, **kwargs):
        super().__init__(**kwargs)
        lower, upper = V_bounds
        self.add_balance('V', val=np.full(num_nodes, guess), units='kn',
                         lhs_name='FF', rhs_name='V_ground_dFF_dV', eq_units='lbm/h',
                         lower=lower, upper=upper, ref=100.0, normalize=True,
                         desc='speed for the best specific range')


class TangencyProductComp(om.ExplicitComponent):
    """(V - V_wind) dFF/dV, the right hand side of the tangency residual."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('V', val=np.full(nn, 100.0), units='kn')
        self.add_input('V_wind', val=0.0, units='kn', desc='headwind, negative for a tailwind')
        self.add_input('dFF_dV', val=np.zeros(nn), units='lbm/h/kn',
                       desc='slope of the fuel flow with speed')
        self.add_output('V_ground_dFF_dV', val=np.zeros(nn), units='lbm/h')

        self.declare_partials('V_ground_dFF_dV', ['V', 'dFF_dV'], rows=ar, cols=ar)
        self.declare_partials('V_ground_dFF_dV', 'V_wind')

    def compute(self, inputs, outputs):
        outputs['V_ground_dFF_dV'] = (inputs['V'] - inputs['V_wind'][0]) * inputs['dFF_dV']

    def compute_partials(self, inputs, partials):
        V_ground = inputs['V'] - inputs['V_wind'][0]
        partials['V_ground_dFF_dV', 'V'] = inputs['dFF_dV']
        partials['V_ground_dFF_dV', 'dFF_dV'] = V_ground
        partials['V_ground_dFF_dV', 'V_wind'] = -inputs['dFF_dV'].reshape(-1, 1)
