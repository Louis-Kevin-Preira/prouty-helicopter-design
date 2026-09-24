"""
HVBoundaryComp -- G2d, height-velocity boundary from the generalized curve.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Generating the Deadman's Curve" pp. 354-355, Figure 5.8 p. 355.

Figure 5.8 (digitized) gives V_x/V_CR along each branch, parametrized here by
the height fraction s in [0, 1] (s = 1 at the nose):

    upper:  h = h_hi - s (h_hi - h_CR),   V = V_CR x_up(s)
    lower:  h = h_lo + s (h_CR - h_lo),   V = V_CR x_lo(s)

The high-speed portion of Figure 5.7 has no analytical method (p. 358).

    s (nn,), h_lo, h_hi, h_CR, V_CR --> V_up, h_up, V_lo, h_lo_branch (nn,)
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

S_UP = np.array([0., .02, .05, .10, .15, .22, .25, .30, .35, .42, .45, .50, .55, .62, .65,
                 .70, .75, .83, .85, .90, .95, 1.])
X_UP = np.array([0., .035, .092, .180, .265, .378, .426, .496, .567, .643, .681, .728, .778,
                 .832, .853, .887, .917, .953, .962, .979, .988, 1.])
S_LO = np.array([0., .01, .03, .05, .07, .10, .12, .15, .18, .22, .25, .30, .35, .42, .45,
                 .50, .55, .62, .65, .70, .75, .83, .85, .88, .90, 1.])
X_LO = np.array([0., .080, .286, .421, .482, .563, .589, .641, .681, .719, .745, .778, .809,
                 .839, .853, .875, .894, .920, .931, .946, .957, .976, .981, .986, .988, 1.])


class HVBoundaryComp(om.ExplicitComponent):
    """Dimensional upper and lower H-V boundaries, Figure 5.8."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=21)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self._up = InterpND(method='akima', points=(S_UP,), values=X_UP, extrapolate=True)
        self._lo = InterpND(method='akima', points=(S_LO,), values=X_LO, extrapolate=True)
        self.add_input('s', val=np.linspace(0.0, 1.0, nn))
        for name, val in (('h_lo', 10.0), ('h_hi', 1300.0), ('h_CR', 95.0)):
            self.add_input(name, val=val, units='ft')
        self.add_input('V_CR', val=80.0, units='kn')
        self.add_output('V_up', val=np.zeros(nn), units='kn')
        self.add_output('h_up', val=np.zeros(nn), units='ft')
        self.add_output('V_lo', val=np.zeros(nn), units='kn')
        self.add_output('h_lo_branch', val=np.zeros(nn), units='ft')
        self.declare_partials(['V_up', 'h_up', 'V_lo', 'h_lo_branch'], 's', rows=ar, cols=ar)
        self.declare_partials(['V_up', 'V_lo'], 'V_CR', rows=ar, cols=zero)
        self.declare_partials('h_up', ['h_hi', 'h_CR'], rows=ar, cols=zero)
        self.declare_partials('h_lo_branch', ['h_lo', 'h_CR'], rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        s = inputs['s']
        outputs['V_up'] = inputs['V_CR'] * self._up.interpolate(s[:, None])
        outputs['V_lo'] = inputs['V_CR'] * self._lo.interpolate(s[:, None])
        outputs['h_up'] = inputs['h_hi'] - s * (inputs['h_hi'] - inputs['h_CR'])
        outputs['h_lo_branch'] = inputs['h_lo'] + s * (inputs['h_CR'] - inputs['h_lo'])

    def compute_partials(self, inputs, J):
        s, V = inputs['s'], inputs['V_CR']
        xu, dxu = self._up.interpolate(s[:, None], compute_derivative=True)
        xl, dxl = self._lo.interpolate(s[:, None], compute_derivative=True)
        J['V_up', 's'] = V * dxu[:, 0]
        J['V_lo', 's'] = V * dxl[:, 0]
        J['V_up', 'V_CR'] = xu
        J['V_lo', 'V_CR'] = xl
        J['h_up', 's'] = -(inputs['h_hi'] - inputs['h_CR']) * np.ones_like(s)
        J['h_up', 'h_hi'] = 1.0 - s
        J['h_up', 'h_CR'] = s
        J['h_lo_branch', 's'] = (inputs['h_CR'] - inputs['h_lo']) * np.ones_like(s)
        J['h_lo_branch', 'h_lo'] = 1.0 - s
        J['h_lo_branch', 'h_CR'] = s
