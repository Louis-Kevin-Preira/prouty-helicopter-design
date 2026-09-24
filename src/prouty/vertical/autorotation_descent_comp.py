"""
AutorotationDescentComp -- rate of descent in vertical autorotation from Figure 2.13.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rate of Descent in Vertical Autorotation" pp. 112-113, Figure 2.13.

Figure 2.13 is read backwards: given y = V_D_bar - v1_bar and the twist, find
V_D_bar on the lower branch (p. 112: y = 0.22, theta_1 = -10 deg --> 1.97). The
chart is the one of G0, blend included, so that G0 and G7 agree:

    y(x) = x - v1_bar(x, theta_1)          AxialInducedVelocityComp, p. 113

For x >= X_LOW = 1.75 both twist curves, their interpolation and the blend
into windmill-brake momentum increase monotonically, so the root is unique.
It is bracketed by bisection, then one Newton step in the working arithmetic
gives the exact complex-step derivative; partials by the implicit function
theorem: dx/dy = 1/y_x, dx/dtheta_1 = -y_theta/y_x. A y below y(X_LOW), i.e.
above the hook of the chart, is held at X_LOW with a warning.

    V_D_bar = x,   v1_bar = x - y,   V_D = x v_1hov                            p. 112

    VD_minus_v1_bar, v_hov (nn,), theta_1 --> V_D_bar_auto, v1_bar_auto, V_D_auto (nn,)
"""

import warnings

import numpy as np
import openmdao.api as om

from prouty.vertical.axial_induced_velocity_comp import axial_inflow

X_LOW, X_HIGH = 1.75, 20.0


def _y(x, theta_1):
    """y = x - v1_bar and its derivatives with respect to x and theta_1."""
    v, dv_dx, dv_dth = axial_inflow(x, theta_1)
    return x - v, 1.0 - dv_dx, -dv_dth


def solve_descent(y_target, theta_1):
    """V_D_bar on the lower branch of Figure 2.13 for a given V_D_bar - v1_bar."""
    yr, th = np.real(y_target), np.real(theta_1)
    lo, hi = np.full_like(yr, X_LOW), np.full_like(yr, X_HIGH)
    below = yr <= _y(lo, th)[0]
    if np.any(below):
        warnings.warn('AutorotationDescentComp: V_D_bar - v1_bar below the lower branch, '
                      'held at V_D_bar = 1.75')
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        up = _y(mid, th)[0] < yr
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    x = 0.5 * (lo + hi)
    y, y_x, _ = _y(x, theta_1)
    x = np.where(below, X_LOW, x + (y_target - y) / y_x)   # exact to first order in cs
    return x, below


class AutorotationDescentComp(om.ExplicitComponent):
    """Figure 2.13 read at the autorotation parameter, pp. 112-113."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('VD_minus_v1_bar', val=np.full(nn, 0.2), desc='V_D_bar - v1_bar')
        self.add_input('v_hov', val=np.ones(nn), units='ft/s', desc='hover induced velocity')
        self.add_input('theta_1', val=-0.17453, units='rad', desc='blade twist')
        self.add_output('V_D_bar_auto', val=np.full(nn, 2.0), desc='V_D / v_1hov')
        self.add_output('v1_bar_auto', val=np.full(nn, 1.8), desc='v_1 / v_1hov')
        self.add_output('V_D_auto', val=np.full(nn, 80.0), units='ft/s',
                        desc='rate of descent in vertical autorotation')
        self.declare_partials(['V_D_bar_auto', 'v1_bar_auto', 'V_D_auto'], 'VD_minus_v1_bar',
                              rows=ar, cols=ar)
        self.declare_partials(['V_D_bar_auto', 'v1_bar_auto', 'V_D_auto'], 'theta_1')
        self.declare_partials('V_D_auto', 'v_hov', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        x, _ = solve_descent(inputs['VD_minus_v1_bar'], inputs['theta_1'][0])
        outputs['V_D_bar_auto'] = x
        outputs['v1_bar_auto'] = x - inputs['VD_minus_v1_bar']
        outputs['V_D_auto'] = x * inputs['v_hov']

    def compute_partials(self, inputs, partials):
        th = inputs['theta_1'][0]
        x, below = solve_descent(inputs['VD_minus_v1_bar'], th)
        _, y_x, y_th = _y(x, th)
        dx_dy = np.where(below, 0.0, 1.0 / y_x)
        dx_dth = np.where(below, 0.0, -y_th / y_x)
        v_hov = inputs['v_hov']
        partials['V_D_bar_auto', 'VD_minus_v1_bar'] = dx_dy
        partials['v1_bar_auto', 'VD_minus_v1_bar'] = dx_dy - 1.0
        partials['V_D_auto', 'VD_minus_v1_bar'] = dx_dy * v_hov
        partials['V_D_bar_auto', 'theta_1'] = dx_dth.reshape(-1, 1)
        partials['v1_bar_auto', 'theta_1'] = dx_dth.reshape(-1, 1)
        partials['V_D_auto', 'theta_1'] = (dx_dth * v_hov).reshape(-1, 1)
        partials['V_D_auto', 'v_hov'] = x
