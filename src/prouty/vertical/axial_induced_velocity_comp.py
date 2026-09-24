"""
AxialInducedVelocityComp -- induced velocity from climb to windmill brake.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "States of Flow" pp. 93-95; Figure 2.13 p. 113 (Castles & Gray,
NACA TN 2474).

With x = V_D_bar = V_D / v_1hov and v1_bar = v_1 / v_1hov, three branches:

    momentum, climb and low descent   v_m  = x/2 + sqrt(x^2/4 + 1)     pp. 94-95
    wind tunnel, Figure 2.13          v_c  = x - y(x, theta_1)         p. 113
    momentum, windmill brake          v_wb = x/2 - sqrt(x^2/4 - 1)     p. 95

Momentum fails in the vortex ring state (p. 95), so the three are blended with
cubic smoothsteps s_lo, s_hi over low_window and high_window:

    v1_bar = (1 - s_lo) v_m + s_lo [(1 - s_hi) v_c + s_hi v_wb]

low_window = (0, 0.25): momentum "breaks down when the rate of descent is
approximately a quarter of the hover-induced velocity" (p. 95); hover and climb
stay pure momentum, v1_bar(0) = 1. high_window = (2.6, 3.6): Figure 2.13 stops
at x = 2.6, short of windmill-brake momentum; beyond its last point the chart
is continued with its end slope (C1) and blended in (C2-4).

Figure 2.13 is stored as y = V_D_bar - v1_bar against x = V_D_bar for
theta_1 = 0 and -12 deg, and interpolated linearly in theta_1 (the book reads
it at -10 deg, p. 112). Digitization: calibrated pixel scan of p. 113, row
scans on the steep branches, column scans on the lower branch; x strictly
increasing, so the -12 deg hook is sampled on its steep side. Akima in x.

    V_D_bar, v_hov (nn,), theta_1 --> v1_bar, v1 (nn,)
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

from prouty.vertical._smooth import smoothstep

_X_T0 = np.array([
    0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900,
    1.000, 1.100, 1.200, 1.260, 1.316, 1.372, 1.410, 1.447, 1.484, 1.522,
    1.559, 1.597, 1.634, 1.671, 1.709, 1.746, 1.784, 1.834, 1.908, 1.989,
    2.070, 2.167, 2.267, 2.376, 2.491, 2.600])
_Y_T0 = np.array([
    -1.070, -1.058, -1.038, -1.016, -1.000, -0.987, -0.980, -0.977, -0.976, -0.976,
    -0.983, -1.001, -1.020, -1.036, -1.053, -1.068, -1.036, -0.990, -0.928, -0.854,
    -0.777, -0.686, -0.556, -0.400, -0.237, -0.090, 0.036, 0.200, 0.400, 0.600,
    0.800, 1.000, 1.200, 1.400, 1.600, 1.773])

_X_T12 = np.array([
    0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900,
    1.000, 1.100, 1.200, 1.300, 1.400, 1.500, 1.600, 1.650, 1.700, 1.740,
    1.765, 1.785, 1.797, 1.804, 1.808, 1.813, 1.820, 1.837, 1.855, 1.896,
    1.937, 1.961, 1.986, 2.052, 2.133, 2.217, 2.310, 2.416, 2.538, 2.600])
_Y_T12 = np.array([
    -1.085, -1.065, -1.045, -1.025, -1.012, -1.004, -1.001, -1.001, -1.004, -1.016,
    -1.032, -1.052, -1.078, -1.112, -1.166, -1.233, -1.311, -1.353, -1.383, -1.370,
    -1.330, -1.260, -1.180, -1.080, -0.960, -0.830, -0.710, -0.550, -0.400, -0.170,
    0.000, 0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 1.200, 1.400, 1.505])

FIGURE_2_13 = {0.0: (_X_T0, _Y_T0), -12.0: (_X_T12, _Y_T12)}   # theta_1 in deg
_THETA_1_LOW = np.radians(-12.0)


class _Figure213Curve:
    """y(x) for one twist: Akima on the chart, end slope beyond it, held below it."""

    def __init__(self, x, y):
        self.x_min, self.x_max = x[0], x[-1]
        self._interp = InterpND(method='akima', points=x, values=y, extrapolate=False)
        y_end, dy_end = self._interp.interpolate(np.array([self.x_max]), compute_derivative=True)
        self.y_end, self.dy_end = np.real(y_end[0]), np.real(dy_end[0, 0])

    def __call__(self, x):
        xr = np.real(x)
        xc = np.where(xr < self.x_min, self.x_min, np.where(xr > self.x_max, self.x_max, x))
        y, dy = self._interp.interpolate(np.atleast_1d(xc), compute_derivative=True)
        above = xr > self.x_max
        y = np.where(above, self.y_end + self.dy_end * (x - self.x_max), y)
        dy = np.where(above, self.dy_end, np.where(xr < self.x_min, 0.0, dy[:, 0]))
        return y, dy


_CURVE_T0 = _Figure213Curve(_X_T0, _Y_T0)
_CURVE_T12 = _Figure213Curve(_X_T12, _Y_T12)


def axial_inflow(x, theta_1, low_window=(0.0, 0.25), high_window=(2.6, 3.6)):
    """Return v1_bar, d/dx and d/dtheta_1 (theta_1 in rad)."""
    # momentum, climb and low rate of descent, pp. 94-95
    r_m = np.sqrt(0.25 * x ** 2 + 1.0)
    v_m, dv_m = 0.5 * x + r_m, 0.5 + 0.25 * x / r_m

    # Figure 2.13, linear in theta_1, p. 113
    (y0, dy0), (y12, dy12) = _CURVE_T0(x), _CURVE_T12(x)
    w = theta_1 / _THETA_1_LOW
    v_c = x - ((1.0 - w) * y0 + w * y12)
    dv_c = 1.0 - ((1.0 - w) * dy0 + w * dy12)
    dv_c_dth = -(y12 - y0) / _THETA_1_LOW

    # momentum, windmill brake, p. 95 -- evaluated only where it is blended in
    used = np.real(x) > high_window[0]
    xs = np.where(used, x, high_window[0])
    r_wb = np.sqrt(0.25 * xs ** 2 - 1.0)
    v_wb, dv_wb = 0.5 * xs - r_wb, np.where(used, 0.5 - 0.25 * xs / r_wb, 0.0)

    s_lo, ds_lo = smoothstep(x, *low_window)
    s_hi, ds_hi = smoothstep(x, *high_window)
    v_h = (1.0 - s_hi) * v_c + s_hi * v_wb
    dv_h = (1.0 - s_hi) * dv_c + s_hi * dv_wb + ds_hi * (v_wb - v_c)

    v = (1.0 - s_lo) * v_m + s_lo * v_h
    dv_dx = (1.0 - s_lo) * dv_m + s_lo * dv_h + ds_lo * (v_h - v_m)
    dv_dth = s_lo * (1.0 - s_hi) * dv_c_dth
    return v, dv_dx, dv_dth


class AxialInducedVelocityComp(om.ExplicitComponent):
    """Induced velocity in axial flight, momentum blended with Figure 2.13, pp. 93-95, 113."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('low_window', types=tuple, default=(0.0, 0.25),
                             desc='V_D_bar range momentum -> Figure 2.13')
        self.options.declare('high_window', types=tuple, default=(2.6, 3.6),
                             desc='V_D_bar range Figure 2.13 -> windmill brake (C2-4)')

    def setup(self):
        lo, hi = self.options['low_window'], self.options['high_window']
        if not (lo[0] < lo[1] <= hi[0] < hi[1] and hi[0] >= 2.0):
            raise ValueError('windows must satisfy lo0 < lo1 <= hi0 < hi1 and hi0 >= 2')

        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('V_D_bar', val=np.zeros(nn), desc='V_D / v_1hov, descent > 0')
        self.add_input('v_hov', val=np.ones(nn), units='ft/s', desc='hover induced velocity')
        self.add_input('theta_1', val=-0.17453, units='rad', desc='blade twist')
        self.add_output('v1_bar', val=np.ones(nn), desc='v_1 / v_1hov')
        self.add_output('v1', val=np.ones(nn), units='ft/s', desc='induced velocity at the disc')

        self.declare_partials(['v1_bar', 'v1'], 'V_D_bar', rows=ar, cols=ar)
        self.declare_partials(['v1_bar', 'v1'], 'theta_1')
        self.declare_partials('v1', 'v_hov', rows=ar, cols=ar)

    def _inflow(self, inputs):
        return axial_inflow(inputs['V_D_bar'], inputs['theta_1'][0],
                            self.options['low_window'], self.options['high_window'])

    def compute(self, inputs, outputs):
        v, _, _ = self._inflow(inputs)
        outputs['v1_bar'] = v
        outputs['v1'] = v * inputs['v_hov']

    def compute_partials(self, inputs, partials):
        v_hov = inputs['v_hov']
        v, dv_dx, dv_dth = self._inflow(inputs)
        partials['v1_bar', 'V_D_bar'] = dv_dx
        partials['v1', 'V_D_bar'] = dv_dx * v_hov
        partials['v1_bar', 'theta_1'] = dv_dth.reshape(-1, 1)
        partials['v1', 'theta_1'] = (dv_dth * v_hov).reshape(-1, 1)
        partials['v1', 'v_hov'] = v
