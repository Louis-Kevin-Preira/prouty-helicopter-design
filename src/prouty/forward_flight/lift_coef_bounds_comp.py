"""
LiftCoefBoundsComp -- bounds on the lift coefficient, Figure 3.58.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 221-222.

Why this exists. Along the reverse flow boundary the calculated angle of
attack, the sweep and the pitch rate are all extreme at once, and the airfoil
equations respond by handing the element a lift coefficient of 100 or more
(p. 221). The dynamic pressure there is small, but "significant errors can be
introduced when an entire blade element is assigned a lift coefficient of 100
or more". Prouty's answer is to clip c_l between two arbitrary curves rather
than to tame any of the three causes -- SweepAngleComp reaching sec(Lambda) of
2.6e7 and StallDelayComp reaching 200 deg are the causes, and this is the
remedy for all three at once.

Maximum boundary. A triangle: zero at alpha = 0, peak at 45 deg, back to zero
at 90 deg, odd in alpha. Prouty describes it as "purely arbitrary and ... based
on a lift curve slope of 2 pi and on the assumption that the sweep and stall
delay effects will produce no benefits beyond an angle of attack of 45 deg".
Reading the figure gives a peak between 6.06 and 6.20, and 2 pi = 6.283 is
adopted: it is inside the reading error and is the only round number nearby.
The resulting slope below 45 deg is 8.0 per radian, comfortably above any
airfoil's own slope, so the bound never binds in attached flow.

Minimum boundary. "The approximate lift characteristics of a sharp-edged flat
plate with extreme thin airfoil stall characteristics" -- a steep rise to about
0.72 by 10 deg, a shoulder, then a broad hump peaking at 1.12 near 45 to
50 deg and falling to zero at 90. It is close to but fuller than sin(2 alpha),
the classic flat plate result, between 10 and 30 deg. Digitised by eye at
5 deg intervals from Figure 3.58; the curve is hand drawn and Prouty calls it
approximate himself, so about +/- 0.05 is the honest accuracy. Automatic pixel
tracking was tried and abandoned: the hatching drawn along both boundaries and
the two curve labels sit inside the band and defeat it.

Outside +/- 90 deg. The angles coming from AlphaComp reach 289 deg, so the
bounds are folded: alpha is wrapped to (-180, 180], then reflected about
+/- 90 deg with a sign change, which is the symmetry of an uncambered section.
At 135 deg the maximum bound is -2 pi, not +2 pi.

The clamp is smooth. Hard min and max would put kinks on curves crossing the
whole disc, right where the Newton solver is already working hardest. The
smooth forms

    smin(x, y) = [x + y - sqrt((x-y)^2 + k^2)] / 2
    smax(x, y) = [x + y + sqrt((x-y)^2 + k^2)] / 2

are C-infinity and biased by k/2 only where the two arguments meet; at the
default k = 0.02 that is 0.01 in c_l.

    alpha, cl --> LiftCoefBoundsComp --> cl_bounded, cl_max, cl_min
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

CL_MAX_PEAK = 2.0 * np.pi           # at 45 deg, Figure 3.58
PEAK_DEG = 45.0

# Minimum boundary, read from Figure 3.58 at 5 deg intervals
MIN_ALPHA_DEG = np.arange(0.0, 95.0, 5.0)
MIN_CL = np.array([0.00, 0.36, 0.72, 0.76, 0.80, 0.88, 0.96, 1.04, 1.09,
                   1.12, 1.12, 1.10, 1.05, 0.97, 0.86, 0.72, 0.54, 0.30,
                   0.00])


class LiftCoefBoundsComp(om.ExplicitComponent):
    """Clip c_l between Prouty's arbitrary boundaries, p. 221."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('blend', types=float, default=0.02,
                             desc='softness of the clamp, in c_l')

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)
        rows = np.arange(nn * n_psi * n_r)

        self._min_table = InterpND(method='akima', points=MIN_ALPHA_DEG,
                                   values=MIN_CL, extrapolate=True)

        self.add_input('alpha', shape=field, units='rad')
        self.add_input('cl', shape=field, desc='lift coefficient before bounds')

        self.add_output('cl_bounded', shape=field)
        self.add_output('cl_max', shape=field, desc='upper boundary')
        self.add_output('cl_min', shape=field, desc='lower boundary')

        self.declare_partials('cl_bounded', ['alpha', 'cl'], rows=rows,
                              cols=rows)
        for out in ('cl_max', 'cl_min'):
            self.declare_partials(out, 'alpha', rows=rows, cols=rows)

    def _fold(self, alpha):
        """Wrap to (-180, 180] then reflect about +/- 90 deg, keeping sign."""
        deg = np.degrees(np.real(alpha))
        wrapped = (deg + 180.0) % 360.0 - 180.0

        folded = np.where(wrapped > 90.0, 180.0 - wrapped,
                          np.where(wrapped < -90.0, -180.0 - wrapped, wrapped))
        sign = np.where(np.abs(wrapped) > 90.0, -1.0, 1.0)
        return folded, sign

    def _bounds(self, inputs):
        """Both boundaries and their slopes with respect to alpha, in rad."""
        folded, sign = self._fold(inputs['alpha'])
        magnitude = np.abs(folded)
        direction = np.sign(folded)
        direction = np.where(direction == 0.0, 1.0, direction)

        rising = magnitude <= PEAK_DEG
        slope = CL_MAX_PEAK / PEAK_DEG
        cl_max = sign * direction * np.where(
            rising, slope * magnitude, slope * (90.0 - magnitude))

        table, d_table = self._min_table.interpolate(
            magnitude.ravel(), compute_derivative=True)
        cl_min = sign * direction * table.reshape(magnitude.shape)

        # The two sign factors cancel. Reflecting about +/- 90 deg flips both
        # the value, through `sign`, and the direction in which the folded
        # angle moves, d(folded)/d(alpha) = sign, so the product is sign^2 = 1.
        # Carrying only one of them leaves the derivative wrong by a factor
        # of -1 over a third of the disc, which is what a finite difference
        # check caught.
        d_max = np.where(rising, slope, -slope)
        d_min = d_table.reshape(magnitude.shape)

        to_deg = 180.0 / np.pi          # tables are indexed in degrees
        return cl_max, cl_min, d_max * to_deg, d_min * to_deg

    def compute(self, inputs, outputs):
        cl_max, cl_min, _, _ = self._bounds(inputs)
        k = self.options['blend']
        cl = inputs['cl']

        hi = np.maximum(np.real(cl_max), np.real(cl_min))
        lo = np.minimum(np.real(cl_max), np.real(cl_min))

        under = 0.5 * (cl + hi - np.sqrt((cl - hi) ** 2 + k ** 2))
        outputs['cl_bounded'] = 0.5 * (
            under + lo + np.sqrt((under - lo) ** 2 + k ** 2))
        outputs['cl_max'] = cl_max
        outputs['cl_min'] = cl_min

    def compute_partials(self, inputs, partials):
        cl_max, cl_min, d_max, d_min = self._bounds(inputs)
        k = self.options['blend']
        cl = inputs['cl']

        top = cl_max >= cl_min
        hi, lo = np.where(top, cl_max, cl_min), np.where(top, cl_min, cl_max)
        d_hi = np.where(top, d_max, d_min)
        d_lo = np.where(top, d_min, d_max)

        root_hi = np.sqrt((cl - hi) ** 2 + k ** 2)
        under = 0.5 * (cl + hi - root_hi)
        du_dcl = 0.5 * (1.0 - (cl - hi) / root_hi)
        du_dhi = 0.5 * (1.0 + (cl - hi) / root_hi)

        root_lo = np.sqrt((under - lo) ** 2 + k ** 2)
        db_du = 0.5 * (1.0 + (under - lo) / root_lo)
        db_dlo = 0.5 * (1.0 - (under - lo) / root_lo)

        partials['cl_bounded', 'cl'] = (db_du * du_dcl).ravel()
        partials['cl_bounded', 'alpha'] = (
            db_du * du_dhi * d_hi + db_dlo * d_lo).ravel()
        partials['cl_max', 'alpha'] = d_max.ravel()
        partials['cl_min', 'alpha'] = d_min.ravel()
