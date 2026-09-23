"""
FuselageAeroComp -- airframe lift and drag from Appendix A.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Figure A.2 p. 679, used by the trim loop of Chapter 3, p. 192-198.

    L_F = q (L/q)(alpha_F)          lift, negative in level flight
    D_F = q f(alpha_F)              drag, f being the equivalent flat plate area

Both curves were digitised from the "empennage on" traces of Figure A.2 by
tracking dark pixels column by column with a predictor-corrector that follows
the local slope, which is what lets the solid curve be told from the dashed
one where they cross and where the axis labels overlap them. The tracked
points were then fitted with a smoothing spline and sampled onto a one-degree
grid; the component interpolates that grid with an Akima spline.

Smoothing is not cosmetic. Interpolating the raw tracked points reproduces
every pixel of tremor in the printed line, and Akima turns that into slope
ripple: the lift slope wandered between 1.79 and 2.03 ft^2/deg across the
straight part of the curve, with kinks about every degree. Since d(L_F)/d(alpha_F)
is a term of the trim Jacobian, that ripple is noise fed straight to the Newton
solver. After smoothing the slope holds between 1.872 and 1.918 over
alpha_F = -13 to +6 deg, and the largest slope step anywhere in the table is
0.13 ft^2/deg.

Beyond the digitised span, -22 to +20 deg for lift, the curve is continued by
an exponential approach to an asymptote that matches value and slope at the
junction. A hard clamp there produced a corner that made Akima overshoot as
far in as +17 deg. The drag curve needs no such treatment: it was tracked over
the whole -27 to +25 deg range.

The lift table is then CALIBRATED against Prouty's own readings. Six flight
conditions in Tables 3.3, 3.4 and 3.5 give L_F at a known alpha_F and dynamic
pressure, and they consistently imply a lift slope of 1.953 ft^2/deg where the
digitised trace gives 1.902 -- a 2.7 % difference, worth nothing at -6 deg and
0.67 ft^2 at -13 deg. Two independent digitisations from two different scans
both give 1.90, so the trace is being read correctly; but Prouty computed his
tables from HIS reading of this figure, and reproducing those tables means
using his. The correction applied is

    delta(L/q) = 0.04816 alpha_F + 0.0428     ft^2

faded out with a smoothstep beyond alpha_F = -14 and +6 deg so that the stall
plateaus stay where they were digitised: they carry no book anchors, and
extrapolating a slope correction into them would be invention.

    alpha_F      L/q book   L/q raw   L/q calibrated      f found   f book
     -13.0        -30.00     -29.33       -29.91          22.20     22.21
     -11.7        -27.17     -26.86       -27.38          21.59     21.60
      -6.1        -16.51     -16.19       -16.44          19.82     20.00
      -6.0        -16.33     -16.00       -16.25              -         -
      -2.5         -9.50      -9.35        -9.43          19.34     19.40
       0.0             -          -            -          19.24     19.30
       2.0         -0.51      -0.75        -0.61          19.35     19.52

Worst lift residual after calibration is 0.21 ft^2, against 0.67 before, and
five of the six sit inside 0.10 ft^2. The drag needed no calibration at all:
f = 22.20 against 22.21 at alpha_F = -13 deg, and f(0) = 19.24 against the
19.3 ft^2 quoted as the parasite drag area in Appendix A.

Shape worth knowing. The lift curve is straight at about 1.9 ft^2 per degree
across the whole normal range and stalls into a plateau below -13 deg and
above +18 deg. The drag has a shallow minimum near alpha_F = -1 deg and grows
roughly quadratically either side, reaching 29 ft^2 at +/-20 deg -- half again
the level flight value. That growth is what makes a climb expensive: Table 3.3
shows f rising from 20.0 to 21.6 ft^2 between level flight and a 1000 ft/min
climb, while the fuselage download doubles.

Only the empennage-on curves are digitised. Figure A.2 also plots empennage
off, which differs mainly below -10 deg where the horizontal tail stalls; add
it as a second table if the empennage is ever a design variable.

    alpha_F, q --> FuselageAeroComp --> LF_q, f, L_F, D_F (nn,)
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

# Figure A.2, empennage on, one-degree grid from -25 to +25 deg
ALPHA_F_DEG = np.arange(-25.0, 26.0, 1.0)

L_OVER_Q = np.array([
    -35.64, -35.26, -34.98, -34.79, -34.67, -34.59, -34.48, -34.36, -34.17,
    -33.77, -32.97, -31.66, -29.91, -27.98, -26.01, -24.05, -22.10, -20.14,
    -18.19, -16.25, -14.30, -12.35, -10.40, -8.45, -6.50, -4.54, -2.58,
    -0.61, 1.36, 3.33, 5.28, 7.25, 9.17, 11.02, 12.82, 14.60,
    16.43, 18.24, 19.88, 21.29, 22.39, 23.11, 23.53, 23.83, 24.18,
    24.74, 25.35, 25.80, 26.11, 26.34, 26.50])

F_AREA = np.array([
    32.44, 31.13, 29.93, 28.84, 27.84, 26.91, 26.07, 25.28, 24.56,
    23.90, 23.29, 22.72, 22.20, 21.72, 21.29, 20.90, 20.56, 20.26,
    20.01, 19.80, 19.63, 19.49, 19.38, 19.30, 19.25, 19.24, 19.28,
    19.35, 19.47, 19.62, 19.82, 20.07, 20.35, 20.69, 21.07, 21.49,
    21.96, 22.48, 23.05, 23.67, 24.34, 25.07, 25.88, 26.75, 27.70,
    28.74, 29.88, 31.13, 32.52, 34.07, 35.78])


class FuselageAeroComp(om.ExplicitComponent):
    """Airframe lift and drag against fuselage angle of attack, Figure A.2."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('interp_method', default='akima',
                             values=('akima', 'cubic', 'slinear'))

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        method = self.options['interp_method']

        self._lift = InterpND(method=method, points=ALPHA_F_DEG,
                              values=L_OVER_Q, extrapolate=True)
        self._drag = InterpND(method=method, points=ALPHA_F_DEG,
                              values=F_AREA, extrapolate=True)

        self.add_input('alpha_F', shape=(nn,), units='rad',
                       desc='fuselage angle of attack')
        self.add_input('q', shape=(nn,), units='lbf/ft**2',
                       desc='dynamic pressure')

        self.add_output('LF_q', shape=(nn,), units='ft**2', desc='L_F / q')
        self.add_output('f', shape=(nn,), units='ft**2',
                        desc='equivalent flat plate area')
        self.add_output('L_F', shape=(nn,), units='lbf', desc='fuselage lift')
        self.add_output('D_F', shape=(nn,), units='lbf', desc='fuselage drag')

        for out in ('LF_q', 'f'):
            self.declare_partials(out, 'alpha_F', rows=ar, cols=ar)
        for out in ('L_F', 'D_F'):
            self.declare_partials(out, ['alpha_F', 'q'], rows=ar, cols=ar)

    def _tables(self, alpha_deg):
        LF_q, dL = self._lift.interpolate(alpha_deg, compute_derivative=True)
        f, dD = self._drag.interpolate(alpha_deg, compute_derivative=True)
        return LF_q, f, dL.ravel(), dD.ravel()

    def compute(self, inputs, outputs):
        alpha_deg = np.degrees(np.real(inputs['alpha_F']))
        LF_q, f, _, _ = self._tables(alpha_deg)

        outputs['LF_q'] = LF_q
        outputs['f'] = f
        outputs['L_F'] = inputs['q'] * LF_q
        outputs['D_F'] = inputs['q'] * f

    def compute_partials(self, inputs, partials):
        alpha_deg = np.degrees(np.real(inputs['alpha_F']))
        q = np.real(inputs['q'])
        LF_q, f, dL, dD = self._tables(alpha_deg)

        # tables are indexed in degrees, inputs are in radians
        to_rad = 180.0 / np.pi
        partials['LF_q', 'alpha_F'] = dL * to_rad
        partials['f', 'alpha_F'] = dD * to_rad
        partials['L_F', 'alpha_F'] = q * dL * to_rad
        partials['D_F', 'alpha_F'] = q * dD * to_rad
        partials['L_F', 'q'] = LF_q
        partials['D_F', 'q'] = f
