"""Effective aspect ratio of the vertical stabiliser.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 504, with the three factors read off
Figure 8.19 p. 505, after Hoak, "USAF Stability and Control Datcom".
Worked example on p. 504, anchors in Table 8.3 p. 511.

The three sub-charts of Figure 8.19 were digitised by tracking dark pixels
column by column with a predictor-corrector that follows the local slope,
the same method used for Figure A.2 in ``prouty.forward_flight``. The tracked
points were smoothed with a spline and sampled onto regular grids, which the
component interpolates with Akima splines.

Three details of the tracking are worth recording, and the first two were got
wrong on a first pass.

The two curves of Figure 8.19a converge but never meet. Past
``b_V/2r_1 = 5.2`` they touch on paper and a column shows one dark band, but
the band is 5 to 6 pixels thick where an isolated line is 2 to 3, so both
lines are still there. They are recovered by splitting the band: where the two
are separately resolved, the band extent minus 2.5 pixels reproduces their
measured spacing exactly, and that same rule carries them apart to the right
frame. The separation falls from 0.152 at ``b_V/2r_1 = 1`` to 0.008 at 7 and
is nowhere zero. Averaging the two together, which an earlier version did,
throws away a real difference.

The four curves of Figure 8.19b do not merge either. They appear to beyond
``Z_H/b_V = -0.78`` only because they turn upward and become steep, so a
single column crosses many rows of one curve. Past that point they are tracked
by *rows* instead: at each ordinate the dark band is found, its left edge is
the ``x/c_V = 0.8`` curve and its right edge the 0.5 curve, since the higher
curve reaches a given value at the lesser sideslip. The two middle curves are
placed evenly between, which the last cleanly resolved column supports -- there
the four are spaced 0.023, 0.023 and 0.032 apart.

The tables stop at ``Z_H/b_V = -0.95`` because the figure does. The curves
leave through the *top* of their own frame at about -0.97, not through the
right-hand edge, so there is nothing printed to read between -0.95 and -1.0.

Figure 8.19c is clean over its whole span except the last 3 % of the abscissa,
where the legend text meets the curve. The table is tracked to
``S_H/S_V = 1.93`` and the spline carries the last stretch.
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_BV_2R1 = np.arange(0.0, 7.001, 0.1)

# A.R._V / A.R._V+B, upper curve of Figure 8.19a, taper ratio <= 0.6
_AR_VB_TAPER_LOW = np.array([
      0.0000,   0.6367,   0.8238,   0.9503,   1.0567,   1.1448,   1.2221,   1.2812,
      1.3398,   1.3921,   1.4389,   1.4803,   1.5164,   1.5473,   1.5731,   1.5940,
      1.6102,   1.6219,   1.6294,   1.6323,   1.6303,   1.6229,   1.4780,   1.4683,
      1.4480,   1.4244,   1.3975,   1.3676,   1.3386,   1.3825,   1.3494,   1.3181,
      1.2882,   1.2600,   1.2343,   1.2115,   1.1932,   1.1898,   1.1578,   1.1781,
      1.1420,   1.1236,   1.1144,   1.1059,   1.0983,   1.0915,   1.0855,   1.0801,
      1.0752,   1.0708,   1.0668,   1.0631,   1.0596,   1.0564,   1.0533,   1.0505,
      1.0482,   1.0466,   1.0456,   1.0450,   1.0443,   1.0431,   1.0410,   1.0387,
      1.0383,   1.0395,   1.0398,   1.0381,   1.0358,   1.0347,   1.0362,
])

# same, lower curve, taper ratio 1.0
_AR_VB_TAPER_UNIT = np.array([
      0.0000,   0.3995,   0.5393,   0.7198,   0.8496,   0.9473,   1.0365,   1.1096,
      1.1746,   1.2361,   1.2866,   1.3324,   1.3725,   1.4065,   1.4349,   1.4583,
      1.4767,   1.4900,   1.4982,   1.5012,   1.4993,   1.4926,   1.4814,   1.4658,
      1.4459,   1.4209,   1.3920,   1.3629,   1.3341,   1.2992,   1.2355,   1.2441,
      1.2197,   1.1985,   1.1802,   1.1640,   1.1492,   1.1356,   1.1232,   1.1121,
      1.1022,   1.0935,   1.0858,   1.0789,   1.0727,   1.0669,   1.0616,   1.0570,
      1.0533,   1.0505,   1.0478,   1.0456,   1.0444,   1.0358,   1.0405,   1.0327,
      1.0328,   1.0331,   1.0328,   1.0322,   1.0313,   1.0304,   1.0296,   1.0290,
      1.0287,   1.0285,   1.0284,   1.0285,   1.0285,   1.0286,   1.0285,
])

_SH_SV = np.arange(0.0, 2.001, 0.05)

# K_H of Figure 8.19c
_K_H = np.array([
      0.0000,   0.0692,   0.1403,   0.2098,   0.2759,   0.3400,   0.4026,   0.4617,
      0.5175,   0.5704,   0.6195,   0.6642,   0.7044,   0.7402,   0.7718,   0.7997,
      0.8303,   0.8552,   0.8771,   0.8974,   0.9163,   0.9339,   0.9502,   0.9654,
      0.9795,   0.9925,   1.0046,   1.0160,   1.0268,   1.0371,   1.0471,   1.0568,
      1.0661,   1.0752,   1.0839,   1.0923,   1.1004,   1.1086,   1.1167,   1.1250,
      1.1336,
])

_ZH_BV = np.arange(0.0, -0.9501, -0.025)

_X_CV = np.array([0.5, 0.6, 0.7, 0.8])

# A.R._V+B+H / A.R._V+B of Figure 8.19b, one row per x/c_V
_AR_VBH = np.array([
    [
          1.0531,   1.0375,   1.0221,   1.0069,   0.9921,   0.9777,   0.9637,   0.9502,
          0.9374,   0.9252,   0.9136,   0.9029,   0.8930,   0.8841,   0.8761,   0.8691,
          0.8633,   0.8586,   0.8551,   0.8530,   0.8524,   0.8534,   0.8563,   0.8613,
          0.8684,   0.8780,   0.8902,   0.9053,   0.9234,   0.9454,   0.9721,   1.0043,
          1.0428,   1.0883,   1.1417,   1.2038,   1.2754,   1.3573,   1.4502,
    ],
    [
          1.1435,   1.1253,   1.1072,   1.0892,   1.0715,   1.0540,   1.0369,   1.0203,
          1.0043,   0.9888,   0.9741,   0.9601,   0.9470,   0.9348,   0.9237,   0.9136,
          0.9048,   0.8971,   0.8908,   0.8860,   0.8826,   0.8812,   0.8818,   0.8849,
          0.8908,   0.8997,   0.9120,   0.9279,   0.9477,   0.9718,   1.0007,   1.0351,
          1.0754,   1.1223,   1.1763,   1.2381,   1.3081,   1.3871,   1.4755,
    ],
    [
          1.2251,   1.1954,   1.1737,   1.1541,   1.1334,   1.1123,   1.0919,   1.0721,
          1.0531,   1.0349,   1.0175,   1.0010,   0.9853,   0.9706,   0.9569,   0.9444,
          0.9335,   0.9250,   0.9195,   0.9177,   0.9106,   0.9115,   0.9141,   0.9184,
          0.9247,   0.9347,   0.9436,   0.9697,   0.9913,   1.0088,   1.0208,   1.0574,
          1.0988,   1.1468,   1.2038,   1.2660,   1.3357,   1.4148,   1.5024,
    ],
    [
          1.2874,   1.2658,   1.2400,   1.2133,   1.1876,   1.1629,   1.1394,   1.1165,
          1.0936,   1.0714,   1.0504,   1.0306,   1.0121,   0.9951,   0.9798,   0.9663,
          0.9546,   0.9445,   0.9359,   0.9233,   0.9132,   0.9115,   0.9148,   0.9177,
          0.9249,   0.9349,   0.9437,   0.9714,   0.9919,   1.0102,   1.0229,   1.0525,
          1.1322,   1.1786,   1.2316,   1.2958,   1.3636,   1.4413,   1.5204,
    ],
])


class VertStabEffectiveARComp(om.ExplicitComponent):
    """Effective aspect ratio of the fin, p. 504::

        A.R._V,eff = A.R._V,geo { f_B [ 1 + K_H f_H ] }

    with the three factors read off the three panels of Figure 8.19 p. 505:

    ============ ========================= ==========================
    ``f_B``       A.R._V / A.R._V+B         vs ``b_V/2r_1``, ``lambda_V``
    ``f_H``       A.R._V+B+H / A.R._V+B     vs ``Z_H/b_V``, ``x/c_V``
    ``K_H``       relative tail size        vs ``S_H/S_V``
    ============ ========================= ==========================

    What the equation says is that a fin does not fly alone. The tail boom
    end-plates it from below and the horizontal stabiliser from the side, and
    both make its effective aspect ratio larger than its geometry suggests.
    For the example helicopter that is 1.8 geometric against 3.2 effective, a
    factor of 1.8, which through ``LiftCurveSlopeComp`` is worth 40 % on the
    fin lift curve slope. Sizing a fin on its geometric aspect ratio would
    undersize it badly.

    Reading the ordinate of Figure 8.19a
    ------------------------------------
    Its axis is labelled ``A.R._V / A.R._V+B`` while the legend printed
    inside it defines the same symbol as "the ratio of the aspect ratio of
    the vertical panel in the presence of the body to that of the isolated
    panel", which is the reciprocal. The two cannot both be right. The
    worked example settles the usage without settling the wording: p. 504
    reads 1.05 off this chart and multiplies by it, and 1.05 is what the
    ordinate shows at ``b_V/2r_1 = 5.13``. This component does the same.

    ``lambda_V`` between the curves
    -------------------------------
    Figure 8.19a draws two curves, labelled ``lambda_V <= .6`` and ``1.0``.
    The first is the upper one: its label sits above both curves, while the
    ``1.0`` is printed on the peak of the lower one, which is how curve
    labels are placed. Between 0.6 and 1.0 this component blends the two with
    a smoothstep, so the transition is continuous in value and slope rather
    than a corner in the middle of a Newton solve. Below 0.6 the chart itself
    says the curve no longer moves.

    Agreement with p. 504
    ---------------------
    The example helicopter has ``S_V = 33``, ``b_V = 7.7``, ``c_V = 4.25``,
    ``A.R._V,geo = 1.8``, ``lambda_V = .21``, ``2r_1 = 1.5``, ``S_H = 18``,
    ``x = 2.5``, ``Z_H = 0``:

    ========= ============ ============ ==========
    factor     digitised    p. 504       error
    ========= ============ ============ ==========
    ``f_B``     1.052        1.05         +0.2 %
    ``f_H``     1.133        1.1          +3.0 %
    ``K_H``     0.660        0.64         +3.1 %
    A.R._eff    3.32         3.2          +3.7 %
    ========= ============ ============ ==========

    Each factor is within chart-reading precision of Prouty's own value, and
    they happen to err the same way, so the product runs 3.7 % high. Through
    the Helmbold relation that is 2 % on the fin lift curve slope, on top of
    the 9.7 % already open between that relation and Table 8.3; see the open
    anchors in ``docs/validation_trim.md``.

    Notes
    -----
    Geometry in, geometry out: scalar, like ``LiftCurveSlopeComp``, whose
    ``A_R`` input this is meant to feed.

    ``Z_H`` is positive for a horizontal surface *below* the fuselage
    centreline, per the legend of Figure 8.19c, and the abscissa of
    Figure 8.19b runs from 0 to -1. Pass ``ZH_bV`` as the figure plots it,
    zero or negative.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        self.add_input('A_R_geo', val=1.8, desc='geometric aspect ratio')
        self.add_input('bV_2r1', val=5.0,
                       desc='fin span over fuselage depth at the fin')
        self.add_input('lambda_V', val=0.5, desc='fin taper ratio')
        self.add_input('ZH_bV', val=0.0,
                       desc='horizontal tail height over fin span, <= 0')
        self.add_input('x_cV', val=0.6,
                       desc='horizontal tail a.c. position over fin chord')
        self.add_input('SH_SV', val=0.5, desc='horizontal tail area over fin')

        self.add_output('f_B', desc='A.R._V / A.R._V+B, Figure 8.19a')
        self.add_output('f_H', desc='A.R._V+B+H / A.R._V+B, Figure 8.19b')
        self.add_output('K_H', desc='relative tail size factor, Figure 8.19c')
        self.add_output('A_R_eff', desc='effective aspect ratio')

        self._body = [InterpND(method='akima', points=_BV_2R1, values=v,
                               extrapolate=True)
                      for v in (_AR_VB_TAPER_LOW, _AR_VB_TAPER_UNIT)]
        self._horiz = [InterpND(method='akima', points=-_ZH_BV, values=v,
                                extrapolate=True) for v in _AR_VBH]
        self._factor = InterpND(method='akima', points=_SH_SV, values=_K_H,
                                extrapolate=True)

        self.declare_partials('f_B', ['bV_2r1', 'lambda_V'])
        self.declare_partials('f_H', ['ZH_bV', 'x_cV'])
        self.declare_partials('K_H', 'SH_SV')
        self.declare_partials('A_R_eff', ['A_R_geo', 'bV_2r1', 'lambda_V',
                                          'ZH_bV', 'x_cV', 'SH_SV'])

    @staticmethod
    def _blend(lam):
        """Smoothstep from the taper <= 0.6 curve to the taper 1.0 curve."""
        t = min(max((lam - 0.6) / 0.4, 0.0), 1.0)
        w = t * t * (3.0 - 2.0 * t)
        dw = 0.0 if t in (0.0, 1.0) else 6.0 * t * (1.0 - t) / 0.4
        return w, dw

    @staticmethod
    def _blend_xcv(x_cV):
        """Linear across the four printed x/c_V curves, clamped at the ends."""
        x = min(max(x_cV, _X_CV[0]), _X_CV[-1])
        i = min(int((x - _X_CV[0]) / 0.1), 2)
        f = (x - _X_CV[i]) / 0.1
        df = 0.0 if x_cV <= _X_CV[0] or x_cV >= _X_CV[-1] else 1.0 / 0.1
        return i, f, df

    def _lookup(self, inputs):
        lam_w, lam_dw = self._blend(inputs['lambda_V'][0])
        low, dlow = self._body[0].interpolate([inputs['bV_2r1'][0]],
                                              compute_derivative=True)
        unit, dunit = self._body[1].interpolate([inputs['bV_2r1'][0]],
                                                compute_derivative=True)
        f_B = (1.0 - lam_w) * low[0] + lam_w * unit[0]
        dfB_dbv = (1.0 - lam_w) * dlow[0, 0] + lam_w * dunit[0, 0]
        dfB_dlam = lam_dw * (unit[0] - low[0])

        i, f, df = self._blend_xcv(inputs['x_cV'][0])
        z = -inputs['ZH_bV'][0]
        lo, dlo = self._horiz[i].interpolate([z], compute_derivative=True)
        hi, dhi = self._horiz[i + 1].interpolate([z], compute_derivative=True)
        f_H = (1.0 - f) * lo[0] + f * hi[0]
        dfH_dz = -((1.0 - f) * dlo[0, 0] + f * dhi[0, 0])
        dfH_dxcv = df * (hi[0] - lo[0])

        K, dK = self._factor.interpolate([inputs['SH_SV'][0]],
                                         compute_derivative=True)
        return (f_B, dfB_dbv, dfB_dlam, f_H, dfH_dz, dfH_dxcv,
                K[0], dK[0, 0])

    def compute(self, inputs, outputs):
        f_B, _, _, f_H, _, _, K_H, _ = self._lookup(inputs)
        outputs['f_B'], outputs['f_H'], outputs['K_H'] = f_B, f_H, K_H
        outputs['A_R_eff'] = inputs['A_R_geo'][0] * f_B * (1.0 + K_H * f_H)

    def compute_partials(self, inputs, J):
        (f_B, dfB_dbv, dfB_dlam, f_H, dfH_dz, dfH_dxcv,
         K_H, dK) = self._lookup(inputs)
        geo = inputs['A_R_geo'][0]
        bracket = 1.0 + K_H * f_H

        J['f_B', 'bV_2r1'] = dfB_dbv
        J['f_B', 'lambda_V'] = dfB_dlam
        J['f_H', 'ZH_bV'] = dfH_dz
        J['f_H', 'x_cV'] = dfH_dxcv
        J['K_H', 'SH_SV'] = dK

        J['A_R_eff', 'A_R_geo'] = f_B * bracket
        J['A_R_eff', 'bV_2r1'] = geo * dfB_dbv * bracket
        J['A_R_eff', 'lambda_V'] = geo * dfB_dlam * bracket
        J['A_R_eff', 'ZH_bV'] = geo * f_B * K_H * dfH_dz
        J['A_R_eff', 'x_cV'] = geo * f_B * K_H * dfH_dxcv
        J['A_R_eff', 'SH_SV'] = geo * f_B * f_H * dK
