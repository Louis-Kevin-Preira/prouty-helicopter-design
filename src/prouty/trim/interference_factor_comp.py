"""Biplane interference factor for the tail rotor and vertical stabiliser.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 510, Figure 8.22, feeding the
interference drag of p. 509.

Digitised by keeping only the rows where all five printed curves are
visible at once, so their left-to-right order identifies them without any
tracking, then fitting each with a quadratic and rejecting the rows the fit
disowns. 143 to 149 of 153 complete rows survive per curve, with maximum
residuals of 0.002 to 0.006 in K_int.
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_Y_RATIO = np.arange(0.085, 0.2351, 0.005)

_KINT_MU = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

# K_int of Figure 8.22 p. 510, one row per b_V/2R_T
_KINT = np.array([
    [
          0.50311,   0.49825,   0.49339,   0.48854,   0.48368,   0.47883,   0.47399,   0.46915,
          0.46431,   0.45947,   0.45464,   0.44982,   0.44499,   0.44017,   0.43536,   0.43054,
          0.42573,   0.42093,   0.41612,   0.41133,   0.40653,   0.40174,   0.39695,   0.39217,
          0.38739,   0.38261,   0.37784,   0.37307,   0.36830,   0.36354,   0.35878,
    ],
    [
          0.57125,   0.56376,   0.55639,   0.54914,   0.54200,   0.53497,   0.52806,   0.52127,
          0.51459,   0.50803,   0.50158,   0.49524,   0.48903,   0.48292,   0.47694,   0.47106,
          0.46531,   0.45967,   0.45414,   0.44873,   0.44343,   0.43825,   0.43319,   0.42823,
          0.42340,   0.41868,   0.41408,   0.40959,   0.40521,   0.40095,   0.39681,
    ],
    [
          0.62431,   0.61573,   0.60726,   0.59892,   0.59070,   0.58260,   0.57463,   0.56678,
          0.55905,   0.55145,   0.54396,   0.53660,   0.52937,   0.52225,   0.51526,   0.50840,
          0.50165,   0.49503,   0.48853,   0.48216,   0.47590,   0.46977,   0.46377,   0.45788,
          0.45212,   0.44648,   0.44097,   0.43557,   0.43030,   0.42516,   0.42013,
    ],
    [
          0.66877,   0.65859,   0.64860,   0.63877,   0.62913,   0.61965,   0.61035,   0.60123,
          0.59228,   0.58350,   0.57490,   0.56647,   0.55822,   0.55014,   0.54224,   0.53451,
          0.52696,   0.51958,   0.51237,   0.50534,   0.49848,   0.49180,   0.48529,   0.47896,
          0.47280,   0.46682,   0.46101,   0.45537,   0.44991,   0.44463,   0.43951,
    ],
    [
          0.68838,   0.67774,   0.66727,   0.65698,   0.64687,   0.63693,   0.62716,   0.61757,
          0.60816,   0.59892,   0.58985,   0.58097,   0.57225,   0.56371,   0.55535,   0.54716,
          0.53915,   0.53131,   0.52365,   0.51616,   0.50884,   0.50171,   0.49474,   0.48796,
          0.48134,   0.47491,   0.46864,   0.46256,   0.45665,   0.45091,   0.44535,
    ],
])


class InterferenceFactorComp(om.ExplicitComponent):
    """Interference factor of Figure 8.22 p. 510::

        K_int = f( 2 y_V / (2 R_T + b_V),  b_V / 2 R_T )

    The factor scaling the biplane interference drag of p. 509. Both
    arguments come from the same three lengths, so this component takes the
    lengths and forms the ratios itself.

    The abscissa of Figure 8.22 is ``K_int`` and the ordinate the separation
    ratio, which is the transpose of how it is used. The digitisation is
    therefore done by rows rather than columns, which returns ``K_int`` as a
    function of the ratio directly.

    Watch the second parameter
    --------------------------
    Figure 8.22 labels its curve family ``mu = b_V/2R_T``. That is a local
    notation and has nothing to do with the tip speed ratio that carries the
    same symbol throughout the rest of the book. It is the fin span measured
    against the tail rotor diameter.

    What it says
    ------------
    Interference falls as the two surfaces are moved apart and rises as the
    fin grows against the tail rotor. Both are what biplane theory would
    say. The example helicopter has ``b_V = 7.7`` and ``R_T = 6.5``, so
    ``b_V/2R_T = 0.59``, essentially the leftmost printed curve; p. 509 reads
    ``K_int = 0.4``, which that curve gives at a separation ratio of 0.192,
    or ``y_V = 2.0 ft``. Prouty does not print ``y_V``, so this is a
    consistency check rather than an anchor: 2 ft between the tail rotor hub
    and the fin centre is the right size for this aircraft.

    Since the interference drag is three quarters of the fin drag (C8-6),
    the factor is not a detail. Halving the separation ratio from 0.19 to
    0.09 raises ``K_int`` from 0.40 to 0.50 and the fin drag from 58 to
    69 lb.

    Geometry in, geometry out: scalar, like ``LiftCurveSlopeComp``. The
    ``K_int`` of ``VertStabInterferenceDragComp`` is a scalar input.

    Notes
    -----
    Tabulated over the printed span, separation ratio 0.085 to 0.235 and
    ``b_V/2R_T`` from 0.6 to 1.0, and clamped outside it.
    """

    def setup(self):
        self.add_input('y_V', val=2.0, units='ft',
                       desc='tail rotor to fin separation')
        self.add_input('R_T', val=6.5, units='ft', desc='tail rotor radius')
        self.add_input('b_V', val=7.7, units='ft', desc='fin span')

        self.add_output('K_int', desc='interference factor, Figure 8.22')

        self._curves = [InterpND(method='akima', points=_Y_RATIO, values=v,
                                 extrapolate=True) for v in _KINT]
        self.declare_partials('K_int', ['y_V', 'R_T', 'b_V'])

    @staticmethod
    def _rows(value, nodes):
        """Linear across the printed curves, clamped outside them."""
        v = min(max(value, nodes[0]), nodes[-1])
        i = int(np.searchsorted(nodes, v, side='right')) - 1
        i = min(max(i, 0), len(nodes) - 2)
        span = nodes[i + 1] - nodes[i]
        f = (v - nodes[i]) / span
        df = 0.0 if value <= nodes[0] or value >= nodes[-1] else 1.0 / span
        return i, f, df

    def _lookup(self, inputs):
        R_T, b_V = inputs['R_T'][0], inputs['b_V'][0]
        span = 2.0 * R_T + b_V
        ratio = 2.0 * inputs['y_V'] / span
        i, f, df = self._rows(b_V / (2.0 * R_T), _KINT_MU)

        r = np.clip(ratio, _Y_RATIO[0], _Y_RATIO[-1])
        lo, dlo = self._curves[i].interpolate(r, compute_derivative=True)
        hi, dhi = self._curves[i + 1].interpolate(r, compute_derivative=True)
        inside = (ratio > _Y_RATIO[0]) & (ratio < _Y_RATIO[-1])
        dK_dr = np.where(inside, (1.0 - f) * dlo.ravel() + f * dhi.ravel(), 0.0)
        return ((1.0 - f) * lo + f * hi, dK_dr, df * (hi - lo),
                ratio, span, R_T, b_V)

    def compute(self, inputs, outputs):
        outputs['K_int'] = self._lookup(inputs)[0]

    def compute_partials(self, inputs, J):
        _, dK_dr, dK_dmu, ratio, span, R_T, b_V = self._lookup(inputs)

        J['K_int', 'y_V'] = dK_dr * 2.0 / span
        J['K_int', 'R_T'] = (dK_dr * (-2.0 * ratio / span)
                             + dK_dmu * (-b_V / (2.0 * R_T ** 2)))
        J['K_int', 'b_V'] = (dK_dr * (-ratio / span)
                             + dK_dmu / (2.0 * R_T))
