"""Span efficiency factor of a tapered surface.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 503, Figure 8.17. Used for the
horizontal stabiliser, Table 8.2 p. 501, and the vertical stabiliser,
Table 8.3 p. 511.

Digitised by tracking the five printed curves column by column from the
right frame, where they are cleanly separated, leftwards. The ordering
delta(A.R.=4) < delta(8) < delta(12) < delta(16) < delta(20) holds at every
tabulated taper ratio, which is the check that no track jumped.
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_LAMBDA = np.arange(0.10, 1.2001, 0.02)

_DELTA_AR = np.array([4.0, 8.0, 12.0, 16.0, 20.0])

# delta_i of Figure 8.17 p. 503, one row per aspect ratio
_DELTA = np.array([
    [
          0.04369,   0.03855,   0.03383,   0.02952,   0.02560,   0.02206,   0.01887,   0.01604,
          0.01353,   0.01134,   0.00946,   0.00787,   0.00654,   0.00548,   0.00466,   0.00407,
          0.00370,   0.00353,   0.00354,   0.00372,   0.00406,   0.00455,   0.00516,   0.00588,
          0.00670,   0.00761,   0.00858,   0.00961,   0.01068,   0.01180,   0.01295,   0.01414,
          0.01536,   0.01660,   0.01786,   0.01914,   0.02043,   0.02173,   0.02303,   0.02433,
          0.02562,   0.02691,   0.02818,   0.02943,   0.03067,   0.03187,   0.03304,   0.03418,
          0.03528,   0.03633,   0.03734,   0.03829,   0.03918,   0.04002,   0.04079,   0.04149,
    ],
    [
          0.06571,   0.05839,   0.05168,   0.04557,   0.04004,   0.03505,   0.03060,   0.02666,
          0.02320,   0.02021,   0.01767,   0.01555,   0.01383,   0.01250,   0.01152,   0.01088,
          0.01056,   0.01053,   0.01078,   0.01128,   0.01201,   0.01295,   0.01407,   0.01537,
          0.01681,   0.01837,   0.02004,   0.02178,   0.02361,   0.02551,   0.02748,   0.02952,
          0.03163,   0.03381,   0.03606,   0.03838,   0.04076,   0.04321,   0.04571,   0.04828,
          0.05092,   0.05361,   0.05636,   0.05916,   0.06202,   0.06494,   0.06791,   0.07093,
          0.07400,   0.07712,   0.08029,   0.08350,   0.08677,   0.09007,   0.09342,   0.09681,
    ],
    [
          0.08339,   0.07474,   0.06681,   0.05956,   0.05297,   0.04703,   0.04172,   0.03700,
          0.03285,   0.02926,   0.02620,   0.02364,   0.02158,   0.01997,   0.01880,   0.01806,
          0.01770,   0.01772,   0.01809,   0.01879,   0.01979,   0.02107,   0.02261,   0.02439,
          0.02638,   0.02857,   0.03092,   0.03342,   0.03605,   0.03882,   0.04172,   0.04476,
          0.04792,   0.05121,   0.05462,   0.05816,   0.06183,   0.06561,   0.06952,   0.07355,
          0.07770,   0.08196,   0.08634,   0.09083,   0.09543,   0.10015,   0.10497,   0.10991,
          0.11495,   0.12009,   0.12534,   0.13069,   0.13615,   0.14170,   0.14735,   0.15310,
    ],
    [
          0.09996,   0.09043,   0.08171,   0.07378,   0.06660,   0.06014,   0.05437,   0.04926,
          0.04476,   0.04086,   0.03752,   0.03470,   0.03238,   0.03051,   0.02909,   0.02808,
          0.02748,   0.02727,   0.02743,   0.02794,   0.02880,   0.02998,   0.03146,   0.03324,
          0.03530,   0.03762,   0.04018,   0.04297,   0.04597,   0.04920,   0.05264,   0.05630,
          0.06017,   0.06426,   0.06855,   0.07306,   0.07779,   0.08272,   0.08786,   0.09321,
          0.09877,   0.10453,   0.11050,   0.11668,   0.12305,   0.12964,   0.13642,   0.14340,
          0.15059,   0.15797,   0.16556,   0.17334,   0.18132,   0.18949,   0.19786,   0.20642,
    ],
    [
          0.13174,   0.11960,   0.10847,   0.09831,   0.08909,   0.08076,   0.07328,   0.06662,
          0.06072,   0.05556,   0.05109,   0.04726,   0.04405,   0.04141,   0.03931,   0.03773,
          0.03665,   0.03604,   0.03588,   0.03616,   0.03684,   0.03791,   0.03934,   0.04112,
          0.04322,   0.04562,   0.04830,   0.05124,   0.05441,   0.05783,   0.06149,   0.06540,
          0.06956,   0.07396,   0.07862,   0.08354,   0.08870,   0.09413,   0.09981,   0.10576,
          0.11197,   0.11845,   0.12519,   0.13220,   0.13948,   0.14704,   0.15487,   0.16297,
          0.17136,   0.18003,   0.18897,   0.19821,   0.20772,   0.21753,   0.22763,   0.23802,
    ],
])



class SpanEfficiencyComp(om.ExplicitComponent):
    """Induced drag penalty of a tapered planform, p. 503::

        C_Di = C_L^2 (1 + delta_i) / (pi A.R.)

    ``delta_i`` is what a planform costs against the elliptic ideal, and
    Figure 8.17 is the classical lifting-line result: zero for an elliptic
    loading, which a straight-tapered wing approaches near ``lambda = 0.4``,
    and rising either side of it. The minimum is shallow and it moves very
    little with aspect ratio, but its depth does not: at ``lambda = 0.4``
    the penalty is 0.5 % at ``A.R. = 4`` and 3.9 % at 20.

    That is the useful reading. Taper is worth having, and it is worth more
    on a slender surface than on a stubby one; but no stabiliser on a
    helicopter is slender, so ``delta_i`` stays at the level of a rounding
    error. Prouty's own values are 0.02 for the horizontal stabiliser and
    0.01 for the fin, and dropping the term entirely would change the
    stabiliser induced drag by 2 %, which is 0.2 lb of the 800 lb X
    equilibrium.

    Below the printed curves
    ------------------------
    Figure 8.17 starts at ``A.R. = 4`` and the example helicopter's fin is
    3.2 effective. The table clamps rather than extrapolating: the curves
    converge downward as aspect ratio falls, so holding the ``A.R. = 4``
    value is conservative and the alternative is inventing a sixth curve.
    At ``lambda_V = 0.21`` that gives 0.021 against the 0.01 Prouty reads,
    a difference worth 0.1 lb on the fin.

    Geometry in, geometry out: scalar, like ``LiftCurveSlopeComp``. Taper
    ratio and aspect ratio do not change with flight condition, and the
    ``delta`` of ``HorizStabLiftDragComp`` is a scalar input.

    Notes
    -----
    Tabulated from ``lambda = 0.10`` to 1.20. Below 0.10 the printed curves
    turn nearly vertical -- ``delta_i`` at ``A.R. = 20`` triples between
    0.1 and 0, and column tracking cannot resolve five near-vertical lines.
    A taper ratio under 0.1 is a pointed tip, which is not a stabiliser
    planform.
    """

    def setup(self):
        self.add_input('lambda_taper', val=0.4,
                       desc='tip chord over root chord')
        self.add_input('A_R', val=4.5, desc='aspect ratio')

        self.add_output('delta', desc='span efficiency factor, Figure 8.17')

        self._curves = [InterpND(method='akima', points=_LAMBDA, values=v,
                                 extrapolate=True) for v in _DELTA]
        self.declare_partials('delta', ['lambda_taper', 'A_R'])

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
        i, f, df = self._rows(inputs['A_R'][0], _DELTA_AR)
        lam = np.clip(inputs['lambda_taper'], _LAMBDA[0], _LAMBDA[-1])
        lo, dlo = self._curves[i].interpolate(lam, compute_derivative=True)
        hi, dhi = self._curves[i + 1].interpolate(lam, compute_derivative=True)
        inside = ((inputs['lambda_taper'] > _LAMBDA[0])
                  & (inputs['lambda_taper'] < _LAMBDA[-1]))
        d_dlam = np.where(inside, (1.0 - f) * dlo.ravel() + f * dhi.ravel(), 0.0)
        return (1.0 - f) * lo + f * hi, d_dlam, df * (hi - lo)

    def compute(self, inputs, outputs):
        outputs['delta'], _, _ = self._lookup(inputs)

    def compute_partials(self, inputs, J):
        _, d_dlam, d_dAR = self._lookup(inputs)
        J['delta', 'lambda_taper'] = d_dlam
        J['delta', 'A_R'] = d_dAR
