"""End plate effect on the horizontal stabiliser aspect ratio.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 491, Figure 8.8, taken from
reference 8.3.
"""

import numpy as np
import openmdao.api as om

# Figure 8.8 is a straight line. Tracked row by row over h/b_H = 0.007 to
# 0.56, a free fit gives 1.0141 + 1.6161 h/b_H with a maximum residual of
# 0.0226; forcing the intercept to the value physics demands leaves
END_PLATE_SLOPE = 1.6546


class EndPlateFactorComp(om.ExplicitComponent):
    """Aspect ratio multiplier from stabiliser end plates, p. 491::

        A.R._eff / A.R. = 1 + 1.655 (h / b_H)

    p. 491 explains the mechanism: end plates block the flow from turning
    round the tip from bottom to top, weakening the tip vortex, so the
    effective span and with it the aspect ratio go up. Many helicopters carry
    small vertical surfaces at the ends of the horizontal stabiliser for
    exactly this.

    Figure 8.8 is a straight line and needs no table. The intercept is forced
    to 1: no end plate can be no effect, and the free fit gives 1.014, which
    is the digitising error rather than a physical offset.

    Not Hoerner's 1.9
    -----------------
    The classical end plate result, ``1 + 1.9 h/b``, is the one usually
    quoted, and Figure 8.8 is 13 % below it. The figure is credited to
    reference 8.3 rather than to Hoerner, and at ``h/b = 0.4`` the difference
    is 1.66 against 1.76 — worth about 3 % on the stabiliser lift curve slope
    through the Helmbold relation. The chart is what Prouty used, so the
    chart is what this reproduces.

    Notes
    -----
    Figure 8.8 is drawn to ``h/b_H = 0.7``. The relation is linear and
    nothing stops it extrapolating, but an end plate more than two thirds of
    the span in height is a fin, and the aspect ratio of a fin is not what
    this equation is about.

    Geometry in, geometry out: scalar, feeding the ``A_R`` of
    ``LiftCurveSlopeComp`` and ``HorizStabLiftDragComp``.
    """

    def setup(self):
        self.add_input('A_R_geo', val=4.5, desc='geometric aspect ratio')
        self.add_input('h_bH', val=0.0,
                       desc='end plate height over stabiliser span')

        self.add_output('factor', desc='aspect ratio multiplier')
        self.add_output('A_R_eff', desc='effective aspect ratio')

        self.declare_partials('factor', 'h_bH', val=END_PLATE_SLOPE)
        self.declare_partials('A_R_eff', ['A_R_geo', 'h_bH'])

    def compute(self, inputs, outputs):
        factor = 1.0 + END_PLATE_SLOPE * inputs['h_bH'][0]
        outputs['factor'] = factor
        outputs['A_R_eff'] = inputs['A_R_geo'][0] * factor

    def compute_partials(self, inputs, J):
        J['A_R_eff', 'A_R_geo'] = 1.0 + END_PLATE_SLOPE * inputs['h_bH'][0]
        J['A_R_eff', 'h_bH'] = inputs['A_R_geo'][0] * END_PLATE_SLOPE
