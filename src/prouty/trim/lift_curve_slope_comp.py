"""Slope of the lift curve of a low aspect ratio swept surface.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", Figure 8.6 p. 490, after Hoak, "USAF
Stability and Control Datcom", 1960. Used for the horizontal stabiliser,
Table 8.2 p. 501, and for the vertical stabiliser, p. 506 and Table 8.3
p. 511.
"""

import numpy as np
import openmdao.api as om


class LiftCurveSlopeComp(om.ExplicitComponent):
    """Lift curve slope against aspect ratio and half-chord sweep.

    Figure 8.6 is not an empirical curve. It is the Helmbold-Diederich
    relation plotted, and the closed form is::

        a / A.R. = 2 pi / (2 + sqrt[ A.R.^2 (1 + tan^2 Lambda) + 4 ])

    with ``Lambda`` the half-chord sweep. Two things identify it. The
    abscissa of the printed chart is ``A.R. sqrt(1 + tan^2 Lambda)``, which
    is exactly the group appearing under the root. And the ordinate tops out
    at 1.6, while the ``A.R. -> 0`` limit of the expression is
    ``pi/2 = 1.5708``: the chart is drawn to its own asymptote.

    So the abaque does not need digitising. This is the one chart of the
    chapter that dissolves into a formula.

    Agreement with the book
    -----------------------
    ==================== ======= ========= ========= =======
    surface              A.R.    Lambda    formula   printed
    ==================== ======= ========= ========= =======
    horizontal stab.     4.5     13 deg    4.020     4.0
    vertical stab.       3.2      27 deg   3.290     3.0
    ==================== ======= ========= ========= =======

    The horizontal stabiliser lands. The vertical one is 9.7 % out, and the
    formula is not the suspect: it reproduces one surface exactly and both
    come off the same chart with the same relation. The likeliest cause is a
    reading of Figure 8.6 at ``A.R.eff = 3.2``, where the curves are steep
    and closely spaced. See the open anchors of ``docs/validation_trim.md``.

    Notes
    -----
    Compressibility is absent, as it is from Figure 8.6. The Datcom form
    carries ``beta = sqrt(1 - M^2)`` and a two-dimensional slope ratio
    ``kappa``; the chart is drawn for ``beta = kappa = 1``. Stabiliser
    Mach numbers in preliminary design sit well below where that matters.

    ``A_R`` is the *effective* aspect ratio, not the geometric one. The end
    plate effect of Figure 8.8 p. 491 and the fuselage and horizontal tail
    effects of Figure 8.19 p. 505 both act on it, upstream of here.

    Geometry in, geometry out: this component is scalar, in the manner of
    ``BladeAreaComp``.
    """

    def setup(self):
        self.add_input('A_R', val=4.5, desc='effective aspect ratio')
        self.add_input('sweep', val=0.0, units='rad',
                       desc='half chord sweep angle')

        self.add_output('a', units='1/rad', desc='slope of the lift curve')

        self.declare_partials('a', ['A_R', 'sweep'])

    def _root(self, inputs):
        A_R, sweep = inputs['A_R'][0], inputs['sweep'][0]
        sec2 = 1.0 + np.tan(sweep) ** 2
        return A_R, sweep, sec2, np.sqrt(A_R ** 2 * sec2 + 4.0)

    def compute(self, inputs, outputs):
        A_R, _, _, root = self._root(inputs)
        outputs['a'] = 2.0 * np.pi * A_R / (2.0 + root)

    def compute_partials(self, inputs, J):
        A_R, sweep, sec2, root = self._root(inputs)
        den = 2.0 + root

        J['a', 'A_R'] = (2.0 * np.pi / den
                         - 2.0 * np.pi * A_R / den ** 2 * A_R * sec2 / root)
        dsec2 = 2.0 * np.tan(sweep) * sec2
        J['a', 'sweep'] = (-2.0 * np.pi * A_R / den ** 2
                           * A_R ** 2 * dsec2 / (2.0 * root))
