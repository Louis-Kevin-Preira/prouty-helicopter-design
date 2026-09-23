"""Fuselage-induced downwash angle at the horizontal stabiliser.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 498, Figure 8.15 p. 500 and
Figure 8.16 p. 502. Anchors in Table 8.2 p. 501 and Table 8.5 pp. 523-524.
"""

import numpy as np
import openmdao.api as om


class FuselageDownwashAtHorizStabComp(om.ExplicitComponent):
    """Downwash the fuselage induces at the horizontal stabiliser.

    p. 498 writes it as a constant plus a slope on fuselage angle of attack::

        eps_FH = eps_F0 + (d eps_FH / d alpha_F) alpha_F

    Nothing is hidden in it. It is a two-parameter linear fit to wind tunnel
    data, and the whole content of the component is where the two parameters
    come from.

    Where the two numbers come from
    -------------------------------
    ``depsF_dalphaF`` is Figure 8.15, p. 500, which measures the downwash
    behind the fuselage of reference 8.12 with a floating stabiliser used as
    a flow vane, for the body alone and for three wing sizes:

    ========== ==================
    no wing     0.06
    small       0.23
    medium      0.39
    large       0.41
    ========== ==================

    p. 498 says a configuration with external engine nacelles — the example
    helicopter — behaves about like the small wing, hence the default 0.23
    of Table 8.5. p. 498 also reports a wind tunnel measurement on the
    Sikorsky S-76, reference 8.13, that gave 0.15.

    ``eps_F0`` is 0.024 rad in Table 8.5, credited to Figure 8.12 p. 496.
    That figure plots downwash against ``C_T/sigma`` for the fuselage alone
    and various combinations, so the constant is the intercept left after
    the angle-of-attack effect is taken out. It is 1.4 deg, small against
    the 3.5 deg the rotor contributes.

    A third route, for a model already in a tunnel, is Figure 8.16 p. 502:
    plot pitching moment against angle of attack with the stabiliser off and
    on at several incidences, and read the downwash at each intersection,
    where the stabiliser carries no lift and the downwash therefore equals
    its geometric angle of attack.

    Note that Prouty writes ``alpha_F`` here in *radians*, unlike the
    ``alpha_F`` in degrees that indexes the Figure A.2 tables of Chapter 3.
    The 0.23 is a radian-per-radian slope.

    Anchor
    ------
    At 115 knots, with ``Theta = -0.0165`` from p. 522, ``gamma_c = 0`` and
    ``eps_MF = T_M/(4 q A_M) = 0.0404``, the fuselage angle of attack of
    p. 513 is ``alpha_F = -0.0569 rad = -3.26 deg``, against the -3.3 deg
    Table 8.8 p. 530 lists for level flight. This component then returns

        eps_FH = 0.024 + 0.23 (-0.0569) = 0.0109 rad = 0.63 deg

    which is 15 % of the rotor's contribution at the same point and of the
    same sign as the stabiliser incidence. It is small, but it is not
    negligible: dropping it moves ``alpha_H`` by 0.63 deg on a surface whose
    trimmed angle of attack is about -8 deg.

    Notes
    -----
    ``alpha_F`` is an input rather than something rebuilt here.
    ``FuselageAlphaComp``, p. 513, supplies it, and rebuilding the chain in
    two places is how the two would drift apart.

    The same Figure 8.15 supplies the vertical stabiliser's sidewash
    derivative, but p. 509 and Table 8.5 disagree on which number to take
    from it — 0.06 for the body alone against -0.23. That is a question for
    the vertical stabiliser, not for this component.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('alpha_F', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage angle of attack')
        self.add_input('eps_F0', val=0.024, units='rad',
                       desc='downwash at zero fuselage angle of attack')
        self.add_input('depsF_dalphaF', val=0.23,
                       desc='downwash slope, Figure 8.15')

        self.add_output('eps_FH', shape=(nn,), units='rad',
                        desc='fuselage downwash at the horizontal stab.')

        self.declare_partials('eps_FH', 'alpha_F', rows=ar, cols=ar)
        self.declare_partials('eps_FH', ['eps_F0', 'depsF_dalphaF'],
                              rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        outputs['eps_FH'] = (inputs['eps_F0'][0]
                             + inputs['depsF_dalphaF'][0] * inputs['alpha_F'])

    def compute_partials(self, inputs, J):
        J['eps_FH', 'alpha_F'] = np.full_like(inputs['alpha_F'],
                                              inputs['depsF_dalphaF'][0])
        J['eps_FH', 'eps_F0'] = np.ones_like(inputs['alpha_F'])
        J['eps_FH', 'depsF_dalphaF'] = inputs['alpha_F']
