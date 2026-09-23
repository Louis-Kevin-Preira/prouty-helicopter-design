"""The longitudinal stability map of Figure 9.15.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 617-620. The map itself is
Figure 9.15, p. 619; the two boundary expressions are p. 618.
"""

import numpy as np
import openmdao.api as om

from prouty.stability.modes import describe_modes

#: Figure 9.15, p. 619: horizontal stabilizer areas the map is drawn for.
STABILIZER_AREAS = (18.0, 36.0, 54.0, 72.0, 90.0)

#: p. 618, Routh's discriminant fitted as a quadratic in the two derivatives,
#: as (constant, dM/dzdot, dM/dxdot, dM/dzdot^2, cross, dM/dxdot^2), times
#: 1e-6. Kept as a validation reference, not used by the component.
ROUTH_MAP_COEFFICIENTS = (15030.0, -425.0, -243.0, 1.24, -5.55, -0.819)


def routh_map(dM_dzdot, dM_dxdot):
    """Routh's discriminant from the fitted quadratic of p. 618."""
    c0, c1, c2, c3, c4, c5 = ROUTH_MAP_COEFFICIENTS
    return 1e-6 * (c0 + c1 * dM_dzdot + c2 * dM_dxdot
                   + c3 * dM_dzdot ** 2 + c4 * dM_dzdot * dM_dxdot
                   + c5 * dM_dxdot ** 2)


def classify(char_coeffs):
    """Which region of Figure 9.15 a characteristic equation falls in.

    p. 618 says the boundary between the unstable oscillations and the
    unstable divergences on the right of the map "was determined by finding
    combinations of the two derivatives that made the roots of the
    characteristic equation switch from complex to real". There is no closed
    form for it, so the classification is done on the roots. That also makes
    it discrete, which is why it is a post-processing function and not an
    output.
    """
    modes = describe_modes(char_coeffs)
    unstable = [mode for mode in modes if mode.root.real > 0.0]
    if not unstable:
        return 'stable'
    if any(mode.oscillatory for mode in unstable):
        return 'unstable oscillation'
    return 'unstable divergence'


class LongStabilityMapComp(om.ExplicitComponent):
    """The two boundaries of Figure 9.15, in closed form.

    p. 617 proposes the map as "a guide to the resizing of the horizontal
    stabilizer": plot the aircraft in the plane of its angle-of-attack
    stability ``dM/dzdot`` against its speed stability ``dM/dxdot``, and the
    regions tell you what kind of instability you have. Figure 9.15 draws the
    example helicopter at five stabilizer areas, from 18 to 90 square feet.

    Two curves bound the regions. Routh's discriminant vanishing separates
    stable from unstable oscillations, and ``E``, the constant term of the
    characteristic equation, vanishing separates oscillations from
    divergences. This component produces the second, exactly::

        E = G.W. [(dZ/dxdot)(dM/dzdot) - (dZ/dzdot)(dM/dxdot)] / (m^2 I_yy)

    and with it the boundary itself, ``E = 0`` at::

        dM/dzdot = (dM/dxdot)(dZ/dzdot)/(dZ/dxdot)

    which is what p. 618 prints. Routh's discriminant is not recomputed here;
    ``RouthDiscriminantComp`` already has it from the characteristic equation.

    Exact, not fitted
    -----------------
    p. 618 also gives Routh's discriminant as a fitted quadratic in the two
    derivatives and the characteristic equation with them left as variables.
    Those are a convenience for drawing the figure by hand, and they are
    2 to 7 % off the values the determinant gives. ``E`` above is derived
    instead: the only entry of the matrix with a constant term is
    ``-G.W.`` in the X row, so only two of the six Leibniz permutations reach
    the ``s^2`` coefficient of the determinant, and they give the expression
    above with no approximation. The fitted forms are kept in this module as
    ``routh_map`` and ``ROUTH_MAP_COEFFICIENTS``, for checking against the
    figure rather than for use.

    Why the sign of ``E`` is the useful thing
    -----------------------------------------
    A negative constant term in a polynomial with a positive leading
    coefficient forces a positive real root, so ``E < 0`` is a divergence with
    no further analysis. ``E > 0`` does not conversely guarantee an
    oscillation -- the example helicopter has ``E = +.095`` and four real
    roots -- which is why the right-hand boundary of Figure 9.15 needed
    Prouty's numerical root search and why :func:`classify` works on the roots.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    ``E = .09484`` against the ``.0949`` of the p. 617 quartic, and the
    divergence boundary sits at ``dM/dzdot = -843 ft lb/(ft/sec)`` against the
    aircraft's ``+650``: a margin of 1,493, comfortably on the oscillation
    side of that particular boundary even though the aircraft is diverging for
    the other reason.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('dM_dxdot', val=np.full(nn, 144.0), units='lbf*s')
        self.add_input('dM_dzdot', val=np.full(nn, 650.0), units='lbf*s')
        self.add_input('dZ_dxdot', val=np.full(nn, 49.0), units='lbf*s/ft')
        self.add_input('dZ_dzdot', val=np.full(nn, -287.0), units='lbf*s/ft')
        self.add_input('G_W', val=np.full(nn, 20000.0), units='lbf')
        self.add_input('I_yy', val=np.full(nn, 40000.0), units='slug*ft**2')
        self.add_input('g', val=np.full(nn, 32.2), units='ft/s**2')

        self.add_output('E', val=np.zeros(nn),
                        desc='Constant term of the characteristic equation.')
        self.add_output('dM_dzdot_at_E_zero', val=np.zeros(nn), units='lbf*s',
                        desc='Divergence boundary of Figure 9.15.')
        self.add_output('E_margin', val=np.zeros(nn), units='lbf*s',
                        desc='dM/dzdot minus the boundary value.')

        self.declare_partials('E', ['dM_dxdot', 'dM_dzdot', 'dZ_dxdot',
                                    'dZ_dzdot', 'G_W', 'I_yy', 'g'],
                              rows=ar, cols=ar)
        for name in ('dM_dzdot_at_E_zero', 'E_margin'):
            self.declare_partials(name, ['dM_dxdot', 'dZ_dxdot', 'dZ_dzdot'],
                                  rows=ar, cols=ar)
        self.declare_partials('E_margin', 'dM_dzdot', rows=ar, cols=ar,
                              val=1.0)

    def compute(self, inputs, outputs):
        mass = inputs['G_W'] / inputs['g']
        numerator = (inputs['dZ_dxdot'] * inputs['dM_dzdot']
                     - inputs['dZ_dzdot'] * inputs['dM_dxdot'])
        boundary = inputs['dM_dxdot'] * inputs['dZ_dzdot'] / inputs['dZ_dxdot']

        outputs['E'] = inputs['G_W'] * numerator / (mass ** 2 * inputs['I_yy'])
        outputs['dM_dzdot_at_E_zero'] = boundary
        outputs['E_margin'] = inputs['dM_dzdot'] - boundary

    def compute_partials(self, inputs, J):
        G_W, g, I_yy = inputs['G_W'], inputs['g'], inputs['I_yy']
        Zx, Zz = inputs['dZ_dxdot'], inputs['dZ_dzdot']
        Mx, Mz = inputs['dM_dxdot'], inputs['dM_dzdot']

        scale = g ** 2 / (G_W * I_yy)               # G_W / (m^2 I_yy)
        numerator = Zx * Mz - Zz * Mx

        J['E', 'dM_dzdot'] = scale * Zx
        J['E', 'dM_dxdot'] = -scale * Zz
        J['E', 'dZ_dxdot'] = scale * Mz
        J['E', 'dZ_dzdot'] = -scale * Mx
        J['E', 'G_W'] = -scale * numerator / G_W
        J['E', 'I_yy'] = -scale * numerator / I_yy
        J['E', 'g'] = 2.0 * scale * numerator / g

        J['dM_dzdot_at_E_zero', 'dM_dxdot'] = Zz / Zx
        J['dM_dzdot_at_E_zero', 'dZ_dzdot'] = Mx / Zx
        J['dM_dzdot_at_E_zero', 'dZ_dxdot'] = -Mx * Zz / Zx ** 2
        for name in ('dM_dxdot', 'dZ_dzdot', 'dZ_dxdot'):
            J['E_margin', name] = -J['dM_dzdot_at_E_zero', name]
