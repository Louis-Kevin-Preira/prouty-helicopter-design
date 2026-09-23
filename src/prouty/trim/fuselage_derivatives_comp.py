"""Fuselage forces and moments from linear aerodynamic derivatives.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", pp. 513-515, Figures 8.24 and 8.25
pp. 514-515. Parameters and anchors in Table 8.5 pp. 524-525, Table 8.4
p. 519 and Table 8.11 pp. 536-537.
"""

import numpy as np
import openmdao.api as om


class FuselageDerivativesComp(om.ExplicitComponent):
    """Fuselage aerodynamics as constants and slopes, pp. 513-515::

        L_F    = q [ (L/q)_0    + (dL/q/dalpha_F)  alpha_F ]
        M_F    = q [ (M/q)_0    + (dM/q/dalpha_F)  alpha_F ]
        D_F    = q f
        SF_F   = q   (dSF/q/dbeta)  beta
        N_F    = q   (dN/q/dbeta)   beta
        R_F    = q   (dR/q/dbeta)   beta

    Longitudinally a constant and a slope, laterally a slope alone: p. 513
    takes the fuselage to produce no side force, yawing moment or rolling
    moment at zero sideslip, which a symmetric airframe must satisfy.

    Drag stays outside the derivative model. p. 513 sends it to the methods
    of Chapter 4, and Table 8.5 lists no ``(D/q)`` entry, so ``f`` arrives
    as an equivalent flat plate area and ``D_F = q f``.

    p. 513 is blunt about the rolling moment. Fuselage dihedral effect is
    "strongly, but mysteriously" influenced by the interference of the rotor
    wake on the tail rotor and the fin, so estimating it has a very low
    priority and the estimation method of reference 8.1 is not repeated. The
    slope is here because the equations need one, not because it is known.

    Do not feed this from Chapter 3
    -------------------------------
    ``prouty.forward_flight.FuselageAeroComp`` reads Figure A.2 p. 679, whose
    curves are labelled *empennage on*. Chapter 8 models the horizontal
    stabiliser as its own component, so using them here would count that
    surface twice. The numbers say so plainly: the calibrated Figure A.2 lift
    slope is 1.953 ft^2/deg, the horizontal stabiliser contributes
    ``(q_H/q) A_H a_H = 43.2 ft^2/rad = 0.754 ft^2/deg``, and the difference,
    1.199, is within 9 % of the 1.309 ft^2/deg that Table 8.5 gives for the
    bare fuselage. See C8-7 in ``docs/validation_trim.md``.

    Defaults are the example helicopter, Table 8.5 pp. 524-525, all credited
    to Appendix A:

    ========================= ============ ===========
    ``(L/q)_0``                -1.5          ft^2
    ``dL/q/dalpha_F``          75            ft^2/rad
    ``(M/q)_0``                -160          ft^3
    ``dM/q/dalpha_F``          1780          ft^3/rad
    ``dSF/q/dbeta``            -220          ft^2/rad
    ``dR/q/dbeta``             230           ft^3/rad
    ``dN/q/dbeta``             -820          ft^3/rad
    ========================= ============ ===========

    Anchors
    -------
    At 115 knots with ``alpha_F = -0.0569 rad``, the pitching moment is
    -11,757 ft-lb against the -11,722 of Table 8.5, 0.3 %. The lift is
    -259.5 lb against a printed -281, 7.6 %: Table 8.5's own resultant
    implies ``alpha_F = -3.62 deg`` where the same table's inputs give
    -3.26 deg, so the gap is in the book's iteration, not in the model.

    Table 8.11 confirms the three lateral slopes to the digit:
    ``q (dSF/q/dbeta) = -9900``, ``q (dN/q/dbeta) = -36,900`` and
    ``q (dR/q/dbeta) = 10,350``, all as printed.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('alpha_F', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage angle of attack')
        self.add_input('beta', shape=(nn,), val=0.0, units='rad',
                       desc='sideslip angle')
        self.add_input('q', shape=(nn,), units='lbf/ft**2',
                       desc='dynamic pressure')
        self.add_input('f', shape=(nn,), val=0.0, units='ft**2',
                       desc='equivalent flat plate area, Chapter 4')

        self.add_input('LF_q_0', val=-1.5, units='ft**2')
        self.add_input('dLF_q_dalpha', val=75.0, units='ft**2/rad')
        self.add_input('MF_q_0', val=-160.0, units='ft**3')
        self.add_input('dMF_q_dalpha', val=1780.0, units='ft**3/rad')
        self.add_input('dSF_q_dbeta', val=-220.0, units='ft**2/rad')
        self.add_input('dRF_q_dbeta', val=230.0, units='ft**3/rad')
        self.add_input('dNF_q_dbeta', val=-820.0, units='ft**3/rad')

        for name, unit in (('L_F', 'lbf'), ('D_F', 'lbf'), ('SF_F', 'lbf'),
                           ('M_F', 'lbf*ft'), ('N_F', 'lbf*ft'),
                           ('R_F', 'lbf*ft')):
            self.add_output(name, shape=(nn,), units=unit)

        # long. outputs: constant + slope on alpha_F
        self._long = (('L_F', 'LF_q_0', 'dLF_q_dalpha'),
                      ('M_F', 'MF_q_0', 'dMF_q_dalpha'))
        # lat. outputs: slope on beta alone
        self._lat = (('SF_F', 'dSF_q_dbeta'), ('N_F', 'dNF_q_dbeta'),
                     ('R_F', 'dRF_q_dbeta'))

        for out, const, slope in self._long:
            self.declare_partials(out, ['alpha_F', 'q'], rows=ar, cols=ar)
            self.declare_partials(out, [const, slope], rows=ar, cols=zeros)
        for out, slope in self._lat:
            self.declare_partials(out, ['beta', 'q'], rows=ar, cols=ar)
            self.declare_partials(out, slope, rows=ar, cols=zeros)
        self.declare_partials('D_F', ['q', 'f'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        q, alpha, beta = inputs['q'], inputs['alpha_F'], inputs['beta']

        for out, const, slope in self._long:
            outputs[out] = q * (inputs[const][0] + inputs[slope][0] * alpha)
        for out, slope in self._lat:
            outputs[out] = q * inputs[slope][0] * beta
        outputs['D_F'] = q * inputs['f']

    def compute_partials(self, inputs, J):
        q, alpha, beta = inputs['q'], inputs['alpha_F'], inputs['beta']

        for out, const, slope in self._long:
            J[out, 'alpha_F'] = q * inputs[slope][0]
            J[out, 'q'] = inputs[const][0] + inputs[slope][0] * alpha
            J[out, const] = q
            J[out, slope] = q * alpha
        for out, slope in self._lat:
            J[out, 'beta'] = q * inputs[slope][0]
            J[out, 'q'] = inputs[slope][0] * beta
            J[out, slope] = q * beta
        J['D_F', 'q'] = inputs['f']
        J['D_F', 'f'] = q
