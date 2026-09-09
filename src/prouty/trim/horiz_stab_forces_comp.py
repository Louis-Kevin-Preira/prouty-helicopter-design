"""Horizontal stabiliser lift and drag resolved into body axes.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 488, Figure 8.5. Anchors in
Table 8.4 pp. 518-521.
"""

import numpy as np
import openmdao.api as om


class HorizStabForcesComp(om.ExplicitComponent):
    """X and Z forces from the horizontal stabiliser, p. 488.

    With ``phi = Theta - (eps_MH + eps_FH + gamma_c)``::

        X_H =  L_H sin(phi) - D_H cos(phi)
        Z_H = -L_H cos(phi) - D_H sin(phi)

    ``phi`` is the local flow direction seen from the body axis, and it is
    ``alpha_H - i_H``: the same angle as p. 489 with the stabiliser's own
    incidence taken back out, since the forces resolve onto the airframe and
    not onto the chord line. It is taken from ``Theta``, the downwashes and
    the climb angle directly, as p. 488 writes it, rather than from
    ``alpha_H`` — feeding ``alpha_H`` back in would make ``i_H`` cancel
    through two components instead of never entering.

    Linearised form, ``linearized=True``
    ------------------------------------
    Table 8.4 pp. 518-519 puts ``sin(phi) -> phi`` and ``cos(phi) -> 1`` and
    freezes the lift and drag at their Chapter 3 trim values::

        X_H =  L_H phi - D_H
        Z_H = -L_H - D_H phi

    Anchor
    ------
    At 115 knots, ``Theta = -0.0165``, ``gamma_c = 0``,
    ``eps_MH + eps_FH = 0.0717``, so ``phi = -0.0881 rad``. With
    ``L_H = -273`` and ``D_H = 15`` from Table 8.5:

    ===== ========= ==========
    X_H     9.1
    Z_H   270.7
    ===== ========= ==========

    Table 8.4 splits each into a constant and a part linear in ``Theta`` and
    ``T_M``, and its rows do not close on these. The ``Z_H`` rows do: their
    coefficients rebuild exactly from ``q_H/q = 0.6``, ``A_H = 18``,
    ``a_H = 4.0`` and the downwash parameters of Table 8.5, giving -1497 for
    the ``Theta`` coefficient against the printed -1497. The ``X_H`` rows do
    not: all three of their printed coefficients are consistent with a trim
    lift of about -305 lb rather than the -273 of Table 8.5. That is an open
    question for the X-equilibrium component, recorded in
    ``docs/validation_trim.md``; it is not settled here, and this component
    computes from whatever ``L_H`` it is handed.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('L_H', shape=(nn,), val=0.0, units='lbf',
                       desc='stabiliser lift')
        self.add_input('D_H', shape=(nn,), val=0.0, units='lbf',
                       desc='stabiliser drag')
        self.add_input('Theta', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage pitch attitude')
        self.add_input('gamma_c', shape=(nn,), val=0.0, units='rad',
                       desc='climb angle')
        self.add_input('eps_MH', shape=(nn,), val=0.0, units='rad',
                       desc='main rotor downwash at the stabiliser')
        self.add_input('eps_FH', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage downwash at the stabiliser')

        self.add_output('X_H', shape=(nn,), units='lbf')
        self.add_output('Z_H', shape=(nn,), units='lbf')

        wrt = ['L_H', 'D_H', 'Theta', 'gamma_c', 'eps_MH', 'eps_FH']
        for out in ('X_H', 'Z_H'):
            self.declare_partials(out, wrt, rows=ar, cols=ar)

    def _phi(self, inputs):
        return (inputs['Theta'] - inputs['gamma_c']
                - inputs['eps_MH'] - inputs['eps_FH'])

    def compute(self, inputs, outputs):
        L, D, phi = inputs['L_H'], inputs['D_H'], self._phi(inputs)

        if self.options['linearized']:
            outputs['X_H'] = L * phi - D
            outputs['Z_H'] = -L - D * phi
            return

        s, c = np.sin(phi), np.cos(phi)
        outputs['X_H'] = L * s - D * c
        outputs['Z_H'] = -L * c - D * s

    def compute_partials(self, inputs, J):
        L, D, phi = inputs['L_H'], inputs['D_H'], self._phi(inputs)

        if self.options['linearized']:
            J['X_H', 'L_H'], J['X_H', 'D_H'] = phi, -np.ones_like(phi)
            J['Z_H', 'L_H'], J['Z_H', 'D_H'] = -np.ones_like(phi), -phi
            dX_dphi, dZ_dphi = L, -D
        else:
            s, c = np.sin(phi), np.cos(phi)
            J['X_H', 'L_H'], J['X_H', 'D_H'] = s, -c
            J['Z_H', 'L_H'], J['Z_H', 'D_H'] = -c, -s
            dX_dphi = L * c + D * s
            dZ_dphi = L * s - D * c

        # phi = Theta - gamma_c - eps_MH - eps_FH
        for name, sign in (('Theta', 1.0), ('gamma_c', -1.0),
                           ('eps_MH', -1.0), ('eps_FH', -1.0)):
            J['X_H', name] = sign * dX_dphi
            J['Z_H', name] = sign * dZ_dphi
