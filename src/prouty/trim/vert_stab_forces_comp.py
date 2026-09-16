"""Vertical stabiliser lift and drag resolved into body axes.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 502, Figure 8.18. Anchors in
Table 8.4 pp. 518-519 and Table 8.11 p. 536.
"""

import numpy as np
import openmdao.api as om


class VertStabForcesComp(om.ExplicitComponent):
    """X, Y and Z forces from the vertical stabiliser, p. 502.

    With ``psi_V = beta + eta_MV + eta_TV + eta_FV`` the flow direction in
    plan view and ``phi_V = Theta - (eps_MV + eps_FV + gamma_c)`` the flow
    direction in side view::

        X_V = -D_V cos(psi_V) - L_V sin(psi_V)
        Y_V =  L_V cos(psi_V) - D_V sin(psi_V)
        Z_V =  X_V sin(phi_V)

    ``Z_V`` is the odd one. The fin produces no lift in the vertical plane,
    so its only contribution to the Z equation is its own X-force tilted by
    the local flow angle in side view — hence ``X_V`` appearing inside it
    rather than ``L_V`` and ``D_V`` separately.

    Two different downwash pairs appear here and they are not the same
    numbers as the sidewash pair. ``eps_MV`` and ``eps_FV`` are the
    *longitudinal* downwash at the fin, entering ``phi_V``;
    ``eta_MV``, ``eta_TV``, ``eta_FV`` are the *lateral* sidewash, already
    summed into ``psi_V`` upstream. Table 8.5 credits Figure 8.15 for both
    the longitudinal and the lateral fuselage slope, at 0.23 and 0.06; see
    C8-5 in ``docs/validation_trim.md``.

    Linearised form, ``linearized=True``
    ------------------------------------
    Sines to angles, cosines to one, and the forces frozen::

        X_V = -D_V - L_V psi_V
        Y_V =  L_V - D_V psi_V
        Z_V =  X_V phi_V

    Table 8.4 p. 518 prints the ``X_V`` row as
    ``-D_V - L_V(eta_MV + eta_TV)``, without ``beta`` or ``eta_FV``: not a
    third form, only the same expression with the lateral unknowns set to
    zero, since they are not unknowns of the longitudinal problem.

    Anchor
    ------
    At 115 knots with ``beta = 0``, ``psi_V = -0.0059`` and
    ``phi_V = -0.0880``, and with ``L_V = 287`` and ``D_V = 58`` from
    Table 8.5:

    ===== ========= ================================
    X_V   -56.6      Table 8.4 rows sum to -56
    Y_V   289.0      Table 8.11, essentially L_V
    Z_V     4.93     Table 8.4 rows sum to 3.77
    ===== ========= ================================

    The ``Z_V`` gap is arithmetic, not physical. Rebuilding that row from
    p. 502 gives ``1.344 - 43.11 Theta + .00013975 T_M``; Table 8.4 prints
    the last coefficient as .0001, and one significant figure on a term
    worth 2.9 lb accounts for the whole 1.2 lb difference. Same class as
    C8-2.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('L_V', shape=(nn,), val=0.0, units='lbf',
                       desc='fin lift')
        self.add_input('D_V', shape=(nn,), val=0.0, units='lbf',
                       desc='fin drag')
        self.add_input('psi_V', shape=(nn,), val=0.0, units='rad',
                       desc='total sidewash at the fin')
        self.add_input('Theta', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage pitch attitude')
        self.add_input('gamma_c', shape=(nn,), val=0.0, units='rad',
                       desc='climb angle')
        self.add_input('eps_MV', shape=(nn,), val=0.0, units='rad',
                       desc='main rotor downwash at the fin')
        self.add_input('eps_FV', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage downwash at the fin')

        for name in ('X_V', 'Y_V', 'Z_V'):
            self.add_output(name, shape=(nn,), units='lbf')

        self.declare_partials(['X_V', 'Y_V'], ['L_V', 'D_V', 'psi_V'],
                              rows=ar, cols=ar)
        self.declare_partials('Z_V',
                              ['L_V', 'D_V', 'psi_V', 'Theta', 'gamma_c',
                               'eps_MV', 'eps_FV'], rows=ar, cols=ar)

    def _angles(self, inputs):
        phi = (inputs['Theta'] - inputs['gamma_c']
               - inputs['eps_MV'] - inputs['eps_FV'])
        return inputs['psi_V'], phi

    def compute(self, inputs, outputs):
        L, D = inputs['L_V'], inputs['D_V']
        psi, phi = self._angles(inputs)

        if self.options['linearized']:
            X = -D - L * psi
            outputs['Y_V'] = L - D * psi
            outputs['Z_V'] = X * phi
        else:
            sp, cp = np.sin(psi), np.cos(psi)
            X = -D * cp - L * sp
            outputs['Y_V'] = L * cp - D * sp
            outputs['Z_V'] = X * np.sin(phi)
        outputs['X_V'] = X

    def compute_partials(self, inputs, J):
        L, D = inputs['L_V'], inputs['D_V']
        psi, phi = self._angles(inputs)
        ones = np.ones_like(L)

        if self.options['linearized']:
            X = -D - L * psi
            J['X_V', 'L_V'], J['X_V', 'D_V'] = -psi, -ones
            J['X_V', 'psi_V'] = -L
            J['Y_V', 'L_V'], J['Y_V', 'D_V'] = ones, -psi
            J['Y_V', 'psi_V'] = -D
            tilt, dtilt = phi, ones
        else:
            sp, cp = np.sin(psi), np.cos(psi)
            X = -D * cp - L * sp
            J['X_V', 'L_V'], J['X_V', 'D_V'] = -sp, -cp
            J['X_V', 'psi_V'] = D * sp - L * cp
            J['Y_V', 'L_V'], J['Y_V', 'D_V'] = cp, -sp
            J['Y_V', 'psi_V'] = -L * sp - D * cp
            tilt, dtilt = np.sin(phi), np.cos(phi)

        for name in ('L_V', 'D_V', 'psi_V'):
            J['Z_V', name] = J['X_V', name] * tilt

        # phi = Theta - gamma_c - eps_MV - eps_FV
        for name, sign in (('Theta', 1.0), ('gamma_c', -1.0),
                           ('eps_MV', -1.0), ('eps_FV', -1.0)):
            J['Z_V', name] = sign * X * dtilt
