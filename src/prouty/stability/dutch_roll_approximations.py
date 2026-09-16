"""Approximations to the Dutch roll mode.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 630-632, after the airplane
treatment of reference 9.15.
"""

import numpy as np
import openmdao.api as om

DERIVATIVE_UNITS = {
    'dR_dydot': 'lbf*s', 'dR_dp': 'lbf*ft*s/rad', 'dR_dr': 'lbf*ft*s/rad',
    'dN_dydot': 'lbf*s', 'dN_dp': 'lbf*ft*s/rad', 'dN_dr': 'lbf*ft*s/rad',
}


class DutchRollMatrixComp(om.ExplicitComponent):
    """The reduced determinant of p. 630.

    The first assumption is that the aircraft rolls and yaws but its centre of
    gravity holds a straight flight path, which strips the Y equation to its
    inertial terms::

        -(G.W./g) y_ddot - (G.W./g) V_bar r = 0    ->    y_ddot + V_bar r = 0

    leaving, in the states ``ydot``, ``p``, ``r``::

        | s          0                    V_bar               |
        | dR/dydot   -I_xx s + dR/dp      dR/dr               |
        | dN/dydot   dN/dp                -I_zz s + dN/dr     |

    Feed it to ``PolyDeterminantComp(n=3, degree=1)``; the determinant is a
    cubic with no rigid-body factor to remove.

    Expanding it gives p. 631's cubic **exactly** -- every term, every sign.
    Unlike the phugoid reduction of p. 623, nothing here is misprinted.

    Options
    -------
    num_nodes : int
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        shape = (nn, 3, 3, 2)

        for name, units in DERIVATIVE_UNITS.items():
            self.add_input(name, val=np.zeros(nn), units=units)
        self.add_input('I_xx', val=np.full(nn, 5000.0), units='slug*ft**2')
        self.add_input('I_zz', val=np.full(nn, 35000.0), units='slug*ft**2')
        self.add_input('V', val=np.full(nn, 194.1), units='ft/s')

        self.add_output('matrix_coeffs', shape=shape)

        def flat(row, col, power):
            return np.ravel_multi_index(
                (ar, np.full(nn, row), np.full(nn, col), np.full(nn, power)),
                shape)

        entries = {(1, 0, 0): 'dR_dydot', (1, 1, 0): 'dR_dp',
                   (1, 2, 0): 'dR_dr', (2, 0, 0): 'dN_dydot',
                   (2, 1, 0): 'dN_dp', (2, 2, 0): 'dN_dr'}
        self._entries = entries
        for (row, col, power), name in entries.items():
            self.declare_partials('matrix_coeffs', name,
                                  rows=flat(row, col, power), cols=ar, val=1.0)
        self.declare_partials('matrix_coeffs', 'I_xx', rows=flat(1, 1, 1),
                              cols=ar, val=-1.0)
        self.declare_partials('matrix_coeffs', 'I_zz', rows=flat(2, 2, 1),
                              cols=ar, val=-1.0)
        self.declare_partials('matrix_coeffs', 'V', rows=flat(0, 2, 0),
                              cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        mat = np.zeros((nn, 3, 3, 2), dtype=inputs._get_data().dtype)

        for (row, col, power), name in self._entries.items():
            mat[:, row, col, power] = inputs[name]

        mat[:, 0, 0, 1] = 1.0
        mat[:, 0, 2, 0] = inputs['V']
        mat[:, 1, 1, 1] = -inputs['I_xx']
        mat[:, 2, 2, 1] = -inputs['I_zz']

        outputs['matrix_coeffs'] = mat


class DutchRollApproxComp(om.ExplicitComponent):
    """The two closed-form Dutch roll quadratics, pp. 631-632.

    p. 631 takes the cubic of :class:`DutchRollMatrixComp` and applies two
    more assumptions. First that ``(dN/dr)/I_zz`` is small beside
    ``(dR/dp)/I_xx`` in the ``s^2`` coefficient, then the Bairstow
    approximation -- for a lightly damped cubic ``c3 s^3 + c2 s^2 + c1 s +
    c0``, take ``s^2 = -c0/c2`` and the cubic collapses to
    ``c2 s^2 + (c1 - c3 c0/c2)s + c0 = 0``. What comes out is::

        s^2 - (1/I_zz)[dN/dr - (dR/dr)(dN/dp)/(dR/dp)
                       + V_bar I_xx (dR/dydot)(dN/dp)/(dR/dp)^2] s
            + (V_bar/I_zz)[dN/dydot - (dR/dydot)(dN/dp)/(dR/dp)] = 0

    p. 632 then drops everything but yaw damping and directional stability::

        s^2 - (1/I_zz)(dN/dr) s + (V_bar/I_zz)(dN/dydot) = 0

    Both are emitted, as ``char_coeffs_bairstow`` and ``char_coeffs_simple``,
    in the ascending-power convention the rest of the package uses.

    The Bairstow step is exact algebra
    ----------------------------------
    Substituting `c2 = -I_zz (dR/dp)` into `c1 - c3 c0/c2`, the two
    `V_bar I_xx (dN/dydot)` terms cancel identically and what is left is the
    bracket above. The approximation is in the two assumptions, not in the
    manipulation.

    How good is "much less than"?
    -----------------------------
    p. 631's first assumption is that ``(dN/dr)/I_zz << (dR/dp)/I_xx``. For the
    example helicopter those are **-1.54 and -6.75**: a ratio of 0.23, not a
    negligible one. It survives because the Bairstow step is what carries the
    frequency, and the assumption only perturbs the damping.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    ============================ ========================= ================
    equation                     roots                     book
    ============================ ========================= ================
    full quartic, p. 628         -.7841 +/- 2.4309i        -.7841 +/- 2.4317i
    Bairstow, p. 631             -.7821 +/- 2.3766i        -.7823 +/- 2.3826i
    yaw only, p. 632             -.7702 +/- 2.4700i        -.7702 +/- 2.4662i
    ============================ ========================= ================

    All three give a period between 2.5 and 2.7 seconds and a real part within
    2 % of each other. p. 632: "thus again little accuracy has been lost in the
    simplification".
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        for name, units in DERIVATIVE_UNITS.items():
            self.add_input(name, val=np.zeros(nn), units=units)
        self.add_input('I_xx', val=np.full(nn, 5000.0), units='slug*ft**2')
        self.add_input('I_zz', val=np.full(nn, 35000.0), units='slug*ft**2')
        self.add_input('V', val=np.full(nn, 194.1), units='ft/s')

        for name in ('char_coeffs_bairstow', 'char_coeffs_simple'):
            self.add_output(name, shape=(nn, 3))

        rows = np.concatenate([3 * ar, 3 * ar + 1])
        cols = np.tile(ar, 2)
        self.declare_partials('char_coeffs_bairstow',
                              ['dR_dydot', 'dR_dp', 'dR_dr', 'dN_dydot',
                               'dN_dp', 'dN_dr', 'I_xx', 'I_zz', 'V'],
                              rows=rows, cols=cols)
        self.declare_partials('char_coeffs_simple',
                              ['dN_dydot', 'dN_dr', 'I_zz', 'V'],
                              rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        Ry, Rp, Rr = inputs['dR_dydot'], inputs['dR_dp'], inputs['dR_dr']
        Ny, Np, Nr = inputs['dN_dydot'], inputs['dN_dp'], inputs['dN_dr']
        I_xx, I_zz, V = inputs['I_xx'], inputs['I_zz'], inputs['V']

        damping = Nr - Rr * Np / Rp + V * I_xx * Ry * Np / Rp ** 2
        stiffness = Ny - Ry * Np / Rp

        bairstow = np.zeros((nn, 3), dtype=inputs._get_data().dtype)
        bairstow[:, 0] = V * stiffness / I_zz
        bairstow[:, 1] = -damping / I_zz
        bairstow[:, 2] = 1.0
        outputs['char_coeffs_bairstow'] = bairstow

        simple = np.zeros((nn, 3), dtype=inputs._get_data().dtype)
        simple[:, 0] = V * Ny / I_zz
        simple[:, 1] = -Nr / I_zz
        simple[:, 2] = 1.0
        outputs['char_coeffs_simple'] = simple

    def compute_partials(self, inputs, J):
        Ry, Rp, Rr = inputs['dR_dydot'], inputs['dR_dp'], inputs['dR_dr']
        Ny, Np, Nr = inputs['dN_dydot'], inputs['dN_dp'], inputs['dN_dr']
        I_xx, I_zz, V = inputs['I_xx'], inputs['I_zz'], inputs['V']

        damping = Nr - Rr * Np / Rp + V * I_xx * Ry * Np / Rp ** 2
        stiffness = Ny - Ry * Np / Rp

        # d(stiffness)/dx, then the constant term V stiffness / I_zz
        d_stiff = {'dN_dydot': np.ones_like(Ny), 'dR_dydot': -Np / Rp,
                   'dN_dp': -Ry / Rp, 'dR_dp': Ry * Np / Rp ** 2}
        # d(damping)/dx, then the s term -damping / I_zz
        d_damp = {
            'dN_dr': np.ones_like(Nr), 'dR_dr': -Np / Rp,
            'dN_dp': -Rr / Rp + V * I_xx * Ry / Rp ** 2,
            'dR_dp': Rr * Np / Rp ** 2 - 2.0 * V * I_xx * Ry * Np / Rp ** 3,
            'dR_dydot': V * I_xx * Np / Rp ** 2,
            'V': I_xx * Ry * Np / Rp ** 2,
            'I_xx': V * Ry * Np / Rp ** 2,
        }

        for name in ('dR_dydot', 'dR_dp', 'dR_dr', 'dN_dydot', 'dN_dp',
                     'dN_dr', 'I_xx', 'I_zz', 'V'):
            constant = V * d_stiff.get(name, 0.0) / I_zz
            linear = -d_damp.get(name, 0.0) / I_zz
            if name == 'V':
                constant = constant + stiffness / I_zz
            if name == 'I_zz':
                constant = -V * stiffness / I_zz ** 2
                linear = damping / I_zz ** 2
            J['char_coeffs_bairstow', name] = np.concatenate(
                [np.broadcast_to(constant, Ny.shape),
                 np.broadcast_to(linear, Ny.shape)])

        zero = np.zeros_like(Ny)
        J['char_coeffs_simple', 'dN_dydot'] = np.concatenate([V / I_zz, zero])
        J['char_coeffs_simple', 'dN_dr'] = np.concatenate(
            [zero, -np.ones_like(Nr) / I_zz])
        J['char_coeffs_simple', 'V'] = np.concatenate([Ny / I_zz, zero])
        J['char_coeffs_simple', 'I_zz'] = np.concatenate(
            [-V * Ny / I_zz ** 2, Nr / I_zz ** 2])
