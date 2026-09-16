"""Main rotor contribution to the six equations of equilibrium.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 485, Figure 8.3 and Table 8.1.
Linearised forms from Table 8.4, pp. 518-521, and Table 8.11, pp. 536-537.
"""

import numpy as np
import openmdao.api as om


class MainRotorForcesComp(om.ExplicitComponent):
    """Forces and moments the main rotor applies to the airframe.

    Nonlinear form, p. 485, for a rotor whose advancing blade is on the
    right, with ``X = a1s_M + i_M``::

        X_M = -H cos X - T_M sin X
        Y_M =  T_M sin b1s_M - H sin beta
        Z_M = -T_M cos X
        R_M = (dR/db1s)_M b1s_M + Q_M sin X
        M_M = (dM/da1s)_M a1s_M - Q_M sin b1s_M
        N_M =  Q_M cos X cos b1s_M

    ``H`` is the H-force at *zero* flapping, Prouty's ``H_{a1s=0,M}``: the
    tilt of the thrust vector is carried by the sine terms, so counting a
    flapping-dependent H-force here would count it twice. Chapter 3 gives it
    by evaluating ``HForceCoefComp`` at ``a1s = 0``.

    Linearised form, ``linearized=True``
    ------------------------------------
    Table 8.4 (X, Z) and Table 8.11 (Y, R, N) drop to small angles and freeze
    the products of unknowns at their Chapter 3 trim values::

        X_M = -H - T_M_bar a1s_M - T_M i_M
        Y_M =  T_M_bar b1s_M - (H + T_M_bar a1s_M + T_M_bar i_M) beta
        Z_M = -T_M
        R_M = (dR/db1s)_M b1s_M
        M_M = (dM/da1s)_M a1s_M
        N_M =  Q_M

    Two things are worth seeing in that table. The thrust appearing with
    ``a1s_M`` is the barred one and the thrust appearing with ``i_M`` is the
    unknown, which is what keeps the X equation linear in ``T_M``. And the
    ``a1s_M`` inside the Y equation is the *converged longitudinal* flapping,
    not a lateral unknown: freezing it is precisely what lets pp. 516 and 531
    treat the two trim problems as independent sets.

    The example helicopter at 115 knots, Tables 8.4 and 8.11, with
    ``H = -145``, ``T_M_bar = 20606``, ``i_M = 0``, ``Q_M = 34726`` and the
    stiffnesses of Chapter 7:

    ==== ====================================
    X_M   145 - 20,606 a1s_M
    Y_M   20,606 b1s_M + 537 beta
    Z_M   -T_M
    R_M   200,940 b1s_M
    M_M   200,940 a1s_M
    N_M   34,726
    ==== ====================================

    Options
    -------
    num_nodes : int
    linearized : bool
        ``True`` reproduces Tables 8.4 and 8.11 term for term.

    Notes
    -----
    ``dRM_db1s`` and ``dMM_da1s`` are separate inputs although the derivation
    of p. 477 makes them equal, since it averages ``cos^2`` in one case and
    ``sin^2`` in the other. Keeping two lets a rotor with unequal pitch and
    roll stiffness be described without touching this component.

    ``i_M`` is Chapter 3's ``i_s`` under the name Chapter 8 uses.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('T_M', shape=(nn,), units='lbf', desc='rotor thrust')
        self.add_input('a1s_M', shape=(nn,), val=0.0, units='rad',
                       desc='longitudinal flapping')
        self.add_input('b1s_M', shape=(nn,), val=0.0, units='rad',
                       desc='lateral flapping')
        self.add_input('beta', shape=(nn,), val=0.0, units='rad',
                       desc='sideslip angle')
        self.add_input('H_a1s0', shape=(nn,), val=0.0, units='lbf',
                       desc='H-force at zero flapping')
        self.add_input('Q_M', shape=(nn,), val=0.0, units='lbf*ft',
                       desc='rotor torque')
        self.add_input('dRM_db1s', shape=(nn,), val=0.0, units='lbf*ft/rad',
                       desc='hub rolling moment stiffness')
        self.add_input('dMM_da1s', shape=(nn,), val=0.0, units='lbf*ft/rad',
                       desc='hub pitching moment stiffness')
        self.add_input('i_M', val=0.0, units='rad', desc='shaft incidence')

        for name, unit in (('X_M', 'lbf'), ('Y_M', 'lbf'), ('Z_M', 'lbf'),
                           ('R_M', 'lbf*ft'), ('M_M', 'lbf*ft'),
                           ('N_M', 'lbf*ft')):
            self.add_output(name, shape=(nn,), units=unit)

        if self.options['linearized']:
            self.add_input('T_M_bar', shape=(nn,), units='lbf',
                           desc='trim thrust, frozen in products of unknowns')
            wrt = {'X_M': ['H_a1s0', 'a1s_M', 'T_M_bar', 'T_M'],
                   'Y_M': ['H_a1s0', 'a1s_M', 'b1s_M', 'beta', 'T_M_bar'],
                   'Z_M': ['T_M'],
                   'R_M': ['b1s_M', 'dRM_db1s'],
                   'M_M': ['a1s_M', 'dMM_da1s'],
                   'N_M': ['Q_M']}
            self.declare_partials('X_M', 'i_M', rows=ar, cols=zeros)
            self.declare_partials('Y_M', 'i_M', rows=ar, cols=zeros)
        else:
            wrt = {'X_M': ['H_a1s0', 'T_M', 'a1s_M'],
                   'Y_M': ['H_a1s0', 'T_M', 'b1s_M', 'beta'],
                   'Z_M': ['T_M', 'a1s_M'],
                   'R_M': ['b1s_M', 'dRM_db1s', 'Q_M', 'a1s_M'],
                   'M_M': ['a1s_M', 'dMM_da1s', 'Q_M', 'b1s_M'],
                   'N_M': ['Q_M', 'a1s_M', 'b1s_M']}
            for out in ('X_M', 'Z_M', 'R_M', 'N_M'):
                self.declare_partials(out, 'i_M', rows=ar, cols=zeros)

        for out, names in wrt.items():
            self.declare_partials(out, names, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        T, a1s, b1s = inputs['T_M'], inputs['a1s_M'], inputs['b1s_M']
        H, Q, beta, i_M = (inputs['H_a1s0'], inputs['Q_M'], inputs['beta'],
                           inputs['i_M'][0])

        outputs['R_M'] = inputs['dRM_db1s'] * b1s
        outputs['M_M'] = inputs['dMM_da1s'] * a1s

        if self.options['linearized']:
            T_bar = inputs['T_M_bar']
            outputs['X_M'] = -H - T_bar * a1s - T * i_M
            outputs['Y_M'] = T_bar * b1s - (H + T_bar * (a1s + i_M)) * beta
            outputs['Z_M'] = -T
            outputs['N_M'] = Q
            return

        sx, cx = np.sin(a1s + i_M), np.cos(a1s + i_M)
        outputs['X_M'] = -H * cx - T * sx
        outputs['Y_M'] = T * np.sin(b1s) - H * np.sin(beta)
        outputs['Z_M'] = -T * cx
        outputs['R_M'] += Q * sx
        outputs['M_M'] -= Q * np.sin(b1s)
        outputs['N_M'] = Q * cx * np.cos(b1s)

    def compute_partials(self, inputs, J):
        T, a1s, b1s = inputs['T_M'], inputs['a1s_M'], inputs['b1s_M']
        H, Q, beta, i_M = (inputs['H_a1s0'], inputs['Q_M'], inputs['beta'],
                           inputs['i_M'][0])
        ones = np.ones_like(T)

        J['R_M', 'b1s_M'] = inputs['dRM_db1s']
        J['R_M', 'dRM_db1s'] = b1s
        J['M_M', 'a1s_M'] = inputs['dMM_da1s']
        J['M_M', 'dMM_da1s'] = a1s

        if self.options['linearized']:
            T_bar = inputs['T_M_bar']
            J['X_M', 'H_a1s0'] = -ones
            J['X_M', 'a1s_M'] = -T_bar
            J['X_M', 'T_M_bar'] = -a1s
            J['X_M', 'T_M'] = -i_M * ones
            J['X_M', 'i_M'] = -T

            J['Y_M', 'H_a1s0'] = -beta
            J['Y_M', 'a1s_M'] = -T_bar * beta
            J['Y_M', 'b1s_M'] = T_bar
            J['Y_M', 'beta'] = -(H + T_bar * (a1s + i_M))
            J['Y_M', 'T_M_bar'] = b1s - (a1s + i_M) * beta
            J['Y_M', 'i_M'] = -T_bar * beta

            J['Z_M', 'T_M'] = -ones
            J['N_M', 'Q_M'] = ones
            return

        sx, cx = np.sin(a1s + i_M), np.cos(a1s + i_M)
        sb, cb = np.sin(b1s), np.cos(b1s)

        J['X_M', 'H_a1s0'] = -cx
        J['X_M', 'T_M'] = -sx
        J['X_M', 'a1s_M'] = J['X_M', 'i_M'] = H * sx - T * cx

        J['Y_M', 'H_a1s0'] = -np.sin(beta)
        J['Y_M', 'T_M'] = sb
        J['Y_M', 'b1s_M'] = T * cb
        J['Y_M', 'beta'] = -H * np.cos(beta)

        J['Z_M', 'T_M'] = -cx
        J['Z_M', 'a1s_M'] = J['Z_M', 'i_M'] = T * sx

        J['R_M', 'Q_M'] = sx
        J['R_M', 'a1s_M'] = J['R_M', 'i_M'] = Q * cx

        J['M_M', 'Q_M'] = -sb
        J['M_M', 'b1s_M'] = -Q * cb

        J['N_M', 'Q_M'] = cx * cb
        J['N_M', 'a1s_M'] = J['N_M', 'i_M'] = -Q * sx * cb
        J['N_M', 'b1s_M'] = -Q * cx * sb
