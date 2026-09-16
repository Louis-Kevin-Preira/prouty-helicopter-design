"""Tail rotor contribution to the six equations of equilibrium.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 487, Figure 8.4. Antitorque relation
also on p. 510. Anchors in Table 8.4, pp. 518-521, Table 8.5, p. 523, and
Table 8.11, pp. 536-537.
"""

import numpy as np
import openmdao.api as om


class TailRotorForcesComp(om.ExplicitComponent):
    """Forces and moments the tail rotor applies to the airframe.

    p. 487, with ``s = +1`` when the blade closest to the main rotor goes up
    and ``s = -1`` when it goes down::

        X_T = -H_T
        Y_T =  T_T
        Z_T =  s b1s_T T_T
        M_T = -s Q_T

    The rolling and yawing contributions are not here: Tables 8.4 and 8.11
    build them as ``Y_T h_T`` and ``-Y_T l_T`` from the outputs above, so the
    moment arms belong to the equilibrium components, not to this one. Same
    treatment as the main rotor, whose ``R_M``, ``M_M`` and ``N_M`` are hub
    moments while ``-X_M h_M`` and ``Z_M l_M`` are separate rows.

    No small-angle option
    ---------------------
    Unlike the main rotor, there is no ``linearized`` switch. p. 487 carries
    no trigonometry to drop, and Tables 8.4 and 8.11 reproduce these four
    expressions unchanged. A ``linearized=True`` branch would be a copy of
    the other one.

    Options
    -------
    num_nodes : int
    blade_closest : {'up', 'down'}
        Which way the blade nearest the main rotor travels, p. 487. It sets
        ``s`` above, so it flips ``Z_T`` and ``M_T`` together. Both signs are
        in service; ``'up'`` is the default.
    thrust : {'given', 'antitorque'}
        Where ``T_T`` comes from, and it differs between the two trim
        problems. In the lateral-directional set of p. 535 the tail rotor
        thrust is one of the three unknowns the N equation is solved for, so
        it arrives as an input: ``'given'``. In the longitudinal set of
        p. 516 it is not an unknown at all, and p. 487 supplies it directly
        from the antitorque balance::

            T_T = Q_M/l_T - Y_V l_V/l_T

        which is ``'antitorque'``. Using ``'antitorque'`` inside the
        lateral-directional group would impose the N equation twice.

    Feeding the vertical stabiliser back
    ------------------------------------
    In ``'antitorque'`` mode the side force ``Y_V`` that unloads the tail
    rotor is itself a function of ``T_T``, through the tail rotor sidewash of
    p. 508. p. 510 breaks that loop by taking ``Y_V`` at its value computed
    without tail rotor interference, which is what
    ``tail_rotor_feedback=False`` does at group level. Set ``Y_V = 0`` for a
    helicopter with no vertical stabiliser, or in hover, where p. 531 gives
    ``Q_M - l_T T_T = 0``.

    Example helicopter at 115 knots
    -------------------------------
    From Table 8.5, ``H_T = 40``, ``Q_T = 127``, ``b1s_T = -0.0054``,
    ``T_T = 661``:

    ===== =========== ===================================
    X_T   -40         Table 8.4, X equation
    Z_T   -3.57       Table 8.4 gives -4
    M_T   -127        Table 8.4, M equation
    ===== =========== ===================================

    Watch the ``Z_T l_T`` row of the M equation, which Table 8.4 gives as
    -148. That is the rounded ``Z_T = -4`` multiplied by ``l_T = 37``, not
    the product of the unrounded values, which is -132.1. Anyone anchoring
    the M equilibrium on -148 will be chasing a 12 % gap that exists only in
    the printing.

    The antitorque relation is 1.0 % off the book. With ``Q_M = 34,726``,
    ``l_T = 37``, ``l_V = 35`` and ``Y_V = 287``, it returns 667.5 lb where
    Table 8.5 lists 661. ``Q_M/l_T`` reproduces the 939 lb of that table
    exactly, so the difference sits in Prouty's ``Y_V``, and 6.5 lb of tail
    rotor thrust moves nothing downstream that matters at this stage.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('blade_closest', values=('up', 'down'),
                             default='up')
        self.options.declare('thrust', values=('given', 'antitorque'),
                             default='given')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        solved = self.options['thrust'] == 'antitorque'

        self.add_input('H_T', shape=(nn,), val=0.0, units='lbf',
                       desc='tail rotor H-force')
        self.add_input('Q_T', shape=(nn,), val=0.0, units='lbf*ft',
                       desc='tail rotor torque')
        self.add_input('b1s_T', shape=(nn,), val=0.0, units='rad',
                       desc='tail rotor lateral flapping')

        self.add_output('X_T', shape=(nn,), units='lbf')
        self.add_output('Y_T', shape=(nn,), units='lbf')
        self.add_output('Z_T', shape=(nn,), units='lbf')
        self.add_output('M_T', shape=(nn,), units='lbf*ft')

        self.declare_partials('X_T', 'H_T', rows=ar, cols=ar)
        self.declare_partials('M_T', 'Q_T', rows=ar, cols=ar)
        self.declare_partials('Z_T', 'b1s_T', rows=ar, cols=ar)

        if solved:
            self.add_input('Q_M', shape=(nn,), units='lbf*ft',
                           desc='main rotor torque')
            self.add_input('Y_V', shape=(nn,), val=0.0, units='lbf',
                           desc='vertical stabiliser side force')
            self.add_input('l_T', val=1.0, units='ft',
                           desc='tail rotor longitudinal offset')
            self.add_input('l_V', val=0.0, units='ft',
                           desc='vertical stabiliser longitudinal offset')
            self.add_output('T_T', shape=(nn,), units='lbf',
                            desc='tail rotor thrust')

            for out in ('T_T', 'Y_T', 'Z_T'):
                self.declare_partials(out, ['Q_M', 'Y_V'], rows=ar, cols=ar)
                self.declare_partials(out, ['l_T', 'l_V'], rows=ar, cols=zeros)
        else:
            self.add_input('T_T', shape=(nn,), units='lbf',
                           desc='tail rotor thrust')
            self.declare_partials('Y_T', 'T_T', rows=ar, cols=ar)
            self.declare_partials('Z_T', 'T_T', rows=ar, cols=ar)

    def _sign(self):
        return 1.0 if self.options['blade_closest'] == 'up' else -1.0

    def compute(self, inputs, outputs):
        s = self._sign()

        if self.options['thrust'] == 'antitorque':
            l_T, l_V = inputs['l_T'][0], inputs['l_V'][0]
            T_T = (inputs['Q_M'] - inputs['Y_V'] * l_V) / l_T
            outputs['T_T'] = T_T
        else:
            T_T = inputs['T_T']

        outputs['X_T'] = -inputs['H_T']
        outputs['Y_T'] = T_T
        outputs['Z_T'] = s * inputs['b1s_T'] * T_T
        outputs['M_T'] = -s * inputs['Q_T']

    def compute_partials(self, inputs, J):
        s = self._sign()
        b1s = inputs['b1s_T']
        ones = np.ones_like(b1s)

        J['X_T', 'H_T'] = -ones
        J['M_T', 'Q_T'] = -s * ones

        if self.options['thrust'] == 'antitorque':
            l_T, l_V = inputs['l_T'][0], inputs['l_V'][0]
            T_T = (inputs['Q_M'] - inputs['Y_V'] * l_V) / l_T

            dT = {'Q_M': ones / l_T, 'Y_V': -l_V / l_T * ones,
                  'l_V': -inputs['Y_V'] / l_T, 'l_T': -T_T / l_T}
            for name, value in dT.items():
                J['T_T', name] = value
                J['Y_T', name] = value
                J['Z_T', name] = s * b1s * value
        else:
            T_T = inputs['T_T']
            J['Y_T', 'T_T'] = ones
            J['Z_T', 'T_T'] = s * b1s

        J['Z_T', 'b1s_T'] = s * T_T
