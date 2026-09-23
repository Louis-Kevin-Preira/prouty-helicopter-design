"""Longitudinal trim in hover.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", pp. 516-517, Figure 8.26. Worked example
on p. 517.
"""

import numpy as np
import openmdao.api as om


class HoverLongitudinalTrimComp(om.ExplicitComponent):
    """Longitudinal trim in hover, solved in closed form, pp. 516-517.

    p. 516 drops the airframe aerodynamics except for vertical drag on the
    fuselage and the horizontal stabiliser, which leaves::

        X   -T_M (i_M + a1s_M) = G.W. Theta
        Z   -T_M + T_M (D_v/G.W.)_H + T_M (D_v/G.W.)_F = -G.W.
        M   (dM_M/da1s) a1s_M + T_M (i_M + a1s_M) h_M - T_M l_M + M_T
            + T_M (D_v/G.W.)_H l_H + T_M (D_v/G.W.)_F l_F = 0

    Why this is not a BalanceComp
    -----------------------------
    p. 517 notes that the first and third equations contain the nonlinear
    product ``T_M a1s_M``, but that the second gives a unique ``T_M``, so the
    set can be solved easily. It is stronger than that: knowing ``T_M`` makes
    the third equation *linear* in ``a1s_M``, and the first then returns
    ``Theta`` outright. The system is triangular, so it is solved as written
    and no solver is needed::

        T_M    = G.W. / [1 - (D_v/G.W.)_H - (D_v/G.W.)_F]
        a1s_M  = -[T_M c + M_T] / [(dM_M/da1s) + T_M h_M]
        Theta  = -T_M (i_M + a1s_M) / G.W.

    with ``c = i_M h_M - l_M + (D_v/G.W.)_H l_H + (D_v/G.W.)_F l_F``.

    The vertical drag ratios
    ------------------------
    ``(D_v/G.W.)`` is the download the rotor wake puts on a surface, as a
    fraction of gross weight, from Chapter 4. It does two things here, and
    the second is easy to miss: it raises the thrust the rotor must produce,
    and, acting at ``l_H`` and ``l_F`` from the c.g., it also pitches the
    aircraft. For the example helicopter the fuselage download alone is
    4.2 % of gross weight, worth 877 lb of extra thrust.

    The sign in front of the tail rotor torque
    ------------------------------------------
    p. 517 writes it as ``± Q_T`` and explains that tail rotors turn either
    way. That is the same choice ``TailRotorForcesComp`` makes with
    ``blade_closest``, which is why the term arrives here as ``M_T`` rather
    than as ``Q_T`` and a second sign convention. For the example helicopter,
    ``M_T = -Q_T = -990 ft-lb``.

    Anchor
    ------
    p. 517, with ``G.W. = 20,000``, ``i_M = 0``, ``(D_v/G.W.)_H = 0``,
    ``(D_v/G.W.)_F = 0.042``, ``M_T = -990``, ``h_M = 7.5``, ``l_M = -0.5``,
    ``l_F = -0.5`` and ``dM_M/da1s = 200,940`` from Chapter 7:

    ======= ============ =====================
    T_M      20,876.8      book 20,877
    a1s_M    -0.02520      book -0.025, -1.5 deg
    Theta     0.02631      book 0.026, 1.4 deg
    ======= ============ =====================

    Both angles are printed to two decimals in radians and to one in
    degrees, and the two roundings disagree: -0.025 rad is -1.43 deg, not the
    -1.5 deg beside it. The unrounded solution is -1.444 deg, so the radian
    figure is the faithful one.

    Notes
    -----
    The rotor stiffness enters as ``(dM_M/da1s) + T_M h_M``, a sum of the hub
    moment and the thrust-times-height term. For this helicopter the second
    is 156,576 against 200,940, so nearly half the pitch stiffness in hover
    comes from the rotor sitting 7.5 ft above the c.g. rather than from the
    hub itself. A teetering rotor, with no hub moment at all, still trims.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('GW', shape=(nn,), val=1.0, units='lbf',
                       desc='gross weight')
        self.add_input('Dv_GW_H', shape=(nn,), val=0.0,
                       desc='horizontal stabiliser vertical drag / G.W.')
        self.add_input('Dv_GW_F', shape=(nn,), val=0.0,
                       desc='fuselage vertical drag / G.W.')
        self.add_input('dMM_da1s', shape=(nn,), val=0.0, units='lbf*ft/rad',
                       desc='hub pitching moment stiffness')
        self.add_input('M_T', shape=(nn,), val=0.0, units='lbf*ft',
                       desc='tail rotor pitching moment, -/+ Q_T')
        self.add_input('i_M', val=0.0, units='rad', desc='shaft incidence')
        self.add_input('h_M', val=0.0, units='ft', desc='rotor height')
        self.add_input('l_M', val=0.0, units='ft', desc='rotor offset')
        self.add_input('l_H', val=0.0, units='ft',
                       desc='horizontal stabiliser offset')
        self.add_input('l_F', val=0.0, units='ft', desc='fuselage offset')

        self.add_output('T_M', shape=(nn,), units='lbf', desc='rotor thrust')
        self.add_output('a1s_M', shape=(nn,), units='rad',
                        desc='longitudinal flapping')
        self.add_output('Theta', shape=(nn,), units='rad',
                        desc='fuselage pitch attitude')

        self._vector = ('GW', 'Dv_GW_H', 'Dv_GW_F', 'dMM_da1s', 'M_T')
        self._scalar = ('i_M', 'h_M', 'l_M', 'l_H', 'l_F')

        self.declare_partials('T_M', ['GW', 'Dv_GW_H', 'Dv_GW_F'],
                              rows=ar, cols=ar)
        for out in ('a1s_M', 'Theta'):
            self.declare_partials(out, self._vector, rows=ar, cols=ar)
            self.declare_partials(out, self._scalar, rows=ar, cols=zeros)

    def _solve(self, inputs):
        dvH, dvF = inputs['Dv_GW_H'], inputs['Dv_GW_F']
        i_M, h_M, l_M = inputs['i_M'][0], inputs['h_M'][0], inputs['l_M'][0]
        l_H, l_F = inputs['l_H'][0], inputs['l_F'][0]

        T = inputs['GW'] / (1.0 - dvH - dvF)
        c = i_M * h_M - l_M + dvH * l_H + dvF * l_F
        N = T * c + inputs['M_T']
        D = inputs['dMM_da1s'] + T * h_M
        a1s = -N / D
        return T, c, N, D, a1s, -T * (i_M + a1s) / inputs['GW']

    def compute(self, inputs, outputs):
        T, _, _, _, a1s, theta = self._solve(inputs)
        outputs['T_M'], outputs['a1s_M'], outputs['Theta'] = T, a1s, theta

    def compute_partials(self, inputs, J):
        GW, dvH, dvF = inputs['GW'], inputs['Dv_GW_H'], inputs['Dv_GW_F']
        i_M, h_M = inputs['i_M'][0], inputs['h_M'][0]
        l_H, l_F = inputs['l_H'][0], inputs['l_F'][0]
        T, c, N, D, a1s, _ = self._solve(inputs)

        d = 1.0 - dvH - dvF
        zero = np.zeros_like(GW)
        one = np.ones_like(GW)

        # each stage differentiated once, then chained
        dT = {'GW': 1.0 / d, 'Dv_GW_H': T / d, 'Dv_GW_F': T / d}
        dc = {'Dv_GW_H': l_H * one, 'Dv_GW_F': l_F * one, 'i_M': h_M * one,
              'h_M': i_M * one, 'l_M': -one, 'l_H': dvH, 'l_F': dvF}

        for name in self._vector + self._scalar:
            t = dT.get(name, zero)
            dN = c * t + T * dc.get(name, zero) + (one if name == 'M_T' else 0)
            dD = h_M * t + (one if name == 'dMM_da1s' else 0) \
                + (T if name == 'h_M' else 0)
            da1s = -dN / D + N * dD / D ** 2

            J['a1s_M', name] = da1s
            J['Theta', name] = -(t * (i_M + a1s) + T * da1s) / GW
            if name == 'i_M':
                J['Theta', name] -= T / GW
            if name == 'GW':
                J['Theta', name] += T * (i_M + a1s) / GW ** 2
            if name in dT:
                J['T_M', name] = t
