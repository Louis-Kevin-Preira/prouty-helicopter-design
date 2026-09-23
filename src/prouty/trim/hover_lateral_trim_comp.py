"""Lateral-directional trim in hover.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", pp. 531-532, Figure 8.28 p. 531.
Special cases in Table 8.9 p. 532, example in Table 8.10 p. 533.
"""

import numpy as np
import openmdao.api as om


class HoverLateralTrimComp(om.ExplicitComponent):
    """Roll angle and lateral flapping in hover, pp. 531-532.

    p. 531 keeps only the two rotors, the airframe aerodynamics being
    negligible in hover::

        Y   T_M b1s_M + T_T = -G.W. Phi
        R   (dR_M/db1s + T_M h_M) b1s_M - T_M y_M + T_T h_T = 0
        N   Q_M - l_T T_T = 0

    The N equation is the antitorque relation and is solved elsewhere, by
    ``TailRotorForcesComp`` in its ``'antitorque'`` mode with no fin, so
    ``T_T`` arrives as an input. What is left is two equations in two
    unknowns, and R contains only ``b1s_M``, so p. 532 solves them in
    sequence::

        b1s_M = (T_M y_M - T_T h_T) / (dR_M/db1s + T_M h_M)
        Phi   = -(T_T + T_M b1s_M) / G.W.

    Like the longitudinal hover problem of p. 517, this is triangular rather
    than coupled, so no solver is needed. ``T_M`` also comes from there:
    p. 532 repeats the same ``G.W./[1 - (D_v/G.W.)]`` and there is no reason
    to define it twice.

    What the helicopter is doing
    ----------------------------
    The tail rotor pushes sideways, so something must push back. Only two
    things can: tilting the main rotor thrust, which needs lateral flapping,
    and banking the whole aircraft, which lets gravity supply the side force.
    How the requirement divides between them is the whole content of the
    equations, and Table 8.9 p. 532 gives the three limits:

    ============== ============= ============ ============ ==============
    ``y_M``         rotor         ``h_T``      ``Phi``      ``b1s_M``
    ============== ============= ============ ============ ==============
    0               teetering     ``h_M``      0            -T_T/G.W.
    0               any           0            -T_T/G.W.    0
    any             very rigid    any          -T_T/G.W.    0
    ============== ============= ============ ============ ==============

    Read together they say the roll angle is avoidable only when the tail
    rotor thrust line passes through the main rotor hub, and the flapping is
    avoidable only when it passes through the c.g. or when the hub is stiff
    enough to refuse to flap. No real helicopter is any of these, so it
    hovers slightly banked with the disc slightly tilted, which is the
    left-skid-low attitude every pilot knows.

    Anchor
    ------
    Table 8.10 p. 533, example helicopter hovering out of ground effect:
    ``G.W. = 20,000``, ``T_M = 20,840``, ``T_T = 1,540``, ``y_M = 0``,
    ``h_M = 7.5``, ``h_T = 6``, ``dR_M/db1s = 200,940``.

    ======= ============ ==================
    b1s_M    -0.02586     -1.48 deg
    Phi      -0.05005     -2.87 deg
    ======= ============ ==================

    p. 532 prints ``Phi = -0.050 rad = -2.9 deg``, which is this value, and
    ``b1s_M = -0.027 rad = -1.5 deg``, which is not: Table 8.10's own numbers
    give -0.0259. The printed roll angle settles it, since rebuilding ``Phi``
    from ``b1s_M = -0.027`` gives -0.0489, which would print as -0.049 and
    -2.8 rather than the -0.050 and -2.9 shown. See C8-9 in
    ``docs/validation_trim.md``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('GW', shape=(nn,), val=1.0, units='lbf',
                       desc='gross weight')
        self.add_input('T_M', shape=(nn,), val=1.0, units='lbf',
                       desc='main rotor thrust, from the hover Z equation')
        self.add_input('T_T', shape=(nn,), val=0.0, units='lbf',
                       desc='tail rotor thrust, from the hover N equation')
        self.add_input('dRM_db1s', shape=(nn,), val=0.0, units='lbf*ft/rad',
                       desc='hub rolling moment stiffness')
        self.add_input('y_M', val=0.0, units='ft',
                       desc='lateral c.g. offset of the rotor')
        self.add_input('h_M', val=0.0, units='ft', desc='rotor height')
        self.add_input('h_T', val=0.0, units='ft', desc='tail rotor height')

        self.add_output('b1s_M', shape=(nn,), units='rad',
                        desc='lateral flapping')
        self.add_output('Phi', shape=(nn,), units='rad', desc='roll angle')

        self._vector = ('GW', 'T_M', 'T_T', 'dRM_db1s')
        self._scalar = ('y_M', 'h_M', 'h_T')

        # the R equation has no weight in it, so b1s_M does not depend on GW
        self.declare_partials('b1s_M', [n for n in self._vector
                                        if n != 'GW'], rows=ar, cols=ar)
        self.declare_partials('b1s_M', self._scalar, rows=ar, cols=zeros)
        self.declare_partials('Phi', self._vector, rows=ar, cols=ar)
        self.declare_partials('Phi', self._scalar, rows=ar, cols=zeros)

    def _solve(self, inputs):
        T_M, T_T = inputs['T_M'], inputs['T_T']
        y_M, h_M, h_T = (inputs['y_M'][0], inputs['h_M'][0],
                         inputs['h_T'][0])

        N = T_M * y_M - T_T * h_T
        D = inputs['dRM_db1s'] + T_M * h_M
        b1s = N / D
        return N, D, b1s, -(T_T + T_M * b1s) / inputs['GW']

    def compute(self, inputs, outputs):
        _, _, b1s, phi = self._solve(inputs)
        outputs['b1s_M'], outputs['Phi'] = b1s, phi

    def compute_partials(self, inputs, J):
        GW, T_M, T_T = inputs['GW'], inputs['T_M'], inputs['T_T']
        y_M, h_M, h_T = (inputs['y_M'][0], inputs['h_M'][0],
                         inputs['h_T'][0])
        N, D, b1s, phi = self._solve(inputs)

        zero = np.zeros_like(GW)
        one = np.ones_like(GW)

        # b1s = N/D, with N and D each depending on a few inputs
        dN = {'T_M': y_M * one, 'T_T': -h_T * one, 'y_M': T_M, 'h_T': -T_T}
        dD = {'T_M': h_M * one, 'dRM_db1s': one, 'h_M': T_M}

        for name in self._vector + self._scalar:
            db1s = dN.get(name, zero) / D - N * dD.get(name, zero) / D ** 2
            if name != 'GW':
                J['b1s_M', name] = db1s
            J['Phi', name] = -T_M * db1s / GW
            if name == 'T_T':
                J['Phi', name] -= 1.0 / GW
            if name == 'T_M':
                J['Phi', name] -= b1s / GW
            if name == 'GW':
                J['Phi', name] -= phi / GW
