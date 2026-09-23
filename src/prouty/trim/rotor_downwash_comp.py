"""Main rotor downwash angle at an airframe station.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 493, Figure 8.11 pp. 494-495.
Anchors in Table 8.2 p. 501, Table 8.4 pp. 518-521 and Table 8.5 p. 523.
"""

import numpy as np
import openmdao.api as om


class RotorDownwashComp(om.ExplicitComponent):
    """Downwash the main rotor induces at an airframe station.

    p. 493 writes the angle two ways, for the horizontal stabiliser::

        eps_M = (v/v1)(v1/V)  =  (v/v1) D.L./(4q)

    and they are the same statement, since the high-speed momentum induced
    velocity of p. 167, ``v1 = T_M/(2 rho A_M V)``, gives
    ``v1/V = D.L./(4q)`` exactly. This component uses the second form, with
    ``D.L. = T_M/A_M``, which is what Table 8.4 carries as
    ``T_M/(4 q A_M)`` and which keeps the unknown thrust visible in the
    Jacobian.

    Everything difficult sits in ``v_H/v1``, the ratio of local to reference
    induced velocity at the stabiliser.

    Options
    -------
    num_nodes : int
    downwash_model : {'input', 'vortex_axis'}
        ``'input'`` takes ``v_ratio`` from Figure 8.11 and is the default.
        ``'vortex_axis'`` computes it from momentum theory; see below.

    Why Figure 8.11 is not digitised
    --------------------------------
    The figure is a survey, not a curve: nineteen traces of ``v_H/v1``
    against ``Y/R``, at two longitudinal stations and ten vertical ones,
    from Heyson and Katzoff, NACA TR 1319. Reading it means picking the
    trace nearest the stabiliser's ``X'/R`` and ``Z'/R`` and averaging over
    its span, which is a judgement, not an interpolation. Two features of it
    are also outside any uniform-disc theory, as p. 495 points out: values
    above 2, the maximum a uniformly loaded disc can produce, and an
    asymmetry between the advancing and retreating sides that couples pitch
    with sideslip. Both come from the non-uniform loading TR 1319 was
    written about. A digitisation would carry neither into a form a design
    sweep could use, so the reading stays where the engineer made it.

    ``downwash_model='vortex_axis'``
    -------------------------------
    A default for cases where the geometry moves and no chart reading
    exists. It is the on-axis induced velocity of a uniform vortex cylinder,

        v(s)/v1 = 1 + s/sqrt(s^2 + R^2)

    which is 1 at the disc and tends to the fully developed 2 far downstream,
    evaluated at the distance ``s`` from the disc measured *along the skewed
    wake axis*. With ``k = v1/V = T_M/(4 q A_M)`` the wake axis runs aft and
    down as ``(1, k)/sqrt(1 + k^2)``, so for a stabiliser at Figure 8.11's
    coordinates::

        s/R = (-X'/R + k Z'/R) / sqrt(1 + k^2)

    Two things it does not do. It ignores lateral position entirely, so the
    advancing-retreating asymmetry is absent by construction. And it keeps
    the rotor radius as the cylinder radius, whereas at high tip speed ratio
    the wake axis is nearly horizontal and the cross-section normal to it is
    a flattened ellipse of semi-axes ``R`` and ``kR``. That second point
    makes the model generous:

    ============================ ==========
    Figure 8.11, as Prouty reads   1.5
    ``'vortex_axis'``              1.737
    ============================ ==========

    for the example helicopter at 115 knots, ``X'/R = -1.08``,
    ``Z'/R = +0.3``, ``k = 0.0405`` — 16 % high, giving 4.0 deg of downwash
    against the 3.5 deg of p. 493. Use it to keep a gradient-based sweep
    moving, not to replace a reading.

    One component, three stations
    -----------------------------
    The same relation serves the horizontal stabiliser, the fuselage and the
    vertical stabiliser; only ``v/v1`` changes. Table 8.5 pp. 524-525 lists
    1.5 for both stabilisers, and Table 8.4 uses 1.0 at the fuselage, which
    is what makes ``eps_MF = T_M/(4 q A_M)`` and reduces the fuselage
    downwash to the ``v1/V`` of Chapter 3 p. 192 exactly.

    The input and output are therefore named neutrally, ``v_ratio`` and
    ``eps_M``, and the group renames them per station on promotion::

        promotes_outputs=[('eps_M', 'eps_MH')]

    Sign convention
    ---------------
    ``X_over_R`` and ``Z_over_R`` are Figure 8.11's survey coordinates
    referred to the hub: ``X'`` positive forward, so a stabiliser behind the
    rotor has ``X'/R < 0``, and ``Z'`` positive *downward*. The example
    helicopter sits at -1.08 and +0.3, and the +0.3 is recovered exactly from
    Table 8.5's offsets, ``(h_M - h_H)/R = (7.5 + 1.5)/30``.

    Anchor
    ------
    At 115 knots, with ``T_M = 20,606``, ``q = 45``, ``A_M = 2,827`` and
    ``v_ratio = 1.5`` from Table 8.2, ``eps_M = 0.0607 rad = 3.48 deg``
    against the 0.06 rad and 3.5 deg of p. 493.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('downwash_model',
                             values=('input', 'vortex_axis'), default='input')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        derived = self.options['downwash_model'] == 'vortex_axis'

        self.add_input('T_M', shape=(nn,), units='lbf', desc='rotor thrust')
        self.add_input('q', shape=(nn,), units='lbf/ft**2',
                       desc='free stream dynamic pressure')
        self.add_input('A_M', val=1.0, units='ft**2', desc='rotor disc area')

        self.add_output('eps_M', shape=(nn,), units='rad',
                        desc='rotor downwash angle at the horizontal stab.')

        self.declare_partials('eps_M', ['T_M', 'q'], rows=ar, cols=ar)
        self.declare_partials('eps_M', 'A_M', rows=ar, cols=zeros)

        if derived:
            self.add_input('X_over_R', val=0.0,
                           desc="X'/R of the stabiliser, positive forward")
            self.add_input('Z_over_R', val=0.0,
                           desc="Z'/R of the stabiliser, positive down")
            self.add_output('v_ratio', shape=(nn,),
                            desc='local over reference induced velocity')

            self.declare_partials('v_ratio', ['T_M', 'q'], rows=ar, cols=ar)
            for name in ('A_M', 'X_over_R', 'Z_over_R'):
                self.declare_partials('v_ratio', name, rows=ar, cols=zeros)
                self.declare_partials('eps_M', name, rows=ar, cols=zeros)
        else:
            self.add_input('v_ratio', shape=(nn,), val=1.0,
                           desc='v_H/v1 read from Figure 8.11')
            self.declare_partials('eps_M', 'v_ratio', rows=ar, cols=ar)

    def _wake(self, inputs, k):
        """s/R along the skewed wake axis, and v_H/v1 on it."""
        Xr, Zr = inputs['X_over_R'][0], inputs['Z_over_R'][0]
        u = np.sqrt(1.0 + k ** 2)
        s = (-Xr + k * Zr) / u
        return s, u, 1.0 + s / np.sqrt(s ** 2 + 1.0)

    def compute(self, inputs, outputs):
        k = inputs['T_M'] / (4.0 * inputs['q'] * inputs['A_M'][0])

        if self.options['downwash_model'] == 'vortex_axis':
            _, _, ratio = self._wake(inputs, k)
            outputs['v_ratio'] = ratio
        else:
            ratio = inputs['v_ratio']

        outputs['eps_M'] = ratio * k

    def compute_partials(self, inputs, J):
        T, q, A = inputs['T_M'], inputs['q'], inputs['A_M'][0]
        k = T / (4.0 * q * A)
        dk = {'T_M': 1.0 / (4.0 * q * A), 'q': -k / q, 'A_M': -k / A}

        if self.options['downwash_model'] == 'input':
            ratio = inputs['v_ratio']
            J['eps_M', 'v_ratio'] = k
            for name, value in dk.items():
                J['eps_M', name] = ratio * value
            return

        Zr = inputs['Z_over_R'][0]
        s, u, ratio = self._wake(inputs, k)

        dratio_ds = (s ** 2 + 1.0) ** -1.5
        ds_dk = Zr / u - s * k / u ** 2

        # position enters only through s
        for name, ds in (('X_over_R', -1.0 / u), ('Z_over_R', k / u)):
            J['v_ratio', name] = dratio_ds * ds
            J['eps_M', name] = k * dratio_ds * ds

        deps_dk = ratio + k * dratio_ds * ds_dk
        for name, value in dk.items():
            J['v_ratio', name] = dratio_ds * ds_dk * value
            J['eps_M', name] = deps_dk * value
