"""Nondimensional horizontal stabilizer derivatives.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.10, p. 584, with the
angle-of-attack equation it rests on given on p. 586. The steady forms are
Chapter 8, pp. 488-489.
"""

import numpy as np
import openmdao.api as om

UNITS = {
    'd_gammaC_d_zdot': 'rad*s/ft',
    'd_epsMH_d_xdot': 'rad*s/ft', 'd_epsMH_d_zdot': 'rad*s/ft',
    'd_epsFH_d_zdot': 'rad*s/ft',
    'd_alphaH_d_xdot': 'rad*s/ft', 'd_alphaH_d_zdot': 'rad*s/ft',
    'd_alphaH_d_zddot': 'rad*s**2/ft',
}


class HorizStabNondimDerivativesComp(om.ExplicitComponent):
    """Table 9.10: how body motion moves the stabilizer's angle of attack.

    p. 584 says the Chapter 8 equations "can be used almost as is", with one
    exception. p. 586 gives it::

        alpha_H = Theta + i_H - (eps_MH + eps_FH) - gamma_c
                  - (deps_MH/dzdot) zddot (l_H/V)

    The last term is new to Chapter 9. It is the time the main rotor downwash
    takes to reach the stabilizer, ``Delta t = l_H/V``, which matters only in
    unsteady motion and which is the whole of ``dalpha_H/dzddot``. For the
    example helicopter that is 33 ft at 194 ft/sec, a 0.17 second lag.

    The seven rows
    --------------
    Three are the separate contributions to the stabilizer's angle::

        dgamma_c/dzdot = -1/V                          the flight path angle
        deps_MH/dxdot, deps_MH/dzdot                   main rotor downwash
        deps_FH/dzdot                                  fuselage upwash

    and three combine them into the angle of attack itself, with the lag term
    making the seventh.

    The main rotor downwash rows work through momentum theory: the induced
    velocity goes as the square root of disc loading, so a perturbation of the
    rotor's Z force changes the downwash at the stabilizer by
    ``(v_H/v_1)(1/4 q A_M)`` times it. ``v_H/v_1`` is how much of the rotor's
    own induced velocity reaches the stabilizer; 1.5 for the example
    helicopter, the wake being partly contracted by the time it gets there.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    ================== ========== =========
    output             model      Table 9.10
    ================== ========== =========
    d_gammaC_d_zdot    -.005152   -.00515
    d_epsMH_d_xdot     -.000761   -.00076
    d_epsMH_d_zdot     .000773    .00077
    d_epsFH_d_zdot     .001080    .00108
    d_alphaH_d_xdot    .000761    .00076
    d_alphaH_d_zdot    .003299    .00331
    d_alphaH_d_zddot   -.000132   -.00014
    ================== ========== =========

    Four inputs are not printed in Chapter 9 and were recovered from the table
    itself: ``v_H/v_1 = 1.5`` from ``deps_MH/dzdot``, ``Z_M_bar = -20,543 lb``
    from ``deps_MH/dxdot`` -- essentially the gross weight plus vertical drag
    -- ``deps_F/dalpha_F = 0.23`` from ``deps_FH/dzdot``, and ``l_H = 33.1 ft``
    from Table 9.11's ``dM/dq = -7,161``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('V', val=np.full(nn, 194.1), units='ft/s')
        self.add_input('q', val=np.full(nn, 44.793), units='lbf/ft**2')
        self.add_input('A_M', val=2827.4, units='ft**2',
                       desc='Main rotor disc area.')
        self.add_input('vH_v1', val=np.full(nn, 1.5),
                       desc='Downwash at the stabilizer over the rotor value.')
        self.add_input('Z_M_bar', val=np.full(nn, -20543.0), units='lbf',
                       desc='Trim main rotor Z force.')
        self.add_input('dZ_dxdot_M', val=np.full(nn, 45.0), units='lbf*s/ft')
        self.add_input('dZ_dzdot_M', val=np.full(nn, -261.0), units='lbf*s/ft')
        self.add_input('deps_F_dalpha_F', val=np.full(nn, 0.23),
                       desc='Fuselage downwash slope, Chapter 8 p. 489.')
        self.add_input('l_H', val=33.1, units='ft')

        for name, units in UNITS.items():
            self.add_output(name, val=np.zeros(nn), units=units)

        self.declare_partials('d_gammaC_d_zdot', 'V', rows=ar, cols=ar)
        self.declare_partials('d_epsMH_d_xdot',
                              ['V', 'q', 'vH_v1', 'Z_M_bar', 'dZ_dxdot_M'],
                              rows=ar, cols=ar)
        self.declare_partials('d_epsMH_d_zdot', ['q', 'vH_v1', 'dZ_dzdot_M'],
                              rows=ar, cols=ar)
        self.declare_partials('d_epsFH_d_zdot',
                              ['q', 'dZ_dzdot_M', 'deps_F_dalpha_F', 'V'],
                              rows=ar, cols=ar)
        self.declare_partials('d_alphaH_d_xdot',
                              ['V', 'q', 'vH_v1', 'Z_M_bar', 'dZ_dxdot_M'],
                              rows=ar, cols=ar)
        self.declare_partials('d_alphaH_d_zdot',
                              ['q', 'vH_v1', 'dZ_dzdot_M', 'deps_F_dalpha_F',
                               'V'], rows=ar, cols=ar)
        self.declare_partials('d_alphaH_d_zddot',
                              ['q', 'vH_v1', 'dZ_dzdot_M', 'V'],
                              rows=ar, cols=ar)
        for name in ('d_epsMH_d_xdot', 'd_epsMH_d_zdot', 'd_epsFH_d_zdot',
                     'd_alphaH_d_xdot', 'd_alphaH_d_zdot', 'd_alphaH_d_zddot'):
            self.declare_partials(name, 'A_M', rows=ar, cols=zeros)
        self.declare_partials('d_alphaH_d_zddot', 'l_H', rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        V, q, A_M = inputs['V'], inputs['q'], inputs['A_M'][0]
        wake = inputs['vH_v1'] / (4.0 * q * A_M)

        gamma = -1.0 / V
        eps_MH_x = wake * (-inputs['dZ_dxdot_M'] + 2.0 * inputs['Z_M_bar'] / V)
        eps_MH_z = -wake * inputs['dZ_dzdot_M']
        eps_FH_z = inputs['deps_F_dalpha_F'] * (
            inputs['dZ_dzdot_M'] / (4.0 * q * A_M) - gamma)

        outputs['d_gammaC_d_zdot'] = gamma
        outputs['d_epsMH_d_xdot'] = eps_MH_x
        outputs['d_epsMH_d_zdot'] = eps_MH_z
        outputs['d_epsFH_d_zdot'] = eps_FH_z
        outputs['d_alphaH_d_xdot'] = -eps_MH_x
        outputs['d_alphaH_d_zdot'] = -(eps_MH_z + eps_FH_z + gamma)
        outputs['d_alphaH_d_zddot'] = -eps_MH_z * inputs['l_H'][0] / V

    def compute_partials(self, inputs, J):
        V, q, A_M = inputs['V'], inputs['q'], inputs['A_M'][0]
        vH, Z_M = inputs['vH_v1'], inputs['Z_M_bar']
        Zx, Zz = inputs['dZ_dxdot_M'], inputs['dZ_dzdot_M']
        epsF, l_H = inputs['deps_F_dalpha_F'], inputs['l_H'][0]
        scale = 4.0 * q * A_M
        wake = vH / scale
        bracket = -Zx + 2.0 * Z_M / V

        J['d_gammaC_d_zdot', 'V'] = 1.0 / V ** 2

        J['d_epsMH_d_xdot', 'vH_v1'] = bracket / scale
        J['d_epsMH_d_xdot', 'q'] = -wake * bracket / q
        J['d_epsMH_d_xdot', 'A_M'] = -wake * bracket / A_M
        J['d_epsMH_d_xdot', 'dZ_dxdot_M'] = -wake
        J['d_epsMH_d_xdot', 'Z_M_bar'] = 2.0 * wake / V
        J['d_epsMH_d_xdot', 'V'] = -2.0 * wake * Z_M / V ** 2

        J['d_epsMH_d_zdot', 'vH_v1'] = -Zz / scale
        J['d_epsMH_d_zdot', 'q'] = wake * Zz / q
        J['d_epsMH_d_zdot', 'A_M'] = wake * Zz / A_M
        J['d_epsMH_d_zdot', 'dZ_dzdot_M'] = -wake

        J['d_epsFH_d_zdot', 'deps_F_dalpha_F'] = Zz / scale + 1.0 / V
        J['d_epsFH_d_zdot', 'q'] = -epsF * Zz / (scale * q)
        J['d_epsFH_d_zdot', 'A_M'] = -epsF * Zz / (scale * A_M)
        J['d_epsFH_d_zdot', 'dZ_dzdot_M'] = epsF / scale
        J['d_epsFH_d_zdot', 'V'] = -epsF / V ** 2

        for name in ('V', 'q', 'vH_v1', 'Z_M_bar', 'dZ_dxdot_M', 'A_M'):
            J['d_alphaH_d_xdot', name] = -J['d_epsMH_d_xdot', name]

        for name in ('q', 'dZ_dzdot_M', 'A_M'):
            J['d_alphaH_d_zdot', name] = -(J['d_epsMH_d_zdot', name]
                                           + J['d_epsFH_d_zdot', name])
        J['d_alphaH_d_zdot', 'vH_v1'] = -J['d_epsMH_d_zdot', 'vH_v1']
        J['d_alphaH_d_zdot', 'deps_F_dalpha_F'] = \
            -J['d_epsFH_d_zdot', 'deps_F_dalpha_F']
        J['d_alphaH_d_zdot', 'V'] = -(J['d_epsFH_d_zdot', 'V']
                                      + J['d_gammaC_d_zdot', 'V'])

        eps_MH_z = -wake * Zz
        for name in ('q', 'vH_v1', 'dZ_dzdot_M', 'A_M'):
            J['d_alphaH_d_zddot', name] = (-J['d_epsMH_d_zdot', name]
                                           * l_H / V)
        J['d_alphaH_d_zddot', 'l_H'] = -eps_MH_z / V
        J['d_alphaH_d_zddot', 'V'] = eps_MH_z * l_H / V ** 2
