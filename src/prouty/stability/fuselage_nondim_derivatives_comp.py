"""Nondimensional fuselage derivatives.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.14, pp. 589-590. The
downwash relation is Chapter 8, Table 8.4, already implemented here as
``RotorDownwashComp``.
"""

import numpy as np
import openmdao.api as om

UNITS = {
    'd_gammaC_d_zdot': 'rad*s/ft', 'd_beta_d_ydot': 'rad*s/ft',
    'd_epsMF_d_xdot': 'rad*s/ft', 'd_epsMF_d_zdot': 'rad*s/ft',
    'd_alphaF_d_xdot': 'rad*s/ft', 'd_alphaF_d_zdot': 'rad*s/ft',
}

#: Table 9.14 also lists six quantities read "from curves in Appendix A at
#: trim conditions". They are data, not equations, and go straight into
#: Table 9.15 rather than through here. Printed values for the example
#: helicopter, p. 590.
APPENDIX_A = {
    'df_dalphaF': -2.0,          # ft^2/rad
    'dLq_dalphaF': 74.5,         # ft^2/rad
    'dSFq_dbeta': -220.0,        # ft^2/rad
    'dMq_dalphaF': 1780.0,       # ft^3/rad
    'dNq_dbeta': -820.0,         # ft^3/rad
    'dRq_dbeta': 230.0,          # ft^3/rad
}


class FuselageNondimDerivativesComp(om.ExplicitComponent):
    """Table 9.14: how body motion moves the fuselage's angle of attack.

    The same construction as Table 9.10 with two rows missing. The fuselage
    sits in the main rotor's downwash exactly as the horizontal stabilizer
    does::

        deps_MF/dxdot = (v_F/v_1)(1/4 q A_M)[-(dZ/dxdot)_M + 2 Z_M_bar/V]
        deps_MF/dzdot = -(v_F/v_1)(1/4 q A_M)(dZ/dzdot)_M

    but it has no upwash of its own to fly in, so there is no ``eps_F`` row,
    and no downwash lag row either -- the fuselage is under the rotor, not
    33 ft behind it.

    ``v_F/v_1 = 1`` where the stabilizer has 1.5
    -------------------------------------------
    The fuselage sits in the rotor's own induced velocity; the stabilizer
    sits further aft where the wake has contracted and sped up. Chapter 8's
    Table 8.4 states the 1 explicitly, and it is what makes
    ``deps_MF/dzdot = .00051`` against the stabilizer's ``.00077`` with the
    same ``4 q A_M``.

    The six Appendix A rows
    -----------------------
    Table 9.14 also lists ``df/dalpha_F``, ``d(L/q)/dalpha_F``,
    ``d(S.F./q)/dbeta``, ``d(M/q)/dalpha_F``, ``d(N/q)/dbeta`` and
    ``d(R/q)/dbeta`` as "from curves in Appendix A at trim conditions". They
    are wind tunnel data, not equations, so they are not outputs here; they go
    straight into Table 9.15 as inputs, with the printed values as defaults.
    ``APPENDIX_A`` in this module records them.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    ================= ========== ==========
    output            model      Table 9.14
    ================= ========== ==========
    d_gammaC_d_zdot   -.005152   -.00515
    d_beta_d_ydot     .005152    .00515
    d_epsMF_d_xdot    -.000507   -.00051
    d_epsMF_d_zdot    .000515    .00051
    d_alphaF_d_xdot   .000507    .00051
    d_alphaF_d_zdot   .004637    .00467
    ================= ========== ==========
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
        self.add_input('vF_v1', val=np.full(nn, 1.0),
                       desc='Downwash at the fuselage over the rotor value.')
        self.add_input('Z_M_bar', val=np.full(nn, -20543.0), units='lbf')
        self.add_input('dZ_dxdot_M', val=np.full(nn, 45.0), units='lbf*s/ft')
        self.add_input('dZ_dzdot_M', val=np.full(nn, -261.0), units='lbf*s/ft')

        for name, units in UNITS.items():
            self.add_output(name, val=np.zeros(nn), units=units)

        self.declare_partials(['d_gammaC_d_zdot', 'd_beta_d_ydot'], 'V',
                              rows=ar, cols=ar)
        wash_x = ['V', 'q', 'vF_v1', 'Z_M_bar', 'dZ_dxdot_M']
        for name in ('d_epsMF_d_xdot', 'd_alphaF_d_xdot'):
            self.declare_partials(name, wash_x, rows=ar, cols=ar)
            self.declare_partials(name, 'A_M', rows=ar, cols=zeros)
        for name in ('d_epsMF_d_zdot', 'd_alphaF_d_zdot'):
            self.declare_partials(name, ['q', 'vF_v1', 'dZ_dzdot_M'],
                                  rows=ar, cols=ar)
            self.declare_partials(name, 'A_M', rows=ar, cols=zeros)
        self.declare_partials('d_alphaF_d_zdot', 'V', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        V, q, A_M = inputs['V'], inputs['q'], inputs['A_M'][0]
        wake = inputs['vF_v1'] / (4.0 * q * A_M)

        gamma = -1.0 / V
        eps_x = wake * (-inputs['dZ_dxdot_M'] + 2.0 * inputs['Z_M_bar'] / V)
        eps_z = -wake * inputs['dZ_dzdot_M']

        outputs['d_gammaC_d_zdot'] = gamma
        outputs['d_beta_d_ydot'] = 1.0 / V
        outputs['d_epsMF_d_xdot'] = eps_x
        outputs['d_epsMF_d_zdot'] = eps_z
        outputs['d_alphaF_d_xdot'] = -eps_x
        outputs['d_alphaF_d_zdot'] = -(eps_z + gamma)

    def compute_partials(self, inputs, J):
        V, q, A_M = inputs['V'], inputs['q'], inputs['A_M'][0]
        vF, Z_M = inputs['vF_v1'], inputs['Z_M_bar']
        Zx, Zz = inputs['dZ_dxdot_M'], inputs['dZ_dzdot_M']
        scale = 4.0 * q * A_M
        wake = vF / scale
        bracket = -Zx + 2.0 * Z_M / V

        J['d_gammaC_d_zdot', 'V'] = 1.0 / V ** 2
        J['d_beta_d_ydot', 'V'] = -1.0 / V ** 2

        J['d_epsMF_d_xdot', 'vF_v1'] = bracket / scale
        J['d_epsMF_d_xdot', 'q'] = -wake * bracket / q
        J['d_epsMF_d_xdot', 'A_M'] = -wake * bracket / A_M
        J['d_epsMF_d_xdot', 'dZ_dxdot_M'] = -wake
        J['d_epsMF_d_xdot', 'Z_M_bar'] = 2.0 * wake / V
        J['d_epsMF_d_xdot', 'V'] = -2.0 * wake * Z_M / V ** 2

        J['d_epsMF_d_zdot', 'vF_v1'] = -Zz / scale
        J['d_epsMF_d_zdot', 'q'] = wake * Zz / q
        J['d_epsMF_d_zdot', 'A_M'] = wake * Zz / A_M
        J['d_epsMF_d_zdot', 'dZ_dzdot_M'] = -wake

        for name in ('V', 'q', 'vF_v1', 'Z_M_bar', 'dZ_dxdot_M', 'A_M'):
            J['d_alphaF_d_xdot', name] = -J['d_epsMF_d_xdot', name]
        for name in ('q', 'vF_v1', 'dZ_dzdot_M', 'A_M'):
            J['d_alphaF_d_zdot', name] = -J['d_epsMF_d_zdot', name]
        J['d_alphaF_d_zdot', 'V'] = -J['d_gammaC_d_zdot', 'V']
