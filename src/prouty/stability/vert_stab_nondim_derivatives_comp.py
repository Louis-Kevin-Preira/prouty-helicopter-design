"""Nondimensional vertical stabilizer derivatives.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.12, p. 587. The sidewash
relations are Chapter 8, pp. 490-491, where they are already implemented in
this repository as ``TailRotorSidewashComp`` and ``FuselageSidewashComp``.
"""

import numpy as np
import openmdao.api as om

UNITS = {
    'd_beta_d_ydot': 'rad*s/ft',
    'd_etaTV_d_xdot': 'rad*s/ft', 'd_etaTV_d_ydot': 'rad*s/ft',
    'd_etaFV_d_ydot': 'rad*s/ft',
    'd_alphaV_d_xdot': 'rad*s/ft', 'd_alphaV_d_ydot': 'rad*s/ft',
}


class VertStabNondimDerivativesComp(om.ExplicitComponent):
    """Table 9.12: how body motion moves the vertical stabilizer's angle.

    The mirror of Table 9.10, with the tail rotor standing where the main
    rotor stood. Where the horizontal stabilizer flies in the main rotor's
    downwash, the vertical stabilizer flies in the tail rotor's sidewash, and
    the momentum relation is the same one with ``A_T`` for ``A_M`` and ``T_T``
    for ``Z_M``::

        deta_TV/dxdot = -[(dY/dxdot)_T - 2 T_T/V] / [4 (q_V/q) q A_T]
        deta_TV/dydot =  (dY/dydot)_T             / [4 (q_V/q) q A_T]

    The fuselage contributes through its own sidewash slope,
    ``deta_FV/dydot = (deta_F/dbeta)(dbeta/dydot)``, with ``deta_F/dbeta =
    0.06`` for the example helicopter -- the same 0.06 the Chapter 8 work in
    this repository already uses.

    The two sidewash rows do not share a sign convention
    ----------------------------------------------------
    This is the trap in the table, and it is not a misprint.
    ``deta_TV/dxdot`` carries a leading minus inside its own equation;
    ``deta_TV/dydot`` does not. So the two angle-of-attack rows treat them
    differently::

        dalpha_V/dxdot = +deta_T/dxdot                       no sign change
        dalpha_V/dydot = -(dbeta/dydot + deta_T/dydot + deta_F/dydot)

    The first looks wrong beside Table 9.10, whose equivalent row is
    ``dalpha_H/dxdot = -deps_MH/dxdot``. It is not. Feeding the positive value
    into Table 9.13's ``dY/dxdot = (2/V) Y_V_bar + (q_V/q) q A_V a_V
    dalpha_V/dxdot = 5`` gives ``Y_V_bar = 287 lb``, which is the Chapter 8
    vertical stabilizer force to three figures. The negative value would need
    606 lb. An independent chapter settles it.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    ================= ========== ==========
    output            model      Table 9.12
    ================= ========== ==========
    d_beta_d_ydot     .005152    .00515
    d_etaTV_d_xdot    .000618    .00061
    d_alphaV_d_xdot   .000618    .00061
    d_etaTV_d_ydot    -.001717   -.00167
    d_etaFV_d_ydot    .000309    .00031
    d_alphaV_d_ydot   -.003744   -.00379
    ================= ========== ==========

    ``deta_TV/dydot`` is the one row 3 % out. It is
    ``(dY/dydot)_T / [4 (q_V/q) q A_T]`` and nothing else, and the same
    denominator makes ``deta_TV/dxdot`` exact, so the gap is in the numerator:
    reproducing -.00167 needs ``(dY/dydot)_T = -23.8`` where Table 9.9 prints
    -24.5. Two tables of the same chapter disagree by 3 % on one derivative.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('V', val=np.full(nn, 194.1), units='ft/s')
        self.add_input('q', val=np.full(nn, 44.793), units='lbf/ft**2')
        self.add_input('qV_q', val=np.full(nn, 0.6),
                       desc='Dynamic pressure ratio at the vertical stabilizer.')
        self.add_input('A_T', val=132.73, units='ft**2',
                       desc='Tail rotor disc area.')
        self.add_input('T_T', val=np.full(nn, 661.0), units='lbf',
                       desc='Trim tail rotor thrust.')
        self.add_input('dY_dxdot_T', val=np.full(nn, -2.0), units='lbf*s/ft')
        self.add_input('dY_dydot_T', val=np.full(nn, -24.5), units='lbf*s/ft')
        self.add_input('detaF_dbeta', val=np.full(nn, 0.06),
                       desc='Fuselage sidewash slope, Chapter 8 p. 491.')

        for name, units in UNITS.items():
            self.add_output(name, val=np.zeros(nn), units=units)

        self.declare_partials('d_beta_d_ydot', 'V', rows=ar, cols=ar)
        wash_x = ['q', 'qV_q', 'T_T', 'dY_dxdot_T', 'V']
        wash_y = ['q', 'qV_q', 'dY_dydot_T']
        for name in ('d_etaTV_d_xdot', 'd_alphaV_d_xdot'):
            self.declare_partials(name, wash_x, rows=ar, cols=ar)
            self.declare_partials(name, 'A_T', rows=ar, cols=zeros)
        self.declare_partials('d_etaTV_d_ydot', wash_y, rows=ar, cols=ar)
        self.declare_partials('d_etaTV_d_ydot', 'A_T', rows=ar, cols=zeros)
        self.declare_partials('d_etaFV_d_ydot', ['detaF_dbeta', 'V'],
                              rows=ar, cols=ar)
        self.declare_partials('d_alphaV_d_ydot',
                              wash_y + ['detaF_dbeta', 'V'], rows=ar, cols=ar)
        self.declare_partials('d_alphaV_d_ydot', 'A_T', rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        V, scale = inputs['V'], (4.0 * inputs['qV_q'] * inputs['q']
                                 * inputs['A_T'][0])
        beta = 1.0 / V
        eta_T_x = -(inputs['dY_dxdot_T'] - 2.0 * inputs['T_T'] / V) / scale
        eta_T_y = inputs['dY_dydot_T'] / scale
        eta_F_y = inputs['detaF_dbeta'] * beta

        outputs['d_beta_d_ydot'] = beta
        outputs['d_etaTV_d_xdot'] = eta_T_x
        outputs['d_alphaV_d_xdot'] = eta_T_x
        outputs['d_etaTV_d_ydot'] = eta_T_y
        outputs['d_etaFV_d_ydot'] = eta_F_y
        outputs['d_alphaV_d_ydot'] = -(beta + eta_T_y + eta_F_y)

    def compute_partials(self, inputs, J):
        V, q, qV = inputs['V'], inputs['q'], inputs['qV_q']
        A_T, T_T = inputs['A_T'][0], inputs['T_T']
        Yx, Yy, etaF = (inputs['dY_dxdot_T'], inputs['dY_dydot_T'],
                        inputs['detaF_dbeta'])
        scale = 4.0 * qV * q * A_T
        eta_T_x = -(Yx - 2.0 * T_T / V) / scale
        eta_T_y = Yy / scale

        J['d_beta_d_ydot', 'V'] = -1.0 / V ** 2

        J['d_etaTV_d_xdot', 'dY_dxdot_T'] = -1.0 / scale
        J['d_etaTV_d_xdot', 'T_T'] = 2.0 / (V * scale)
        J['d_etaTV_d_xdot', 'V'] = -2.0 * T_T / (V ** 2 * scale)
        J['d_etaTV_d_xdot', 'q'] = -eta_T_x / q
        J['d_etaTV_d_xdot', 'qV_q'] = -eta_T_x / qV
        J['d_etaTV_d_xdot', 'A_T'] = -eta_T_x / A_T
        for name in ('dY_dxdot_T', 'T_T', 'V', 'q', 'qV_q', 'A_T'):
            J['d_alphaV_d_xdot', name] = J['d_etaTV_d_xdot', name]

        J['d_etaTV_d_ydot', 'dY_dydot_T'] = 1.0 / scale
        J['d_etaTV_d_ydot', 'q'] = -eta_T_y / q
        J['d_etaTV_d_ydot', 'qV_q'] = -eta_T_y / qV
        J['d_etaTV_d_ydot', 'A_T'] = -eta_T_y / A_T

        J['d_etaFV_d_ydot', 'detaF_dbeta'] = 1.0 / V
        J['d_etaFV_d_ydot', 'V'] = -etaF / V ** 2

        for name in ('dY_dydot_T', 'q', 'qV_q', 'A_T'):
            J['d_alphaV_d_ydot', name] = -J['d_etaTV_d_ydot', name]
        J['d_alphaV_d_ydot', 'detaF_dbeta'] = -1.0 / V
        J['d_alphaV_d_ydot', 'V'] = (1.0 + etaF) / V ** 2
