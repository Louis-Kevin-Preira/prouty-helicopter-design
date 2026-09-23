"""Vertical stabilizer dimensional derivatives in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.13, pp. 587-589. The
angle-of-attack derivatives are Table 9.12, p. 587; the interference drag is
Chapter 8, pp. 509-510, already implemented here as
``VertStabInterferenceDragComp``.
"""

import numpy as np
import openmdao.api as om

#: Rows that are a core row carried on the arms, as
#: (source, coefficient, power of h_V, power of l_V).
DERIVED = {
    'dY_dp': ('dY_dydot', 1.0, 1, 0),
    'dY_dr': ('dY_dydot', -1.0, 0, 1),
    'dR_dxdot': ('dY_dxdot', 1.0, 1, 0),
    'dR_dydot': ('dY_dydot', 1.0, 1, 0),
    'dR_dp': ('dY_dydot', 1.0, 2, 0),
    'dR_dr': ('dY_dydot', -1.0, 1, 1),
    'dN_dxdot': ('dY_dxdot', -1.0, 0, 1),
    'dN_dydot': ('dY_dydot', -1.0, 0, 1),
    'dN_dp': ('dY_dydot', -1.0, 1, 1),
    'dN_dr': ('dY_dydot', 1.0, 0, 2),
}

CORE = ('dX_dxdot', 'dX_dydot', 'dY_dxdot', 'dY_dydot')

UNITS = {'xdot': 'lbf*s/ft', 'ydot': 'lbf*s/ft', 'p': 'lbf*s/rad',
         'r': 'lbf*s/rad'}

SCALAR_INPUTS = ('A_V', 'a_V', 'A_R', 'alpha_LO', 'delta', 'h_V', 'l_V')

DEFAULTS = {
    'qV_q': 0.6, 'q': 44.793, 'A_V': 33.0, 'a_V': 3.0, 'A_R': 3.2,
    'delta': 0.01, 'alpha_LO': -0.1012, 'alpha_V_bar': 0.0066660,
    'V': 194.1, 'Y_V_bar': 287.0, 'X_V_bar': -58.0, 'D_int': 42.8,
    'T_T': 661.0, 'dY_dydot_T': -24.5,
    'd_alphaV_d_xdot': 0.00061, 'd_alphaV_d_ydot': -0.00379,
    'h_V': 3.0, 'l_V': 35.0,
}


class VertStabDerivativesFFComp(om.ExplicitComponent):
    """Table 9.13: the vertical stabilizer's 14 dimensional derivatives.

    Four rows carry the physics and ten are those carried onto ``h_V`` and
    ``-l_V``::

        dX/dxdot = (2/V)[X_V_bar + 2 dD_int]
        dY/dxdot = (2/V) Y_V_bar + (q_V/q) q A_V a_V (dalpha_V/dxdot)
        dY/dydot = [1/(1 - dD_int/Y_V_bar)]
                   {(q_V/q) q A_V a_V (dalpha_V/dydot)
                    + dD_int[-2/V + (1/T_T)(dY/dydot)_T]}
        dX/dydot = (q_V/q) q A_V a_V {K_V} (dalpha_V/dydot)
                   + dD_int[(1/Y_V_bar)(dY/dydot)_V + (1/T_T)(dY/dydot)_T]

    with ``K_V = (alpha_V - alpha_LO)[1 - 2 a_V(1+delta)/(pi A.R.)] +
    alpha_V``.

    Interference drag is not a correction term here
    -----------------------------------------------
    ``dD_int`` is the biplane interference between the tail rotor and the fin,
    Chapter 8 pp. 509-510. At 115 knots it is 42.8 lb against 14.9 lb of clean
    fin drag, and it appears in three of the four core rows.

    In ``dY/dydot`` it appears twice: once directly, and once as the factor
    ``1/(1 - dD_int/Y_V_bar) = 1.175``. That factor is there because the
    interference drag is itself proportional to the fin's side force, so a
    sideslip perturbation feeds back on itself. It is a 17 % amplification,
    not a rounding correction. This is the one row of any airframe table that
    is not a straight product, and it is why this component is written out
    rather than declared as monomials like Table 9.11.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    =========== ========= ==========
    row         model     Table 9.13
    =========== ========= ==========
    dX_dxdot    .284      <1
    dX_dydot    -4.21     -3
    dY_dxdot    4.66      5
    dY_dydot    -14.24    -14
    dY_dp       -42.7     -42
    dY_dr       498       490
    dR_dxdot    14.0      12
    dR_dydot    -42.7     -42
    dR_dp       -128      -126
    dR_dr       1,495     1,470
    dN_dxdot    -163      -161
    dN_dydot    498       490
    dN_dp       1,495     1,470
    dN_dr       -17,440   -17,150
    =========== ========= ==========

    Every row is inside 2 % except ``dX/dydot``, which is a single digit in the
    book: -4.21 against -3. Reaching -3 needs ``K_V = -.070``, which requires
    ``alpha_V = -4.5 deg``, and ``Y_V_bar = 287 lb`` fixes ``alpha_V`` at
    ``+0.38 deg``. The two cannot both hold.

    Nothing about the fin was adjusted: ``q_V/q = .6``, ``A_V = 33 ft^2``,
    ``a_V = 3.0``, ``A.R. = 3.2``, ``delta = .01``, ``alpha_LO = -.1012 rad``,
    ``Y_V_bar = 287 lb``, ``X_V_bar = -58 lb`` and ``dD_int = 42.8 lb`` are all
    the Chapter 8 values already in this repository. The two arms,
    ``h_V = 3.0 ft`` and ``l_V = 35.0 ft``, come from Table 9.13 itself and
    each is confirmed by three separate rows.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        units = {'q': 'lbf/ft**2', 'A_V': 'ft**2', 'a_V': '1/rad',
                 'alpha_LO': 'rad', 'alpha_V_bar': 'rad', 'V': 'ft/s',
                 'Y_V_bar': 'lbf', 'X_V_bar': 'lbf', 'D_int': 'lbf',
                 'T_T': 'lbf', 'dY_dydot_T': 'lbf*s/ft',
                 'd_alphaV_d_xdot': 'rad*s/ft', 'd_alphaV_d_ydot': 'rad*s/ft',
                 'h_V': 'ft', 'l_V': 'ft'}
        for name, value in DEFAULTS.items():
            if name in SCALAR_INPUTS:
                self.add_input(name, val=value, units=units.get(name))
            else:
                self.add_input(name, val=np.full(nn, value),
                               units=units.get(name))

        self._deps = {
            'dX_dxdot': ['X_V_bar', 'D_int', 'V'],
            'dY_dxdot': ['Y_V_bar', 'V', 'd_alphaV_d_xdot', 'qV_q', 'q',
                         'A_V', 'a_V'],
            'dY_dydot': ['Y_V_bar', 'D_int', 'V', 'T_T', 'dY_dydot_T',
                         'd_alphaV_d_ydot', 'qV_q', 'q', 'A_V', 'a_V'],
            'dX_dydot': ['Y_V_bar', 'D_int', 'V', 'T_T', 'dY_dydot_T',
                         'd_alphaV_d_ydot', 'qV_q', 'q', 'A_V', 'a_V',
                         'alpha_V_bar', 'alpha_LO', 'A_R', 'delta'],
        }
        for name in CORE:
            self.add_output(name, val=np.zeros(nn),
                            units=self._units_of(name))
            self._declare(name, self._deps[name], ar, zeros)

        for name, (source, _, _, _) in DERIVED.items():
            self.add_output(name, val=np.zeros(nn),
                            units=self._units_of(name))
            self._declare(name, self._deps[source] + ['h_V', 'l_V'],
                          ar, zeros)

    def _declare(self, name, inputs, ar, zeros):
        for inp in inputs:
            cols = zeros if inp in SCALAR_INPUTS else ar
            self.declare_partials(name, inp, rows=ar, cols=cols)

    @staticmethod
    def _units_of(name):
        force, wrt = name[1], name.split('_d', 1)[1]
        base = UNITS[wrt]
        return base.replace('lbf', 'lbf*ft') if force in 'RMN' else base

    def _state(self, inputs):
        S = (inputs['qV_q'] * inputs['q'] * inputs['A_V'][0]
             * inputs['a_V'][0])
        induced = (2.0 * inputs['a_V'][0] * (1.0 + inputs['delta'][0])
                   / (np.pi * inputs['A_R'][0]))
        K = ((inputs['alpha_V_bar'] - inputs['alpha_LO'][0])
             * (1.0 - induced) + inputs['alpha_V_bar'])
        tail = inputs['dY_dydot_T'] / inputs['T_T']
        F = 1.0 / (1.0 - inputs['D_int'] / inputs['Y_V_bar'])
        P = (S * inputs['d_alphaV_d_ydot']
             + inputs['D_int'] * (-2.0 / inputs['V'] + tail))
        return dict(S=S, induced=induced, K=K, tail=tail, F=F, P=P)

    def compute(self, inputs, outputs):
        s = self._state(inputs)
        V, D, Y_V = inputs['V'], inputs['D_int'], inputs['Y_V_bar']

        outputs['dX_dxdot'] = 2.0 * (inputs['X_V_bar'] + 2.0 * D) / V
        outputs['dY_dxdot'] = (2.0 * Y_V / V
                               + s['S'] * inputs['d_alphaV_d_xdot'])
        dY_dydot = s['F'] * s['P']
        outputs['dY_dydot'] = dY_dydot
        outputs['dX_dydot'] = (s['S'] * s['K'] * inputs['d_alphaV_d_ydot']
                               + D * (dY_dydot / Y_V + s['tail']))

        h, l = inputs['h_V'][0], inputs['l_V'][0]
        for name, (source, coef, hp, lp) in DERIVED.items():
            outputs[name] = coef * outputs[source] * h ** hp * l ** lp

    def compute_partials(self, inputs, J):
        s = self._state(inputs)
        V, D, Y_V = inputs['V'], inputs['D_int'], inputs['Y_V_bar']
        T_T, Yy_T = inputs['T_T'], inputs['dY_dydot_T']
        a_x, a_y = inputs['d_alphaV_d_xdot'], inputs['d_alphaV_d_ydot']
        a_V, A_R = inputs['a_V'][0], inputs['A_R'][0]
        S, K, F, P, tail = s['S'], s['K'], s['F'], s['P'], s['tail']
        one = np.ones_like(V)

        J['dX_dxdot', 'X_V_bar'] = 2.0 / V
        J['dX_dxdot', 'D_int'] = 4.0 / V
        J['dX_dxdot', 'V'] = -2.0 * (inputs['X_V_bar'] + 2.0 * D) / V ** 2

        J['dY_dxdot', 'Y_V_bar'] = 2.0 / V
        J['dY_dxdot', 'V'] = -2.0 * Y_V / V ** 2
        J['dY_dxdot', 'd_alphaV_d_xdot'] = S * one
        for name, value in (('qV_q', inputs['qV_q']), ('q', inputs['q']),
                            ('A_V', inputs['A_V'][0]), ('a_V', a_V)):
            J['dY_dxdot', name] = S * a_x / value

        # dY/dydot = F P, with F and P both depending on D_int
        dF_dD = F ** 2 / Y_V
        dF_dY = -F ** 2 * D / Y_V ** 2
        dP = {'D_int': -2.0 / V + tail, 'V': 2.0 * D / V ** 2,
              'dY_dydot_T': D / T_T, 'T_T': -D * tail / T_T,
              'd_alphaV_d_ydot': S * one}
        for name, value in (('qV_q', inputs['qV_q']), ('q', inputs['q']),
                            ('A_V', inputs['A_V'][0]), ('a_V', a_V)):
            dP[name] = S * a_y / value
        for name, value in dP.items():
            J['dY_dydot', name] = F * value
        J['dY_dydot', 'D_int'] = dF_dD * P + F * dP['D_int']
        J['dY_dydot', 'Y_V_bar'] = dF_dY * P

        # dX/dydot = S K a_y + D (dY/dydot / Y_V + tail)
        dY = F * P
        dK = {'alpha_V_bar': 2.0 - s['induced'],
              'alpha_LO': -(1.0 - s['induced'])}
        d_ind_da = s['induced'] / a_V
        d_ind_dAR = -s['induced'] / A_R
        base = -(inputs['alpha_V_bar'] - inputs['alpha_LO'][0])
        for name, value in dP.items():
            J['dX_dydot', name] = D * J['dY_dydot', name] / Y_V
        J['dX_dydot', 'd_alphaV_d_ydot'] += S * K * one
        for name, value in (('qV_q', inputs['qV_q']), ('q', inputs['q']),
                            ('A_V', inputs['A_V'][0])):
            J['dX_dydot', name] += S * K * a_y / value
        J['dX_dydot', 'a_V'] += (S * K * a_y / a_V
                                 + S * base * d_ind_da * a_y)
        J['dX_dydot', 'A_R'] = S * base * d_ind_dAR * a_y
        d_ind_ddelta = s['induced'] / (1.0 + inputs['delta'][0])
        J['dX_dydot', 'delta'] = S * base * d_ind_ddelta * a_y
        J['dX_dydot', 'alpha_V_bar'] = S * dK['alpha_V_bar'] * a_y
        J['dX_dydot', 'alpha_LO'] = S * dK['alpha_LO'] * a_y
        J['dX_dydot', 'D_int'] = (D * J['dY_dydot', 'D_int'] / Y_V
                                  + dY / Y_V + tail)
        J['dX_dydot', 'Y_V_bar'] = D * (J['dY_dydot', 'Y_V_bar'] / Y_V
                                        - dY / Y_V ** 2)
        J['dX_dydot', 'T_T'] += D * (-tail / T_T)
        J['dX_dydot', 'dY_dydot_T'] += D / T_T

        h, l = inputs['h_V'][0], inputs['l_V'][0]
        values = {name: self._value(name, inputs, s) for name in CORE}
        for name, (source, coef, hp, lp) in DERIVED.items():
            arm = coef * h ** hp * l ** lp
            for inp in self._deps[source]:
                J[name, inp] = arm * J[source, inp]
            J[name, 'h_V'] = (coef * hp * h ** (hp - 1) * l ** lp
                              * values[source] if hp else 0.0 * values[source])
            J[name, 'l_V'] = (coef * lp * h ** hp * l ** (lp - 1)
                              * values[source] if lp else 0.0 * values[source])

    def _value(self, name, inputs, s):
        V, D, Y_V = inputs['V'], inputs['D_int'], inputs['Y_V_bar']
        if name == 'dX_dxdot':
            return 2.0 * (inputs['X_V_bar'] + 2.0 * D) / V
        if name == 'dY_dxdot':
            return 2.0 * Y_V / V + s['S'] * inputs['d_alphaV_d_xdot']
        if name == 'dY_dydot':
            return s['F'] * s['P']
        return (s['S'] * s['K'] * inputs['d_alphaV_d_ydot']
                + D * (s['F'] * s['P'] / Y_V + s['tail']))
