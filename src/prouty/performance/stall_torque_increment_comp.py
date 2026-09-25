"""
StallTorqueIncrementComp -- retreating blade stall torque increment, calibrated
on the rotor charts of Chapter 3 (correction C5-6).

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, rotor charts pp. 258-266 (mu = 0.20-0.40, theta_1 = -5 deg,
M_1,90 = 0.7) and the twist correction of p. 230.

    dC_Q/sigma = [C_Q/sigma chart - C_Q/sigma closed form](mu, C_T/sigma_eff, X mu^3)
                 - per-curve low-load baseline (mean over C_T/sigma = 0.05-0.07)

The table (data/stall_increment.json, 40 theta_0 curves) is regridded per
plate on (C_T/sigma, X mu^3) -- across the curves at each C_T/sigma -- and
interpolated by Akima in 3-D. X mu^3 =
-2 (C_T/sigma) lambda' is the chart ordinate X times mu^3: a one-to-one
change of variable at given mu that puts the five plates on one axis.
Outside the charted region the table is extended flat (nearest value);
mu is held to [0.20, 0.40].

    C_T/sigma_eff = C_T/sigma + 0.003 (theta_1 + 5 deg)           p. 230
                    - (a / 6) d_alpha_stall                         airfoil stall angle shift
    output: smooth floor at 0 (quadratic fillet, w = 0.0002)

    mu, CT_sigma, lambda_p (nn,), theta_1, a, d_alpha_stall --> dCQ_sigma_stall (nn,)
"""

import json
import pathlib

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

from prouty.performance._smooth import smoothmin

DATA = pathlib.Path(__file__).parents[1] / 'special_performance' / 'data' / 'stall_increment.json'
MU_GRID = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
CT_GRID = np.round(np.arange(0.03, 0.1301, 0.0025), 5)
XM_GRID = np.round(np.arange(-0.016, 0.0301, 0.002), 5)
TWIST_SHIFT = 0.003          # dC_T/sigma per degree of twist, p. 230
FLOOR = 0.0002


def build_table():
    """Stall increment on (mu, C_T/sigma, X mu^3) from the digitized charts.

    Per plate and per C_T/sigma row, the theta_0 curves present at that
    C_T/sigma are interpolated linearly in X mu^3 and held flat beyond the
    outermost curves; rows beyond the last charted C_T/sigma repeat the last
    charted row (flat extension, no extrapolated stall growth)."""
    data = json.load(open(DATA))
    table = np.zeros((len(MU_GRID), len(CT_GRID), len(XM_GRID)))
    for i, mu in enumerate(MU_GRID):
        curves = []
        for curve in data[f'{mu:.2f}'].values():
            c = np.array(curve['CT_sigma'])
            curves.append((c, np.array(curve['X']) * mu ** 3, np.array(curve['dCQ_stall'])))
        last = None
        for j, ct in enumerate(CT_GRID):
            row = [(np.interp(ct, c, xm), np.interp(ct, c, d)) for c, xm, d in curves
                   if c[0] <= ct <= c[-1]]
            if len(row) >= 2:
                row.sort()
                xs, ds = np.array([r[0] for r in row]), np.array([r[1] for r in row])
                last = np.interp(XM_GRID, xs, ds)
            elif last is None:
                last = np.zeros(len(XM_GRID))
            table[i, j] = last
    return table


class StallTorqueIncrementComp(om.ExplicitComponent):
    """Chart-calibrated stall torque increment, added after the trim (C5-6)."""

    _table = None

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        if StallTorqueIncrementComp._table is None:
            StallTorqueIncrementComp._table = build_table()
        self._interp = InterpND(method='akima', points=(MU_GRID, CT_GRID, XM_GRID),
                                values=StallTorqueIncrementComp._table, extrapolate=True)
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('mu', val=0.3 * np.ones(nn))
        self.add_input('CT_sigma', val=0.08 * np.ones(nn))
        self.add_input('lambda_p', val=-0.02 * np.ones(nn))
        self.add_input('theta_1', val=-0.17453, units='rad')
        self.add_input('a', val=6.0, units='1/rad')
        self.add_input('d_alpha_stall', val=0.0, units='rad')
        self.add_output('CT_sigma_eff', val=0.08 * np.ones(nn))
        self.add_output('dCQ_sigma_stall', val=np.zeros(nn))
        self.declare_partials('CT_sigma_eff', 'CT_sigma', rows=ar, cols=ar, val=1.0)
        self.declare_partials('CT_sigma_eff', ['theta_1', 'a', 'd_alpha_stall'], rows=ar, cols=zero)
        self.declare_partials('dCQ_sigma_stall', ['mu', 'CT_sigma', 'lambda_p'], rows=ar, cols=ar)
        self.declare_partials('dCQ_sigma_stall', ['theta_1', 'a', 'd_alpha_stall'],
                              rows=ar, cols=zero)

    @staticmethod
    def _clamp(x, lo, hi):
        """Smooth clamp (quadratic fillets) and its derivative."""
        w = 0.02 * (hi - lo)
        v1, d1, _ = smoothmin(x, hi * np.ones_like(x), w)
        v2, d2, _ = smoothmin(-v1, -lo * np.ones_like(x), w)
        return -v2, d1 * d2

    def _eval(self, inputs):
        i = inputs
        ct_eff = (i['CT_sigma'] + TWIST_SHIFT * (np.degrees(i['theta_1']) + 5.0)
                  - i['a'] / 6.0 * i['d_alpha_stall'])
        xm = -2.0 * i['CT_sigma'] * i['lambda_p']
        mu_c, dmu = self._clamp(i['mu'], MU_GRID[0], MU_GRID[-1])
        ct_c, dct = self._clamp(ct_eff, CT_GRID[0], CT_GRID[-1])
        xm_c, dxm = self._clamp(xm, XM_GRID[0], XM_GRID[-1])
        raw, d = self._interp.interpolate(np.column_stack([mu_c, ct_c, xm_c]),
                                          compute_derivative=True)
        # smooth floor at 0: max(raw, 0) = -min(-raw, 0)
        v, dv, _ = smoothmin(-raw, np.zeros_like(raw), FLOOR)
        return ct_eff, xm, -v, dv, d, dmu, dct, dxm

    def compute(self, inputs, outputs):
        ct_eff, _, out, *_ = self._eval(inputs)
        outputs['CT_sigma_eff'] = ct_eff
        outputs['dCQ_sigma_stall'] = out

    def compute_partials(self, inputs, J):
        i = inputs
        _, _, _, dv, d, dmu, dct, dxm = self._eval(inputs)
        g_mu, g_ct, g_xm = (dv * d[:, k] for k in range(3))
        g_ct, g_xm = g_ct * dct, g_xm * dxm
        J['CT_sigma_eff', 'theta_1'] = TWIST_SHIFT * np.degrees(1.0) * np.ones_like(g_ct)
        J['CT_sigma_eff', 'a'] = -i['d_alpha_stall'] / 6.0 * np.ones_like(g_ct)
        J['CT_sigma_eff', 'd_alpha_stall'] = -i['a'] / 6.0 * np.ones_like(g_ct)
        J['dCQ_sigma_stall', 'mu'] = g_mu * dmu
        J['dCQ_sigma_stall', 'CT_sigma'] = g_ct + g_xm * (-2.0 * i['lambda_p'])
        J['dCQ_sigma_stall', 'lambda_p'] = g_xm * (-2.0 * i['CT_sigma'])
        J['dCQ_sigma_stall', 'theta_1'] = g_ct * TWIST_SHIFT * np.degrees(1.0)
        J['dCQ_sigma_stall', 'a'] = g_ct * (-i['d_alpha_stall'] / 6.0)
        J['dCQ_sigma_stall', 'd_alpha_stall'] = g_ct * (-i['a'] / 6.0)
