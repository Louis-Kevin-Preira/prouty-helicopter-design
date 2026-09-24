"""
ReturnToTargetComp -- G6, return-to-target maneuver integrated at fixed step.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Return-to-Target Maneuver" pp. 368-371, Figure 5.17.

Phase 1, banked decelerating turn at constant altitude (t in [0, t1]):
    x' = V cos psi,  y' = V sin psi,  psi' = g sqrt(n^2 - 1) / V   (R_T = V^2/(g sqrt(n^2-1)))
    V' = s(V) V_dot_auto(V),   n = s(V) n_auto(V) + (1 - s(V)) n_p
    s: cubic smoothstep over V_sw +/- w; below the autorotative limit V_sw the
    turn becomes a powered steady turn at n_p and constant speed (p. 371).
Phase 2, straight acceleration toward the target (t in [0, t2]):
    s' = V,  V' = acc(V)

n_auto, V_dot_auto, acc: Akima tables on V_grid (SpeedTable). Heun's method,
N steps per phase (dt = t1/N, t2/N). Durations are set outside by
    r_point = psi1 - (pi + atan(y1/x1))   heading through the target (3rd
                                           quadrant, x1 > 0; single root)
    r_dist  = s(t2) - d1                   target reached
Analytic partials: the sensitivities d(state)/d(inputs) are carried through
every Heun step (forward tangent), inputs stacked as
    p = [V_0, t1, t2, V_sw, n_p, n_tab (M), Vdot_tab (M), acc_tab (M)].

    --> r_point, r_dist, t_total, V_min, V_end, x, y, V1_hist (N+1,) (phase 1),
        V2_hist (N+1,) (phase 2, straight line from (x1, y1) to the origin)
"""

import numpy as np
import openmdao.api as om

from prouty.special_performance._table import SpeedTable
from prouty.vertical._smooth import smoothstep

G = 32.2       # ft/s^2, as printed in Chapter 5
EPS = 1e-4     # regularizes sqrt(n^2 - 1) at n = 1
SCALARS = ('V_0', 't1', 't2', 'V_sw', 'n_p')


class ReturnToTargetComp(om.ExplicitComponent):
    """Unrolled two-phase integration of the return-to-target maneuver."""

    def initialize(self):
        self.options.declare('V_grid', types=np.ndarray, desc='table speeds, ft/s, increasing')
        self.options.declare('num_steps', types=int, default=40)
        self.options.declare('switch_width', default=1.5, desc='ft/s')

    def setup(self):
        M = len(self.options['V_grid'])
        N = self.options['num_steps']
        self.add_input('V_0', val=115.0 * 1.6878, units='ft/s')
        self.add_input('t1', val=12.0, units='s')
        self.add_input('t2', val=10.0, units='s')
        self.add_input('V_sw', val=30.0 * 1.6878, units='ft/s')
        self.add_input('n_p', val=1.5)
        self.add_input('n_tab', val=2.0 * np.ones(M))
        self.add_input('Vdot_tab', val=-10.0 * np.ones(M), units='ft/s**2')
        self.add_input('acc_tab', val=10.0 * np.ones(M), units='ft/s**2')
        self.add_output('r_point', val=0.0, units='rad')
        self.add_output('r_dist', val=0.0, units='ft')
        self.add_output('t_total', val=20.0, units='s')
        self.add_output('V_min', val=50.0, units='ft/s')
        self.add_output('V_end', val=100.0, units='ft/s')
        for name in ('x', 'y', 'V1_hist', 'V2_hist'):
            self.add_output(name, val=np.zeros(N + 1), units='ft/s' if name[0] == 'V' else 'ft')
        self.declare_partials('*', '*')
        self.declare_partials('t_total', ['t1', 't2'], val=1.0)

    # ------------------------------------------------------------------ core
    def _integrate(self, inputs):
        """States and their sensitivities to p, both phases."""
        Vg, N, w = self.options['V_grid'], self.options['num_steps'], self.options['switch_width']
        M = len(Vg)
        P = 5 + 3 * M
        tab_n, tab_vd, tab_acc = slice(5, 5 + M), slice(5 + M, 5 + 2 * M), slice(5 + 2 * M, P)
        n_i = SpeedTable(Vg, inputs['n_tab'])
        vd_i = SpeedTable(Vg, inputs['Vdot_tab'])
        acc_i = SpeedTable(Vg, inputs['acc_tab'])
        V_sw, n_p = inputs['V_sw'][0], inputs['n_p'][0]

        def f1(z):
            _, _, psi, V = z
            s, ds = smoothstep(V, V_sw - w, V_sw + w)
            nA, dnA, dnA_tab = n_i(V)
            vd, dvd, dvd_tab = vd_i(V)
            n = s * nA + (1.0 - s) * n_p
            q = np.sqrt(n ** 2 - 1.0 + EPS ** 2)
            f = np.array([V * np.cos(psi), V * np.sin(psi), G * q / V, s * vd])
            Fz = np.zeros((4, 4))
            Fp = np.zeros((4, P))
            dn_dV = ds * (nA - n_p) + s * dnA
            c = G * n / (q * V)
            Fz[0, 2], Fz[0, 3] = -V * np.sin(psi), np.cos(psi)
            Fz[1, 2], Fz[1, 3] = V * np.cos(psi), np.sin(psi)
            Fz[2, 3] = c * dn_dV - G * q / V ** 2
            Fz[3, 3] = ds * vd + s * dvd
            Fp[2, 3] = c * (-ds) * (nA - n_p)             # V_sw
            Fp[2, 4] = c * (1.0 - s)                      # n_p
            Fp[2, tab_n] = c * s * dnA_tab
            Fp[3, 3] = -ds * vd
            Fp[3, tab_vd] = s * dvd_tab
            return f, Fz, Fp

        def f2(u):
            a, da, da_tab = acc_i(u[1])
            Fz = np.array([[0.0, 1.0], [0.0, da]])
            Fp = np.zeros((2, P))
            Fp[1, tab_acc] = da_tab
            return np.array([u[1], a]), Fz, Fp

        def heun(f, z, S, T, t_index):
            dt = T / N
            ddt = np.zeros(P)
            ddt[t_index] = 1.0 / N
            zs, Ss = [z], [S]
            for _ in range(N):
                k1, A1, B1 = f(z)
                dk1 = A1 @ S + B1
                zm = z + dt * k1
                Sm = S + dt * dk1 + np.outer(k1, ddt)
                k2, A2, B2 = f(zm)
                dk2 = A2 @ Sm + B2
                z = z + 0.5 * dt * (k1 + k2)
                S = S + 0.5 * dt * (dk1 + dk2) + 0.5 * np.outer(k1 + k2, ddt)
                zs.append(z)
                Ss.append(S)
            return np.array(zs), np.array(Ss)

        S0 = np.zeros((4, P))
        S0[3, 0] = 1.0
        z1, S1 = heun(f1, np.array([0.0, 0.0, 0.0, inputs['V_0'][0]]), S0,
                      inputs['t1'][0], 1)
        u0 = np.array([0.0, z1[-1, 3]])
        U0 = np.zeros((2, P))
        U0[1] = S1[-1, 3]
        z2, S2 = heun(f2, u0, U0, inputs['t2'][0], 2)
        return z1, S1, z2, S2

    def compute(self, inputs, outputs):
        z1, _, z2, _ = self._integrate(inputs)
        x1, y1, psi1, V1 = z1[-1]
        outputs['r_point'] = psi1 - (np.pi + np.arctan(y1 / x1))
        outputs['r_dist'] = z2[-1, 0] - np.hypot(x1, y1)
        outputs['t_total'] = inputs['t1'] + inputs['t2']
        outputs['V_min'] = V1
        outputs['V_end'] = z2[-1, 1]
        outputs['x'], outputs['y'], outputs['V1_hist'] = z1[:, 0], z1[:, 1], z1[:, 3]
        outputs['V2_hist'] = z2[:, 1]

    def compute_partials(self, inputs, J):
        z1, S1, z2, S2 = self._integrate(inputs)
        x1, y1 = z1[-1, 0], z1[-1, 1]
        d1 = np.hypot(x1, y1)
        r2 = x1 ** 2 + y1 ** 2
        d_point = S1[-1, 2] - (x1 * S1[-1, 1] - y1 * S1[-1, 0]) / r2
        d_dist = S2[-1, 0] - (x1 * S1[-1, 0] + y1 * S1[-1, 1]) / d1
        rows = {'r_point': d_point, 'r_dist': d_dist, 'V_min': S1[-1, 3], 'V_end': S2[-1, 1],
                'x': S1[:, 0], 'y': S1[:, 1], 'V1_hist': S1[:, 3], 'V2_hist': S2[:, 1]}
        M = len(self.options['V_grid'])
        cols = {name: k for k, name in enumerate(SCALARS)}
        cols.update(n_tab=slice(5, 5 + M), Vdot_tab=slice(5 + M, 5 + 2 * M),
                    acc_tab=slice(5 + 2 * M, 5 + 3 * M))
        for out, D in rows.items():
            D = np.atleast_2d(D)
            for name, c in cols.items():
                J[out, name] = D[:, c]
