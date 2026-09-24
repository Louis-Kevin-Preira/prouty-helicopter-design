"""
ReturnToTargetComp -- G6, return-to-target maneuver integrated at fixed step.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Return-to-Target Maneuver" pp. 368-371, Figure 5.17.

Phase 1, banked decelerating turn at constant altitude (t in [0, t1]):
    x' = V cos psi,  y' = V sin psi,  psi' = g sqrt(n^2 - 1) / V   (R_T = V^2/(g sqrt(n^2-1)))
    V' = s(V) V_dot_auto(V),   n = s(V) n_auto(V) + (1 - s(V)) n_p
    s(V) = (1 + tanh((V - V_sw)/w))/2: below the autorotative limit V_sw the
    turn becomes a powered steady turn at n_p and constant speed (p. 371).
Phase 2, straight acceleration toward the target (t in [0, t2]):
    s' = V,  V' = acc(V)

n_auto, V_dot_auto and acc are tables on the speed grid, interpolated by a
C1 cubic Hermite with finite-difference slopes (written out so that complex
step goes through; OpenMDAO's Akima rejects complex values). Heun's
method with N steps per phase (dt = t1/N, t2/N). Durations are set outside by
the two residuals:
    r_point = psi1 - (pi + atan(y1/x1))         heading through the target (3rd quadrant,
                                                x1 > 0); a single root, unlike the cross product
    r_dist  = s(t2) - d1                        target reached
Partials by complex step (the integration is unrolled in this component).

    V_0, t1, t2, V_sw, n_p, n_tab, Vdot_tab, acc_tab (M,) -->
        r_point, r_dist, t_total, V_min, V_end, x, y, V1_hist (N+1,) (phase 1),
        V2_hist (N+1,) (phase 2, straight line from (x1, y1) to the origin)
"""

import numpy as np
import openmdao.api as om

G = 32.2       # ft/s^2, as printed in Chapter 5
EPS = 1e-4     # regularizes sqrt(n^2 - 1) at n = 1


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
        self.declare_partials('*', '*', method='cs')

    def _interp(self, values):
        xg = self.options['V_grid']
        m = np.gradient(values, xg)

        def f(v):
            i = int(np.clip(np.searchsorted(xg, v.real) - 1, 0, len(xg) - 2))
            h = xg[i + 1] - xg[i]
            s = (v - xg[i]) / h
            return ((2 * s ** 3 - 3 * s ** 2 + 1) * values[i] + (s ** 3 - 2 * s ** 2 + s) * h * m[i]
                    + (-2 * s ** 3 + 3 * s ** 2) * values[i + 1] + (s ** 3 - s ** 2) * h * m[i + 1])
        return f

    def compute(self, inputs, outputs):
        N, w = self.options['num_steps'], self.options['switch_width']
        n_i = self._interp(inputs['n_tab'])
        vd_i = self._interp(inputs['Vdot_tab'])
        acc_i = self._interp(inputs['acc_tab'])
        V_sw, n_p = inputs['V_sw'][0], inputs['n_p'][0]

        def f1(z):
            x, y, psi, V = z
            s = 0.5 * (1.0 + np.tanh((V - V_sw) / w))
            n = s * n_i(V) + (1.0 - s) * n_p
            V_dot = s * vd_i(V)
            return np.array([V * np.cos(psi), V * np.sin(psi),
                             G * np.sqrt(n ** 2 - 1.0 + EPS ** 2) / V, V_dot])

        def f2(z):
            return np.array([z[1], acc_i(z[1])])

        def heun(f, z, dt):
            hist = [z]
            for _ in range(N):
                k1 = f(z)
                k2 = f(z + dt * k1)
                z = z + 0.5 * dt * (k1 + k2)
                hist.append(z)
            return np.array(hist)

        z0 = np.array([0.0, 0.0, 0.0, inputs['V_0'][0]])
        h1 = heun(f1, z0, inputs['t1'][0] / N)
        x1, y1, psi1, V1 = h1[-1]
        d1 = np.sqrt(x1 ** 2 + y1 ** 2)
        h2 = heun(f2, np.array([0.0 * x1, V1]), inputs['t2'][0] / N)

        outputs['r_point'] = psi1 - (np.pi + np.arctan(y1 / x1))
        outputs['r_dist'] = h2[-1, 0] - d1
        outputs['t_total'] = inputs['t1'] + inputs['t2']
        outputs['V_min'] = V1
        outputs['V_end'] = h2[-1, 1]
        outputs['x'] = h1[:, 0]
        outputs['y'] = h1[:, 1]
        outputs['V1_hist'] = h1[:, 3]
        outputs['V2_hist'] = h2[:, 1]
