"""
InducedVelocityRatioComp -- induced velocity ratio for the wake twist correction.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Drag in Hover", p. 279 and Figure 4.6 p. 281;
Chapter 1, step 5, p. 70 (inflow).

Figure 4.6 is measured under a rotor with -4 deg twist. For another twist,
p. 279 scales q/D.L. by the square of the ratio of induced velocities computed
by the Chapter 1 method for the two twists. Both rotors are run by
HoverRotorGroup at the same thrust and tip speed, so the ratio of induced
velocities is the ratio of the inflow ratios v1/(Omega r) at the same station:

    vi_ratio = v1_Or(r_rotor) / v1_Or_ref(r_rotor)

A segment sits in the wake at (r/R, z/R). The wake has contracted there, so it
is fed by the rotor station that maps to the wake edge (C4-12):

    r_rotor = (r/R) / R_edge(z/R),   R_edge = 0.867, 0.849, 0.845 at z/R = 0.1, 0.3, 0.5

(R_edge where the measured q/D.L. returns to zero, linear in z/R, nearest
panel held outside, as in WakeDynamicPressureComp.) Along the blade the inflow
is interpolated linearly between the stations and held at the end stations.

    r_R, v1_Or, v1_Or_ref (ns,), seg_r_R, seg_z_R (n,) --> vi_ratio (n,)
"""

import numpy as np
import openmdao.api as om

from prouty.performance.wake_dynamic_pressure_comp import WAKE_EDGE, Z_PANELS


def _linear(x_nodes, x):
    """Interval index, weight t, dt/dx, dt/dx_i, dt/dx_i+1 for linear interpolation."""
    xr = np.real(x_nodes)
    i = np.clip(np.searchsorted(xr, np.real(x), side='right') - 1, 0, len(xr) - 2)
    x0, x1 = x_nodes[i], x_nodes[i + 1]
    h = x1 - x0
    inside = (np.real(x) >= xr[0]) & (np.real(x) <= xr[-1])
    xc = np.where(inside, x, np.where(np.real(x) < xr[0], x0, x1))
    t = (xc - x0) / h
    held = ~inside                      # value held at an end node, no slope anywhere
    return (i, t, np.where(held, 0.0, 1.0 / h), np.where(held, 0.0, (xc - x1) / h ** 2),
            np.where(held, 0.0, -(xc - x0) / h ** 2))


class InducedVelocityRatioComp(om.ExplicitComponent):
    """Ratio of Chapter 1 inflows, rotor twist over -4 deg, at each segment."""

    def initialize(self):
        self.options.declare('num_segments', types=int, default=1)
        self.options.declare('num_stations', types=int, default=11,
                             desc='blade stations of HoverRotorGroup, num_elements + 1')

    def setup(self):
        n, ns = self.options['num_segments'], self.options['num_stations']

        self.add_input('r_R', val=np.linspace(0.15, 1.0, ns), desc='blade stations, r/R')
        self.add_input('v1_Or', val=np.full(ns, 0.1), desc='inflow ratio, rotor twist')
        self.add_input('v1_Or_ref', val=np.full(ns, 0.1), desc='inflow ratio, -4 deg twist')
        self.add_input('seg_r_R', val=np.full(n, 0.5), desc='segment radial position, r/R')
        self.add_input('seg_z_R', val=np.full(n, 0.2), desc='segment distance below the rotor, z/R')
        self.add_output('vi_ratio', val=np.ones(n))

        ar = np.arange(n)
        self.declare_partials('vi_ratio', ['r_R', 'v1_Or', 'v1_Or_ref'])
        self.declare_partials('vi_ratio', ['seg_r_R', 'seg_z_R'], rows=ar, cols=ar)

    def _evaluate(self, inputs):
        z = inputs['seg_z_R']
        k, tz, dtz_dz, _, _ = _linear(Z_PANELS, z)
        edge = WAKE_EDGE[k] + tz * (WAKE_EDGE[k + 1] - WAKE_EDGE[k])
        dedge_dz = dtz_dz * (WAKE_EDGE[k + 1] - WAKE_EDGE[k])

        x = inputs['seg_r_R'] / edge
        dx_dr, dx_dz = 1.0 / edge, -x / edge * dedge_dz

        i, t, dt_dx, dt_dx0, dt_dx1 = _linear(inputs['r_R'], x)
        v, v_ref = inputs['v1_Or'], inputs['v1_Or_ref']
        a = v[i] + t * (v[i + 1] - v[i])
        b = v_ref[i] + t * (v_ref[i + 1] - v_ref[i])
        return dict(ratio=a / b, a=a, b=b, i=i, t=t, dt_dx=dt_dx, dt_dx0=dt_dx0, dt_dx1=dt_dx1,
                    dv=v[i + 1] - v[i], dv_ref=v_ref[i + 1] - v_ref[i], dx_dr=dx_dr, dx_dz=dx_dz)

    def compute(self, inputs, outputs):
        outputs['vi_ratio'] = self._evaluate(inputs)['ratio']

    def compute_partials(self, inputs, partials):
        e = self._evaluate(inputs)
        n, ns = self.options['num_segments'], self.options['num_stations']
        ar, i, t = np.arange(n), e['i'], e['t']
        da, db = 1.0 / e['b'], -e['a'] / e['b'] ** 2          # d ratio / d a, d b

        J_v, J_ref, J_r = np.zeros((n, ns)), np.zeros((n, ns)), np.zeros((n, ns))
        np.add.at(J_v, (ar, i), da * (1.0 - t))
        np.add.at(J_v, (ar, i + 1), da * t)
        np.add.at(J_ref, (ar, i), db * (1.0 - t))
        np.add.at(J_ref, (ar, i + 1), db * t)
        slope = da * e['dv'] + db * e['dv_ref']              # d ratio / d t
        np.add.at(J_r, (ar, i), slope * e['dt_dx0'])
        np.add.at(J_r, (ar, i + 1), slope * e['dt_dx1'])

        partials['vi_ratio', 'v1_Or'] = J_v
        partials['vi_ratio', 'v1_Or_ref'] = J_ref
        partials['vi_ratio', 'r_R'] = J_r
        partials['vi_ratio', 'seg_r_R'] = slope * e['dt_dx'] * e['dx_dr']
        partials['vi_ratio', 'seg_z_R'] = slope * e['dt_dx'] * e['dx_dz']
