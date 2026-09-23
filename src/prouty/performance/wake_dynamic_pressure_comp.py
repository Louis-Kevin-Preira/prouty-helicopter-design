"""
WakeDynamicPressureComp -- dynamic pressure in the hover wake at the airframe.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Drag in Hover", pp. 278-282: method p. 278, Figure 4.6
p. 281, Table 4.1 p. 282.

Each airframe segment n sits at (r/R, z/R) below the rotor. Its dynamic
pressure ratio (q/D.L.)_n is read from the distribution measured under a
full-scale rotor with -4 deg twist (Boatwright, Figure 4.6). For another
twist, p. 279 multiplies it by the square of the ratio of induced
velocities computed by the Chapter 1 method for the two twists:

    (q/D.L.)_n = (q/D.L.)_-4(r/R, z/R) * vi_ratio_n^2
    vi_ratio   = v_i(r, twist) / v_i(r, -4 deg)        (computed upstream)

Figure 4.6 -- pixel digitization of the three panels z/R = 0.1, 0.3, 0.5
(C4-11). Both curves are kept: the measured -4 deg one is the model, the
extrapolated -10 deg one validates the twist correction and matches the
readings of Table 4.1.

Interpolation: Akima in r/R at each panel (C1, no overshoot at the tip
peak), linear in z/R between panels. Outside 0.1 <= z/R <= 0.5 the nearest
panel is held and a warning is raised. The wake is zero outside the table.

    seg_r_R, seg_z_R, vi_ratio (n,) --> q_DL (n,)

Segment positions are prefixed seg_ so they never clash with the blade
stations r_R of the Chapter 1 rotor in the same model.
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

Z_PANELS = np.array([0.1, 0.3, 0.5])
WAKE_EDGE = np.array([0.867, 0.849, 0.845])     # r/R where q/D.L. returns to zero, -4 deg

# Figure 4.6 p. 281: radius stations of the six curves, finer at the tip
_R = np.array([
    0.000, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090,
    0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190,
    0.200, 0.210, 0.220, 0.230, 0.240, 0.250, 0.260, 0.270, 0.280, 0.290,
    0.300, 0.310, 0.320, 0.330, 0.340, 0.350, 0.360, 0.370, 0.380, 0.390,
    0.400, 0.410, 0.420, 0.430, 0.440, 0.450, 0.460, 0.470, 0.480, 0.490,
    0.500, 0.510, 0.520, 0.530, 0.540, 0.550, 0.560, 0.570, 0.580, 0.590,
    0.600, 0.610, 0.620, 0.630, 0.640, 0.650, 0.660, 0.670, 0.680, 0.690,
    0.700, 0.710, 0.720, 0.730, 0.740, 0.750, 0.755, 0.760, 0.765, 0.770,
    0.775, 0.780, 0.785, 0.790, 0.795, 0.800, 0.805, 0.810, 0.815, 0.820,
    0.825, 0.830, 0.835, 0.840, 0.845, 0.850, 0.855, 0.860, 0.865, 0.870,
    0.875, 0.880, 0.885, 0.890, 0.895, 0.900, 0.910, 0.920, 0.930, 0.940,
    0.950, 0.960, 0.970, 0.980, 0.990, 1.000])

# q/D.L. measured, -4 deg twist; rows z/R = 0.1, 0.3, 0.5
_Q_MINUS4 = np.array([
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.017, 0.053, 0.097, 0.130, 0.153,
     0.171, 0.189, 0.207, 0.224, 0.240, 0.254, 0.267, 0.279, 0.289, 0.300,
     0.310, 0.321, 0.332, 0.343, 0.356, 0.368, 0.380, 0.392, 0.404, 0.416,
     0.427, 0.439, 0.451, 0.462, 0.475, 0.488, 0.502, 0.516, 0.529, 0.542,
     0.554, 0.568, 0.582, 0.597, 0.611, 0.627, 0.643, 0.661, 0.680, 0.699,
     0.716, 0.732, 0.749, 0.768, 0.788, 0.809, 0.830, 0.852, 0.877, 0.903,
     0.930, 0.959, 0.990, 1.021, 1.054, 1.101, 1.126, 1.148, 1.172, 1.196,
     1.224, 1.252, 1.283, 1.316, 1.356, 1.400, 1.444, 1.488, 1.533, 1.577,
     1.494, 1.410, 1.327, 1.217, 1.054, 0.872, 0.658, 0.347, 0.089, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.012, 0.025, 0.042, 0.063, 0.084, 0.105,
     0.125, 0.147, 0.170, 0.193, 0.215, 0.238, 0.259, 0.281, 0.303, 0.325,
     0.347, 0.369, 0.390, 0.411, 0.432, 0.454, 0.475, 0.496, 0.516, 0.537,
     0.558, 0.578, 0.599, 0.619, 0.638, 0.658, 0.677, 0.696, 0.714, 0.733,
     0.757, 0.784, 0.804, 0.816, 0.825, 0.835, 0.847, 0.859, 0.870, 0.878,
     0.885, 0.894, 0.907, 0.918, 0.926, 0.931, 0.934, 0.936, 0.937, 0.936,
     0.936, 0.935, 0.934, 0.930, 0.924, 0.918, 0.915, 0.906, 0.898, 0.889,
     0.879, 0.865, 0.848, 0.827, 0.796, 0.756, 0.701, 0.639, 0.564, 0.479,
     0.409, 0.334, 0.246, 0.160, 0.078, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.014, 0.036, 0.066, 0.094, 0.123,
     0.152, 0.182, 0.211, 0.238, 0.265, 0.290, 0.314, 0.338, 0.361, 0.383,
     0.404, 0.423, 0.441, 0.457, 0.472, 0.486, 0.499, 0.513, 0.527, 0.540,
     0.555, 0.571, 0.587, 0.603, 0.621, 0.639, 0.659, 0.680, 0.702, 0.725,
     0.751, 0.783, 0.814, 0.835, 0.851, 0.867, 0.884, 0.898, 0.910, 0.921,
     0.931, 0.936, 0.937, 0.938, 0.940, 0.939, 0.932, 0.919, 0.903, 0.882,
     0.857, 0.828, 0.796, 0.759, 0.719, 0.675, 0.652, 0.626, 0.599, 0.570,
     0.538, 0.512, 0.479, 0.447, 0.414, 0.381, 0.345, 0.303, 0.264, 0.225,
     0.183, 0.141, 0.085, 0.040, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
])

# q/D.L. extrapolated by Prouty, -10 deg twist; rows z/R = 0.1, 0.3, 0.5
_Q_MINUS10 = np.array([
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.009, 0.044, 0.079, 0.114, 0.149, 0.183,
     0.213, 0.239, 0.262, 0.283, 0.301, 0.316, 0.331, 0.345, 0.358, 0.370,
     0.381, 0.391, 0.400, 0.409, 0.417, 0.427, 0.437, 0.447, 0.457, 0.467,
     0.479, 0.490, 0.501, 0.511, 0.521, 0.533, 0.544, 0.556, 0.567, 0.577,
     0.589, 0.600, 0.613, 0.626, 0.641, 0.659, 0.679, 0.700, 0.722, 0.743,
     0.764, 0.785, 0.806, 0.827, 0.848, 0.869, 0.891, 0.912, 0.933, 0.954,
     0.975, 0.996, 1.017, 1.039, 1.060, 1.081, 1.095, 1.117, 1.139, 1.155,
     1.177, 1.203, 1.232, 1.253, 1.284, 1.322, 1.361, 1.399, 1.438, 1.398,
     1.307, 1.189, 1.036, 0.867, 0.811, 0.611, 0.561, 0.327, 0.105, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.012, 0.036, 0.060, 0.084, 0.108, 0.133,
     0.160, 0.187, 0.213, 0.239, 0.265, 0.291, 0.318, 0.343, 0.367, 0.391,
     0.416, 0.441, 0.465, 0.488, 0.511, 0.532, 0.552, 0.573, 0.594, 0.613,
     0.632, 0.650, 0.668, 0.686, 0.703, 0.720, 0.736, 0.751, 0.762, 0.770,
     0.772, 0.775, 0.783, 0.795, 0.806, 0.817, 0.827, 0.837, 0.847, 0.857,
     0.867, 0.877, 0.886, 0.894, 0.897, 0.899, 0.900, 0.900, 0.899, 0.898,
     0.897, 0.896, 0.892, 0.886, 0.879, 0.870, 0.863, 0.858, 0.847, 0.839,
     0.826, 0.813, 0.848, 0.824, 0.767, 0.698, 0.644, 0.608, 0.559, 0.479,
     0.406, 0.333, 0.246, 0.160, 0.078, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.030, 0.062, 0.095, 0.128, 0.161,
     0.195, 0.230, 0.263, 0.295, 0.326, 0.355, 0.383, 0.409, 0.434, 0.457,
     0.478, 0.497, 0.516, 0.535, 0.553, 0.567, 0.578, 0.590, 0.601, 0.612,
     0.624, 0.638, 0.652, 0.667, 0.683, 0.699, 0.715, 0.732, 0.749, 0.769,
     0.788, 0.801, 0.810, 0.820, 0.833, 0.848, 0.864, 0.879, 0.894, 0.909,
     0.921, 0.928, 0.930, 0.929, 0.920, 0.906, 0.890, 0.872, 0.849, 0.823,
     0.795, 0.774, 0.756, 0.718, 0.647, 0.633, 0.581, 0.550, 0.524, 0.500,
     0.484, 0.486, 0.412, 0.380, 0.352, 0.328, 0.305, 0.264, 0.210, 0.178,
     0.165, 0.141, 0.085, 0.040, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
])


_CHARTS = {}


def _charts(table):
    key = id(table)
    if key not in _CHARTS:
        _CHARTS[key] = [InterpND(method='akima', points=_R, values=row, extrapolate=False)
                        for row in table]
    return _CHARTS[key]


def wake_q_ratio(r_R, z_R, table=_Q_MINUS4):
    """q/D.L. from Figure 4.6 and its derivatives d/d(r/R), d/d(z/R)."""
    charts = _charts(table)
    r = np.clip(np.asarray(r_R, dtype=float), _R[0], _R[-1])
    z = np.clip(np.asarray(z_R, dtype=float), Z_PANELS[0], Z_PANELS[-1])
    vals, slopes = zip(*(c.interpolate(r, compute_derivative=True) for c in charts))
    vals, slopes = np.array(vals), np.array([s.ravel() for s in slopes])

    upper = z > Z_PANELS[1]
    lo = upper.astype(int)
    t = (z - Z_PANELS[lo]) / (Z_PANELS[lo + 1] - Z_PANELS[lo])
    cols = np.arange(len(r))
    q = (1 - t) * vals[lo, cols] + t * vals[lo + 1, cols]
    dq_dr = (1 - t) * slopes[lo, cols] + t * slopes[lo + 1, cols]
    inside = (np.asarray(z_R) > Z_PANELS[0]) & (np.asarray(z_R) < Z_PANELS[-1])
    dq_dz = np.where(inside, (vals[lo + 1, cols] - vals[lo, cols]) / 0.2, 0.0)
    return q, dq_dr, dq_dz


class WakeDynamicPressureComp(om.ExplicitComponent):
    """Figure 4.6 dynamic pressure ratio at each airframe segment."""

    def initialize(self):
        self.options.declare('num_segments', types=int, default=1)

    def setup(self):
        n = self.options['num_segments']
        ar = np.arange(n)

        self.add_input('seg_r_R', val=np.full(n, 0.5), desc='segment radial position, r/R')
        self.add_input('seg_z_R', val=np.full(n, 0.2), desc='segment distance below the rotor, z/R')
        self.add_input('vi_ratio', val=np.ones(n),
                       desc='induced velocity over that of the -4 deg twist rotor')
        self.add_output('q_DL', val=np.zeros(n), desc='wake dynamic pressure over disc loading')

        self.declare_partials('q_DL', ['seg_r_R', 'seg_z_R', 'vi_ratio'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        z = np.real(inputs['seg_z_R'])
        if np.any(z < Z_PANELS[0] - 1e-9) or np.any(z > Z_PANELS[-1] + 1e-9):
            warnings.warn('z/R outside 0.1-0.5, the range of Figure 4.6; '
                          'the nearest panel is used.', stacklevel=2)
        q = wake_q_ratio(np.real(inputs['seg_r_R']), z)[0]
        outputs['q_DL'] = q * inputs['vi_ratio'] ** 2

    def compute_partials(self, inputs, partials):
        k = inputs['vi_ratio']
        q, dq_dr, dq_dz = wake_q_ratio(inputs['seg_r_R'], inputs['seg_z_R'])
        partials['q_DL', 'seg_r_R'] = dq_dr * k ** 2
        partials['q_DL', 'seg_z_R'] = dq_dz * k ** 2
        partials['q_DL', 'vi_ratio'] = 2.0 * q * k
