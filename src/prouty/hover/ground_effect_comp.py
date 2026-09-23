"""
GroundEffectComp -- induced velocity ratio in ground effect at constant thrust.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, "Ground Effect", pp. 63-68, Figure 1.41 p. 66.

Model rotor tests (NACA TN 835, USAAMRDL TR 72-4) reduced to the ratio of
induced velocity in and out of ground effect at constant thrust, against the
rotor height above ground over the rotor diameter. The solid curve is used,
as in the examples of the book: 0.75 at z/D = 0.3 (p. 67), 0.62 at
z/D = 0.12 (p. 281). The dashed full-scale curve (Hayden) is not.

Digitization: 396 pixels of the solid line, quartic fit, rms 0.002; tabulated
at every 0.05 up to z/D = 1.1. The curve ends at z/D = 1.17 near 0.99; beyond
it the table closes on 1.0 at z/D = 1.5 (out of ground effect, an assumption).
Akima interpolation; below z/D = 0.05 the first point is held and a warning
is raised.

Uses: in ground effect power (p. 67) and pseudo ground effect of the fuselage
(Chapter 4, p. 280), with z/D the height of the ground plane under the rotor.

    z_D (nn,) --> vi_IGE_OGE (nn,)
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_Z_D = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35, 1.50])
_VI = np.array([0.552, 0.601, 0.646, 0.685, 0.720, 0.752, 0.780, 0.805, 0.827, 0.848, 0.866, 0.883,
                0.898, 0.912, 0.925, 0.937, 0.947, 0.957, 0.965, 0.972, 0.978, 0.982, 0.990, 0.997,
                1.000])

_CHART = InterpND(method='akima', points=_Z_D, values=_VI, extrapolate=False)


def ground_effect_ratio(z_D):
    """Figure 1.41 solid curve and its slope; held outside the table."""
    z = np.clip(np.real(np.asarray(z_D, dtype=float)), _Z_D[0], _Z_D[-1])
    v, dv = _CHART.interpolate(z, compute_derivative=True)
    inside = (np.real(z_D) > _Z_D[0]) & (np.real(z_D) < _Z_D[-1])
    return v, np.where(inside, dv.ravel(), 0.0)


class GroundEffectComp(om.ExplicitComponent):
    """Figure 1.41: v_IGE / v_OGE at constant thrust."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('z_D', val=np.full(nn, 2.0), desc='height above ground over rotor diameter')
        self.add_output('vi_IGE_OGE', val=np.ones(nn), desc='induced velocity ratio, constant thrust')
        self.declare_partials('vi_IGE_OGE', 'z_D', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        if np.any(np.real(inputs['z_D']) < _Z_D[0]):
            warnings.warn('z/D below 0.05, the lowest point of Figure 1.41; it is held.',
                          stacklevel=2)
        outputs['vi_IGE_OGE'] = ground_effect_ratio(inputs['z_D'])[0]

    def compute_partials(self, inputs, partials):
        partials['vi_IGE_OGE', 'z_D'] = ground_effect_ratio(inputs['z_D'])[1]
