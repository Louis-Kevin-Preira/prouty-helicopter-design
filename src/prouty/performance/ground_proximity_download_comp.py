"""
GroundProximityDownloadComp -- ground proximity factors on download and pseudo ground effect.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Drag in Hover", p. 283 and Figure 4.8 p. 285; hover in
ground effect of the example helicopter p. 309.

Model tests (Fradenburgh, AGARD CP 111) show that both the vertical drag and
the pseudo ground effect fall, and even change sign, below about one rotor
diameter above the ground (p. 283; S-76 model: +3 % download out of ground
effect, -1 % at 1 ft wheel height). The book gives no method, only Figure 4.8
for one model, fuselage alone and fuselage with wing.

The curves are used as shapes (C4-13): each is divided by its value at
z/D = 2.6, taken as out of ground effect, and the factor multiplies the out of
ground effect results of VerticalDragComp and PseudoGroundEffectComp:

    D_v/GW (z/D)   = k_Dv (z/D)  D_v/GW OGE
    dCQ_sigma (z/D) = k_PGE (z/D) dCQ_sigma OGE

configuration
    'fuselage'        Figure 4.8, fuselage alone
    'fuselage_wing'   Figure 4.8, fuselage plus wing
    'none'            k = 1, ground proximity ignored
    'removed'         k = 0, as the book does in ground effect, p. 309

Digitization of Figure 4.8 (pixel columns, smoothing spline, rms within one
pixel), sampled from z/D = 0.2 to 2.6; the first points of the fuselage alone
curves are extrapolated from z/D = 0.28. Akima interpolation, held outside
the table with a warning below z/D = 0.2.

    z_D (nn,) --> k_Dv, k_PGE (nn,)
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_Z_D = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])

# Figure 4.8 over its value at z/D = 2.6 (fuselage: Dv/T 0.0249, dCQ/sigma -0.00014;
# fuselage plus wing: Dv/T 0.168, dCQ/sigma -0.00044)
_FACTORS = {
    'fuselage': (
        [-0.586, -0.482, -0.383, -0.289, -0.199, -0.113, -0.032, 0.118, 0.253, 0.372,
         0.477, 0.569, 0.718, 0.825, 0.897, 0.943, 0.968, 0.982, 0.990, 1.000],
        [-1.423, -1.218, -1.013, -0.811, -0.616, -0.433, -0.264, 0.024, 0.255, 0.439,
         0.587, 0.707, 0.879, 0.975, 1.014, 1.014, 0.994, 0.973, 0.969, 1.000]),
    'fuselage_wing': (
        [-0.009, 0.037, 0.099, 0.168, 0.237, 0.298, 0.343, 0.446, 0.530, 0.605,
         0.670, 0.724, 0.806, 0.866, 0.912, 0.948, 0.972, 0.988, 0.997, 1.000],
        [0.042, -0.075, -0.180, -0.271, -0.342, -0.390, -0.412, -0.370, -0.235, -0.044,
         0.162, 0.342, 0.535, 0.637, 0.720, 0.791, 0.849, 0.900, 0.948, 1.000]),
}
CONFIGURATIONS = ('fuselage', 'fuselage_wing', 'none', 'removed')


class GroundProximityDownloadComp(om.ExplicitComponent):
    """Figure 4.8 factors on the out of ground effect download and pseudo ground effect."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('configuration', default='fuselage', values=CONFIGURATIONS)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        config = self.options['configuration']

        self.add_input('z_D', val=np.full(nn, 3.0), desc='rotor height above ground over diameter')
        self.add_output('k_Dv', val=np.ones(nn), desc='download factor on the OGE value')
        self.add_output('k_PGE', val=np.ones(nn), desc='pseudo ground effect factor on the OGE value')

        self._charts = None
        if config in _FACTORS:
            self._charts = [InterpND(method='akima', points=_Z_D, values=np.array(v),
                                     extrapolate=False) for v in _FACTORS[config]]
            self.declare_partials(['k_Dv', 'k_PGE'], 'z_D', rows=ar, cols=ar)

    def _factors(self, z_D):
        z = np.clip(np.real(z_D), _Z_D[0], _Z_D[-1])
        inside = (np.real(z_D) > _Z_D[0]) & (np.real(z_D) < _Z_D[-1])
        out = []
        for chart in self._charts:
            v, dv = chart.interpolate(z, compute_derivative=True)
            out.append((v, np.where(inside, dv.ravel(), 0.0)))
        return out

    def compute(self, inputs, outputs):
        config = self.options['configuration']
        if self._charts is None:
            value = 1.0 if config == 'none' else 0.0
            outputs['k_Dv'] = value
            outputs['k_PGE'] = value
            return
        if np.any(np.real(inputs['z_D']) < _Z_D[0]):
            warnings.warn('z/D below 0.2, the lowest point of Figure 4.8; it is held.',
                          stacklevel=2)
        (k_dv, _), (k_pge, _) = self._factors(inputs['z_D'])
        outputs['k_Dv'] = k_dv
        outputs['k_PGE'] = k_pge

    def compute_partials(self, inputs, partials):
        if self._charts is None:
            return
        (_, dk_dv), (_, dk_pge) = self._factors(inputs['z_D'])
        partials['k_Dv', 'z_D'] = dk_dv
        partials['k_PGE', 'z_D'] = dk_pge
