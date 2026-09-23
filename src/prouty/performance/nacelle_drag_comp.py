"""
NacelleDragComp -- equivalent flat plate area of externally mounted engine nacelles.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Parasite Drag in Forward Flight", p. 294 and Figure 4.19 p. 296
(Keys & Wiesner, JAHS 20-1, 1975); procedure and example p. 306.

A nacelle of diameter D_N mounted a distance y from the fuselage has a drag
coefficient, on its frontal area, falling with the distance ratio y/D_N as the
interference with the fuselage fades (Figure 4.19):

    Drag_NAC / q = C_DN pi (D_N/2)^2          per nacelle
    f_N = n_nacelles C_DN pi (D_N/2)^2

Example helicopter, p. 306: two nacelles, D_N = 2.8 ft, y = 1.4 ft, y/D_N = 0.5,
C_DN = 0.09, A_N = 12 ft^2, f_N = 1.1 ft^2.

Digitization (C4-17): clean copy, piecewise tick calibration, pixel columns
of the curve, smoothing spline (rms 0.001), tabulated every 0.1 up to
y/D_N = 0.9 where the drawn curve ends. It reads 0.088 at y/D_N = 0.5, where
the book reads 0.09. Akima, held outside 0-0.9 with a warning.

    D_N, y_N --> y_D_N, C_DN, A_N, f_N
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_Y_D = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_CD_N = np.array([0.180, 0.149, 0.125, 0.108, 0.096, 0.088, 0.083, 0.080, 0.078, 0.077])
_CHART = InterpND(method='akima', points=_Y_D, values=_CD_N, extrapolate=False)


def nacelle_drag_coefficient(y_D):
    """Figure 4.19 C_DN and its slope; held outside the drawn range."""
    x = np.clip(np.real(y_D), _Y_D[0], _Y_D[-1])
    v, dv = _CHART.interpolate(np.atleast_1d(x), compute_derivative=True)
    inside = _Y_D[0] < np.real(y_D) < _Y_D[-1]
    return v[0], (dv.ravel()[0] if inside else 0.0)


class NacelleDragComp(om.ExplicitComponent):
    """Figure 4.19 engine nacelle drag, p. 306."""

    def initialize(self):
        self.options.declare('num_nacelles', types=int, default=1)

    def setup(self):
        self.add_input('D_N', val=1.0, units='ft', desc='nacelle diameter')
        self.add_input('y_N', val=0.5, units='ft', desc='nacelle distance from the fuselage')
        self.add_output('y_D_N', val=0.5, desc='distance ratio y/D_N')
        self.add_output('C_DN', val=0.085, desc='nacelle drag coefficient on frontal area')
        self.add_output('A_N', val=1.0, units='ft**2', desc='frontal area of all nacelles')
        self.add_output('f_N', val=0.0, units='ft**2', desc='nacelle equivalent flat plate area')
        self.declare_partials('*', ['D_N', 'y_N'])

    def compute(self, inputs, outputs):
        D, y = inputs['D_N'][0], inputs['y_N'][0]
        ratio = y / D
        if np.real(ratio) > _Y_D[-1] or np.real(ratio) < 0.0:
            warnings.warn(f'y/D_N = {np.real(ratio):.2f} outside Figure 4.19 (0-0.9); edge held.',
                          stacklevel=2)
        C_DN = nacelle_drag_coefficient(ratio)[0]
        A_N = self.options['num_nacelles'] * np.pi * D ** 2 / 4.0
        outputs['y_D_N'] = ratio
        outputs['C_DN'] = C_DN
        outputs['A_N'] = A_N
        outputs['f_N'] = C_DN * A_N

    def compute_partials(self, inputs, partials):
        D, y = inputs['D_N'][0], inputs['y_N'][0]
        ratio = y / D
        C_DN, slope = nacelle_drag_coefficient(ratio)
        n = self.options['num_nacelles']
        A_N, dA_dD = n * np.pi * D ** 2 / 4.0, n * np.pi * D / 2.0
        dr_dD, dr_dy = -y / D ** 2, 1.0 / D

        partials['y_D_N', 'D_N'], partials['y_D_N', 'y_N'] = dr_dD, dr_dy
        partials['C_DN', 'D_N'], partials['C_DN', 'y_N'] = slope * dr_dD, slope * dr_dy
        partials['A_N', 'D_N'], partials['A_N', 'y_N'] = dA_dD, 0.0
        partials['f_N', 'D_N'] = slope * dr_dD * A_N + C_DN * dA_dD
        partials['f_N', 'y_N'] = slope * dr_dy * A_N
