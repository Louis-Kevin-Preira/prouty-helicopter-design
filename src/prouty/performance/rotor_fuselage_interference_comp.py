"""
RotorFuselageInterferenceComp -- drag of the rotor wake on the fuselage.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Bluff Body Drag" p. 300, Figure 4.25 p. 302 (wind tunnel
configuration); procedure and example p. 308.

The rotor wake changes the flow over the fuselage and raises its drag. The
measured increment, on fuselage frontal area, grows with the fuselage angle of
attack:

    f_int = dC_D(alpha_F) A_F

The procedure reads it at alpha_F = 0 (p. 308), which is the default here.

Example helicopter: dC_D = 0.018, A_F = 74 ft^2, f_int = 1.3 ft^2.

Digitization (C4-23): pixel columns of the fairing, smoothing spline
(rms 0.0002), on a clean copy; a first pass on the book scan agreed within
0.0001. It reads 0.0180 at alpha_F = 0, the book's value, and runs from
0.0058 at -10 deg to 0.0229 at 8 deg. Akima; held outside
-10..8 deg with a warning.

    A_F, alpha_F --> dC_D, f_int
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_ALPHA_F = np.arange(-10.0, 8.01, 2.0)
_DCD = np.array([0.0058, 0.0091, 0.0119, 0.0145, 0.0164, 0.0180, 0.0195, 0.0208, 0.0216, 0.0229])
_CHART = InterpND(method='akima', points=_ALPHA_F, values=_DCD, extrapolate=False)


def rotor_fuselage_interference(alpha_F):
    """Figure 4.25 dC_D and its slope; held outside the drawn range."""
    x = np.clip(np.real(alpha_F), _ALPHA_F[0], _ALPHA_F[-1])
    v, dv = _CHART.interpolate(np.atleast_1d(x), compute_derivative=True)
    inside = _ALPHA_F[0] < np.real(alpha_F) < _ALPHA_F[-1]
    return v[0], (dv.ravel()[0] if inside else 0.0)


class RotorFuselageInterferenceComp(om.ExplicitComponent):
    """Figure 4.25 rotor-fuselage interference drag, p. 308."""

    def setup(self):
        self.add_input('A_F', val=1.0, units='ft**2', desc='fuselage frontal area')
        self.add_input('alpha_F', val=0.0, units='deg', desc='fuselage angle of attack')
        self.add_output('dC_D', val=0.018, desc='drag increment on frontal area')
        self.add_output('f_int', val=0.0, units='ft**2', desc='interference flat plate area')
        self.declare_partials('dC_D', 'alpha_F')
        self.declare_partials('f_int', ['A_F', 'alpha_F'])

    def compute(self, inputs, outputs):
        alpha = np.real(inputs['alpha_F'][0])
        if not _ALPHA_F[0] <= alpha <= _ALPHA_F[-1]:
            warnings.warn(f'alpha_F = {alpha:.1f} deg outside Figure 4.25 (-10..8); edge held.',
                          stacklevel=2)
        dC_D = rotor_fuselage_interference(alpha)[0]
        outputs['dC_D'] = dC_D
        outputs['f_int'] = inputs['A_F'] * dC_D

    def compute_partials(self, inputs, partials):
        dC_D, slope = rotor_fuselage_interference(inputs['alpha_F'][0])
        partials['dC_D', 'alpha_F'] = slope
        partials['f_int', 'A_F'] = dC_D
        partials['f_int', 'alpha_F'] = inputs['A_F'][0] * slope
