"""
RotorHubDragComp -- equivalent flat plate area of a rotor hub.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Bluff Body Drag": Table 4.2 p. 298, Figure 4.22 p. 299
(Linville, USAAMRDL TR 71-46); procedure and example pp. 306-307.

Table 4.2 gives hub drag coefficients on hub frontal area at zero shaft angle
of attack and zero rpm. Figure 4.22 corrects them, for an unfaired or a faired
hub, to the shaft angle of attack alpha_s and the rotor speed:

    DR    = [f/f(alpha=0)](alpha_s) * [f/f(100%)](rpm) / [f/f(100%)](0)
    C_D   = C_D0 DR
    f_hub = A_hub C_D

Example helicopter, pp. 306-307: main rotor hub A = 5 ft^2, C_D0 = 1.1,
alpha_s = 0, 100 % rpm, DR = 1.00/0.95 = 1.05, C_D = 1.16, f_MH = 5.8 ft^2;
tail rotor hub A = 0.6 ft^2, f_T = 0.7 ft^2. The same component serves both.

Digitization (C4-18): clean copies. Angle of attack: symbol centres of both
hubs every 2 deg, divided by their common value at alpha = 0 (1.017 on the
copy). rpm: pixel columns of both fairings, smoothing spline, divided by the
value at 100 %. Akima; held outside -10..10 deg and 0..100 % with a warning.

    A_hub, C_D0, alpha_s, rpm_pct --> DR, C_D, f_hub
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_ALPHA = np.arange(-10.0, 10.1, 2.0)
_RPM = np.arange(0.0, 100.1, 10.0)

# Figure 4.22 p. 299: f/f(alpha=0) at 100 % rpm, and f/f(100 % rpm) at alpha = 0
_ALPHA_RATIO = {
    'unfaired': [1.102, 1.057, 1.036, 0.975, 0.953, 1.000, 0.969, 1.016, 0.993, 1.014, 1.077],
    'faired': [1.558, 1.363, 1.254, 1.154, 1.028, 1.000, 1.007, 1.061, 1.075, 1.271, 1.309],
}
_RPM_RATIO = {
    'unfaired': [0.949, 0.961, 0.971, 0.980, 0.986, 0.991, 0.995, 0.997, 0.999, 1.000, 1.000],
    'faired': [0.839, 0.856, 0.872, 0.889, 0.905, 0.921, 0.937, 0.952, 0.967, 0.983, 1.000],
}
_CHARTS = {h: (InterpND(method='akima', points=_ALPHA, values=np.array(_ALPHA_RATIO[h]),
                        extrapolate=False),
               InterpND(method='akima', points=_RPM, values=np.array(_RPM_RATIO[h]),
                        extrapolate=False)) for h in _ALPHA_RATIO}

# Table 4.2 p. 298: blades, hub frontal area ft^2, hub/disc area, C_D0 unfaired, faired
# (zero angle of attack and zero rpm); None where not reported
TABLE_4_2 = {
    'CH-47': (3, 5.0, 0.0027, 1.38, 0.88),
    'OH-6A': (4, 1.5, 0.0028, 1.13, 0.80),
    'UH-1B': (2, 5.6, 0.0037, 0.98, 0.45),
    'LOH wind tunnel model, 3 blades': (3, 1.65, None, 0.61, 0.53),
    'LOH wind tunnel model, 2 blades': (2, 1.15, None, 0.47, None),
    'S-58': (4, 7.5, 0.0031, 1.53, 0.57),
    'S-65': (4, 16.6, 0.0041, 1.01, None),
    'S-65, rigid head fairing': (4, 21.7, None, None, 0.59),
    'S-65, floating head fairing': (4, 33.9, None, None, 0.22),
    'wind tunnel model, unfaired': (3, 0.062, None, 1.26, None),
    'wind tunnel model, faired': (3, 0.102, None, None, 0.76),
    'AS Twinstar': (3, 2.5, 0.0026, 1.55, None),
    'AS Puma': (4, 5.8, 0.0030, 0.98, None),
    'AS Dauphin': (4, 4.3, 0.0037, 1.56, None),
}


def _chart(chart, x, lo, hi):
    xc = np.clip(np.real(x), lo, hi)
    v, dv = chart.interpolate(np.atleast_1d(xc), compute_derivative=True)
    return v[0], (dv.ravel()[0] if lo < np.real(x) < hi else 0.0)


def hub_drag_ratio(alpha_s, rpm_pct, fairing):
    """Figure 4.22 drag ratio and its derivatives d/d(alpha_s), d/d(rpm)."""
    c_alpha, c_rpm = _CHARTS[fairing]
    a, da = _chart(c_alpha, alpha_s, _ALPHA[0], _ALPHA[-1])
    r, dr = _chart(c_rpm, rpm_pct, _RPM[0], _RPM[-1])
    r0 = _RPM_RATIO[fairing][0]
    return a * r / r0, da * r / r0, a * dr / r0


class RotorHubDragComp(om.ExplicitComponent):
    """Hub drag corrected for shaft angle of attack and rpm, p. 306."""

    def initialize(self):
        self.options.declare('fairing', default='unfaired', values=tuple(_ALPHA_RATIO))

    def setup(self):
        self.add_input('A_hub', val=1.0, units='ft**2', desc='hub frontal area')
        self.add_input('C_D0', val=1.0, desc='hub drag coefficient, Table 4.2 (alpha 0, rpm 0)')
        self.add_input('alpha_s', val=0.0, units='deg', desc='shaft angle of attack')
        self.add_input('rpm_pct', val=100.0, desc='rotor speed, percent of normal')
        self.add_output('DR', val=1.0, desc='drag ratio of Figure 4.22')
        self.add_output('C_D', val=1.0, desc='corrected hub drag coefficient')
        self.add_output('f_hub', val=1.0, units='ft**2', desc='hub equivalent flat plate area')
        self.declare_partials('DR', ['alpha_s', 'rpm_pct'])
        self.declare_partials('C_D', ['C_D0', 'alpha_s', 'rpm_pct'])
        self.declare_partials('f_hub', ['A_hub', 'C_D0', 'alpha_s', 'rpm_pct'])

    def compute(self, inputs, outputs):
        alpha, rpm = np.real(inputs['alpha_s'][0]), np.real(inputs['rpm_pct'][0])
        if not (_ALPHA[0] <= alpha <= _ALPHA[-1] and _RPM[0] <= rpm <= _RPM[-1]):
            warnings.warn(f'alpha_s = {alpha:.1f} deg, rpm = {rpm:.0f} % outside Figure 4.22 '
                          '(-10..10 deg, 0..100 %); edge held.', stacklevel=2)
        DR = hub_drag_ratio(alpha, rpm, self.options['fairing'])[0]
        outputs['DR'] = DR
        outputs['C_D'] = inputs['C_D0'] * DR
        outputs['f_hub'] = inputs['A_hub'] * inputs['C_D0'] * DR

    def compute_partials(self, inputs, partials):
        A, CD0 = inputs['A_hub'][0], inputs['C_D0'][0]
        DR, dDR_da, dDR_dr = hub_drag_ratio(inputs['alpha_s'][0], inputs['rpm_pct'][0],
                                            self.options['fairing'])
        partials['DR', 'alpha_s'], partials['DR', 'rpm_pct'] = dDR_da, dDR_dr
        partials['C_D', 'C_D0'] = DR
        partials['C_D', 'alpha_s'], partials['C_D', 'rpm_pct'] = CD0 * dDR_da, CD0 * dDR_dr
        partials['f_hub', 'A_hub'] = CD0 * DR
        partials['f_hub', 'C_D0'] = A * DR
        partials['f_hub', 'alpha_s'] = A * CD0 * dDR_da
        partials['f_hub', 'rpm_pct'] = A * CD0 * dDR_dr
