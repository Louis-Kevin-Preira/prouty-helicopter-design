"""
RotorShaftDragComp -- equivalent flat plate area of an exposed rotor shaft.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Bluff Body Drag": shaft p. 298, Figure 4.23 p. 300 (Hoerner,
circular cylinders); Reynolds number p. 288; procedure and example p. 306.

The shaft is a circular cylinder normal to the flow. Its drag coefficient on
frontal area D_s l_s depends on the Reynolds number on the diameter:

    R.N. = rho V D_s / mu          (p. 288: 6,400 V D_s at sea level, ft/s and ft)
    f_s  = D_s l_s C_D(R.N.)

mu follows Sutherland's law from the air temperature of AtmosphereComp
(3.737e-7 slug/ft/s at 518.67 R, rho/mu = 6,360 s/ft^2 at sea level).

Example helicopter, p. 306: D_s = 0.5 ft at 115 kt, R.N. = 0.6e6, C_D = 0.3,
f_MS = 1(0.3) ft^2 (1 ft^2 frontal area).

Digitization (C4-19): clean copy, log axis calibrated per decade; flat parts by
pixel columns, the drag crisis by pixel rows. The curve reads 0.27 at
R.N. = 0.6e6, near the bottom of the post-critical dip, where the book reads
0.3. Akima in log10 R.N.; held outside 1e4-1e8 with a warning.

    rho, T_air, V, D_s, l_s --> RN, C_D, f_s
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

MU_0, T_0, S = 3.737e-7, 518.67, 198.72         # slug/ft/s, degR, degR (Sutherland)

_LOG_RN = np.array([4.0, 4.5, 5.0, 5.1, 5.2, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65,
                    5.7, 5.75, 5.8, 5.9, 6.0, 6.2, 6.5, 7.0, 7.5, 8.0])
_CD = np.array([1.200, 1.200, 1.199, 1.189, 1.144, 1.038, 0.947, 0.805, 0.613, 0.474, 0.391,
                0.336, 0.305, 0.285, 0.275, 0.270, 0.279, 0.291, 0.306, 0.316, 0.314, 0.309,
                0.298])
_CHART = InterpND(method='akima', points=_LOG_RN, values=_CD, extrapolate=False)


def cylinder_drag_coefficient(RN):
    """Figure 4.23 C_D and dC_D/dR.N.; held outside 1e4-1e8."""
    log_rn = np.log10(np.real(RN))
    x = np.clip(log_rn, _LOG_RN[0], _LOG_RN[-1])
    v, dv = _CHART.interpolate(np.atleast_1d(x), compute_derivative=True)
    inside = _LOG_RN[0] < log_rn < _LOG_RN[-1]
    return v[0], (dv.ravel()[0] / (np.real(RN) * np.log(10.0)) if inside else 0.0)


class RotorShaftDragComp(om.ExplicitComponent):
    """Circular cylinder drag of the rotor shaft, p. 306."""

    def setup(self):
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('T_air', val=T_0, units='degR')
        self.add_input('V', val=1.0, units='ft/s', desc='reference flight speed')
        self.add_input('D_s', val=0.5, units='ft', desc='shaft diameter')
        self.add_input('l_s', val=1.0, units='ft', desc='exposed shaft length')
        self.add_output('RN', val=1e6, desc='Reynolds number on the shaft diameter')
        self.add_output('C_D', val=0.3, desc='cylinder drag coefficient, Figure 4.23')
        self.add_output('f_s', val=0.3, units='ft**2', desc='shaft equivalent flat plate area')
        self.declare_partials('RN', ['rho', 'T_air', 'V', 'D_s'])
        self.declare_partials(['C_D', 'f_s'], ['rho', 'T_air', 'V', 'D_s'])
        self.declare_partials('f_s', 'l_s')

    @staticmethod
    def _mu(T):
        mu = MU_0 * (T / T_0) ** 1.5 * (T_0 + S) / (T + S)
        return mu, mu * (1.5 / T - 1.0 / (T + S))

    def compute(self, inputs, outputs):
        mu = self._mu(inputs['T_air'][0])[0]
        RN = inputs['rho'][0] * inputs['V'][0] * inputs['D_s'][0] / mu
        if not 1e4 <= np.real(RN) <= 1e8:
            warnings.warn(f'R.N. = {np.real(RN):.3g} outside Figure 4.23 (1e4-1e8); edge held.',
                          stacklevel=2)
        C_D = cylinder_drag_coefficient(RN)[0]
        outputs['RN'] = RN
        outputs['C_D'] = C_D
        outputs['f_s'] = inputs['D_s'] * inputs['l_s'] * C_D

    def compute_partials(self, inputs, partials):
        rho, T, V, D, l = (inputs[k][0] for k in ('rho', 'T_air', 'V', 'D_s', 'l_s'))
        mu, dmu_dT = self._mu(T)
        RN = rho * V * D / mu
        C_D, dC_dRN = cylinder_drag_coefficient(RN)
        dRN = {'rho': V * D / mu, 'V': rho * D / mu, 'D_s': rho * V / mu, 'T_air': -RN / mu * dmu_dT}
        for name, d in dRN.items():
            partials['RN', name] = d
            partials['C_D', name] = dC_dRN * d
            partials['f_s', name] = D * l * dC_dRN * d
        partials['f_s', 'D_s'] += l * C_D
        partials['f_s', 'l_s'] = D * C_D
