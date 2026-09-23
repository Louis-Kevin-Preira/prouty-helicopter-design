"""
StabilizerDragComp -- equivalent flat plate area of a wing or stabilizer.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4: skin friction Figure 4.12 p. 289, surfaces p. 294-295, junction
drag Figure 4.21 p. 297, procedure and example p. 307.

    C_D   = C_d0 + C_L^2 (1 + delta) / (pi AR) + n_j C_dJ t^2 / A
    f     = (q_local/q) C_D A

C_d0 comes from the skin friction coefficient of Figure 4.12 at the Reynolds
number of the mean aerodynamic chord, with the usual thickness form factor
(C4-22):

    R.N.  = rho V MAC / mu
    C_d0  = 2 C_F(R.N.) [1 + 2 (t/c) + 60 (t/c)^4]

transition = 'turbulent' uses the forced turbulent curve, which reproduces the
book's readings and suits a real surface with imperfections (p. 291);
'natural' uses the natural transition curve.

The junction coefficient C_dJ of Figure 4.21 is applied at n_junctions roots
(2 in the book's example, as in Delta D/q = t^2 C_dJ per junction). The
dynamic pressure ratio q_local/q is 1 for a wing, 0.5 to 0.8 for stabilizers
in the wake of the airframe (p. 295); the example uses 0.75.

Example helicopter, p. 307. Horizontal stabilizer A = 18 ft^2, b = 9 ft,
t/c = 0.12, MAC = 2 ft, C_L = -0.3, delta = 0.2, q ratio 0.75: R.N. = 2e6,
C_d0 = 0.010, C_Di = 0.008, junction 0.001, f_H = 0.2 ft^2. Vertical
stabilizer A = 24 ft^2, MAC = 3 ft, no lift, no junction: f_V = 0.2 ft^2.

    A, b, t_c, MAC, C_L, delta, q_ratio, rho, T_air, V --> RN, C_F, C_D, f_stab
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

from prouty.performance.rotor_shaft_drag_comp import MU_0, S, T_0

# Figure 4.12 p. 289, skin friction coefficient of one side of a flat plate
_LOG_RN = np.arange(4.0, 9.01, 0.5)
_C_F = {
    'natural': np.array([0.01306, 0.00698, 0.00449, 0.00232, 0.00257, 0.00309, 0.00297,
                         0.00255, 0.00205, 0.00168, 0.00164]),
    'turbulent': np.array([0.01306, 0.00864, 0.00761, 0.00616, 0.00459, 0.00364, 0.00297,
                           0.00255, 0.00205, 0.00168, 0.00164]),
}
_CF_CHARTS = {k: InterpND(method='akima', points=_LOG_RN, values=v, extrapolate=False)
              for k, v in _C_F.items()}

# Figure 4.21 p. 297, junction drag coefficient against airfoil thickness ratio
_T_C = np.array([0.000, 0.020, 0.040, 0.060, 0.065, 0.080, 0.100, 0.120, 0.150, 0.200, 0.250, 0.300, 0.350,
                 0.400, 0.450, 0.500, 0.550, 0.600])
_CD_J = np.array([0.000, 0.000, 0.000, 0.000, 0.003, 0.026, 0.055, 0.076, 0.104, 0.145, 0.177, 0.207, 0.239,
                  0.273, 0.306, 0.339, 0.373, 0.409])
_CDJ_CHART = InterpND(method='akima', points=_T_C, values=_CD_J, extrapolate=False)


def skin_friction(RN, transition='turbulent'):
    """Figure 4.12 C_F and dC_F/dR.N.; held outside 1e4-1e9."""
    log_rn = np.log10(np.real(RN))
    x = np.clip(log_rn, _LOG_RN[0], _LOG_RN[-1])
    v, dv = _CF_CHARTS[transition].interpolate(np.atleast_1d(x), compute_derivative=True)
    inside = _LOG_RN[0] < log_rn < _LOG_RN[-1]
    return v[0], (dv.ravel()[0] / (np.real(RN) * np.log(10.0)) if inside else 0.0)


def junction_drag_coefficient(t_c):
    """Figure 4.21 C_dJ and its slope; zero below t/c = 0.065, held above 0.6."""
    x = np.clip(np.real(t_c), _T_C[0], _T_C[-1])
    v, dv = _CDJ_CHART.interpolate(np.atleast_1d(x), compute_derivative=True)
    inside = _T_C[0] < np.real(t_c) < _T_C[-1]
    value = v[0]
    if np.real(value) <= 0.0:                 # the curve starts at t/c = 0.065
        return 0.0, 0.0
    return value, (dv.ravel()[0] if inside else 0.0)


class StabilizerDragComp(om.ExplicitComponent):
    """Profile, induced and junction drag of a lifting surface, p. 307."""

    def initialize(self):
        self.options.declare('transition', default='turbulent', values=tuple(_C_F))
        self.options.declare('num_junctions', types=int, default=2)

    def setup(self):
        self.add_input('A', val=1.0, units='ft**2', desc='surface area')
        self.add_input('b', val=1.0, units='ft', desc='span')
        self.add_input('MAC', val=1.0, units='ft', desc='mean aerodynamic chord')
        self.add_input('t_c', val=0.12, desc='thickness ratio')
        self.add_input('C_L', val=0.0, desc='lift coefficient, from trim')
        self.add_input('delta', val=0.2, desc='span efficiency factor')
        self.add_input('q_ratio', val=1.0, desc='local dynamic pressure over free stream')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('T_air', val=T_0, units='degR')
        self.add_input('V', val=1.0, units='ft/s', desc='reference flight speed')

        self.add_output('RN', val=1e6, desc='Reynolds number on the MAC')
        self.add_output('C_F', val=0.003, desc='skin friction coefficient, Figure 4.12')
        self.add_output('C_D', val=0.01, desc='total drag coefficient on surface area')
        self.add_output('f_stab', val=0.0, units='ft**2', desc='equivalent flat plate area')

        self.declare_partials('*', ['rho', 'T_air', 'V', 'MAC', 't_c', 'A', 'b', 'C_L',
                                    'delta', 'q_ratio'], method='fd', step=1e-7,
                              form='central')

    @staticmethod
    def _mu(T):
        return MU_0 * (T / T_0) ** 1.5 * (T_0 + S) / (T + S)

    def compute(self, inputs, outputs):
        A, b, MAC, t_c = (inputs[k][0] for k in ('A', 'b', 'MAC', 't_c'))
        RN = inputs['rho'][0] * inputs['V'][0] * MAC / self._mu(inputs['T_air'][0])
        if not 1e4 <= np.real(RN) <= 1e9:
            warnings.warn(f'R.N. = {np.real(RN):.3g} outside Figure 4.12 (1e4-1e9); edge held.',
                          stacklevel=2)
        C_F = skin_friction(RN, self.options['transition'])[0]
        C_d0 = 2.0 * C_F * (1.0 + 2.0 * t_c + 60.0 * t_c ** 4)
        C_Di = inputs['C_L'][0] ** 2 * (1.0 + inputs['delta'][0]) / (np.pi * b ** 2 / A)
        t = t_c * MAC
        C_Dj = self.options['num_junctions'] * junction_drag_coefficient(t_c)[0] * t ** 2 / A

        outputs['RN'] = RN
        outputs['C_F'] = C_F
        outputs['C_D'] = C_d0 + C_Di + C_Dj
        outputs['f_stab'] = inputs['q_ratio'][0] * (C_d0 + C_Di + C_Dj) * A
