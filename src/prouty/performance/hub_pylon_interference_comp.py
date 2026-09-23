"""
HubPylonInterferenceComp -- interference drag between hub, shaft and pylon.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Bluff Body Drag" pp. 298-301, Figure 4.24 p. 301 (Keys & Wiesner,
JAHS 20-1, 1975); procedure and example p. 306.

The wake of the hub separates the flow on the pylon below it, and the drag of
the pair exceeds the sum of the isolated parts. Figure 4.24 gives the factor

    k_i = interference drag / drag of the isolated hub component

against the hub gap over pylon width Z/W_p, for fuselage angles of attack of
-3, 0, 3, 6 and 9 deg. The hub group drag is then, p. 306:

    f_M = (1 + K_i) (f_hub + f_shaft)

Example helicopter: Z/W_p = 2.8/9 = 0.3, alpha_F = -5 deg, K_i = 0.15,
f_M = 1.15 (5.8 + 0.3) = 7.0 ft^2. This component reads 0.19 there (C4-20).

Digitization (C4-20): clean copy; row by row, keeping only the rows where all
five curves are separated (Z/W_p from 0.02 to 0.55); each curve fitted by
a exp(-b Z/W_p) + c (rms 0.011 to 0.024) and tabulated every 0.05 up to 0.7.
Akima in Z/W_p, linear in alpha_F between curves and linearly extrapolated
outside -3..9 deg, since helicopters cruise nose down; a warning is raised
outside the figure.

    Z, W_p, alpha_F, f_hub, f_shaft --> Z_Wp, K_i, f_M
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_Z_WP = np.round(np.arange(0.0, 0.701, 0.05), 2)
_ALPHA_F = np.array([-3.0, 0.0, 3.0, 6.0, 9.0])
_K_I = np.array([
    [0.779, 0.624, 0.502, 0.406, 0.330, 0.270, 0.224, 0.187, 0.158, 0.135, 0.117, 0.103,
     0.092, 0.083, 0.076],
    [0.911, 0.738, 0.601, 0.492, 0.405, 0.336, 0.281, 0.237, 0.202, 0.175, 0.153, 0.135,
     0.121, 0.110, 0.101],
    [1.085, 0.865, 0.694, 0.561, 0.457, 0.377, 0.314, 0.265, 0.227, 0.198, 0.175, 0.157,
     0.143, 0.132, 0.124],
    [1.316, 1.037, 0.824, 0.661, 0.536, 0.441, 0.368, 0.312, 0.269, 0.236, 0.211, 0.192,
     0.177, 0.166, 0.157],
    [1.557, 1.236, 0.987, 0.795, 0.646, 0.531, 0.442, 0.374, 0.320, 0.279, 0.248, 0.223,
     0.204, 0.189, 0.178]])
_CURVES = [InterpND(method='akima', points=_Z_WP, values=row, extrapolate=False) for row in _K_I]


def hub_pylon_factor(Z_Wp, alpha_F):
    """Figure 4.24 K_i and its derivatives d/d(Z/W_p), d/d(alpha_F)."""
    z = np.clip(np.real(Z_Wp), _Z_WP[0], _Z_WP[-1])
    vals, slopes = zip(*(c.interpolate(np.atleast_1d(z), compute_derivative=True)
                         for c in _CURVES))
    vals, slopes = np.ravel(vals), np.ravel(slopes)

    a = np.real(alpha_F)
    k = min(max(int(np.searchsorted(_ALPHA_F, a, side='right')) - 1, 0), len(_ALPHA_F) - 2)
    t = (alpha_F - _ALPHA_F[k]) / (_ALPHA_F[k + 1] - _ALPHA_F[k])       # linear, extrapolates
    K_i = (1 - t) * vals[k] + t * vals[k + 1]
    dK_dz = ((1 - t) * slopes[k] + t * slopes[k + 1]) if _Z_WP[0] < np.real(Z_Wp) < _Z_WP[-1] else 0.0
    dK_da = (vals[k + 1] - vals[k]) / (_ALPHA_F[k + 1] - _ALPHA_F[k])
    return K_i, dK_dz, dK_da


class HubPylonInterferenceComp(om.ExplicitComponent):
    """Hub-pylon interference factor and hub group drag, p. 306."""

    def setup(self):
        self.add_input('Z', val=1.0, units='ft', desc='gap between hub and pylon')
        self.add_input('W_p', val=1.0, units='ft', desc='pylon width')
        self.add_input('alpha_F', val=0.0, units='deg', desc='fuselage angle of attack')
        self.add_input('f_hub', val=0.0, units='ft**2', desc='isolated hub flat plate area')
        self.add_input('f_shaft', val=0.0, units='ft**2', desc='shaft flat plate area')
        self.add_output('Z_Wp', val=0.3, desc='hub gap over pylon width')
        self.add_output('K_i', val=0.2, desc='interference factor, Figure 4.24')
        self.add_output('f_M', val=0.0, units='ft**2', desc='hub group equivalent flat plate area')

        self.declare_partials('Z_Wp', ['Z', 'W_p'])
        self.declare_partials('K_i', ['Z', 'W_p', 'alpha_F'])
        self.declare_partials('f_M', ['Z', 'W_p', 'alpha_F', 'f_hub', 'f_shaft'])

    def compute(self, inputs, outputs):
        Z_Wp = inputs['Z'][0] / inputs['W_p'][0]
        alpha = np.real(inputs['alpha_F'][0])
        if not (_Z_WP[0] <= np.real(Z_Wp) <= _Z_WP[-1] and _ALPHA_F[0] <= alpha <= _ALPHA_F[-1]):
            warnings.warn(f'Z/W_p = {np.real(Z_Wp):.2f}, alpha_F = {alpha:.1f} deg outside '
                          'Figure 4.24 (0-0.7, -3..9 deg); alpha is extrapolated, Z/W_p held.',
                          stacklevel=2)
        K_i = hub_pylon_factor(Z_Wp, inputs['alpha_F'][0])[0]
        outputs['Z_Wp'] = Z_Wp
        outputs['K_i'] = K_i
        outputs['f_M'] = (1.0 + K_i) * (inputs['f_hub'] + inputs['f_shaft'])

    def compute_partials(self, inputs, partials):
        Z, W, f = inputs['Z'][0], inputs['W_p'][0], inputs['f_hub'][0] + inputs['f_shaft'][0]
        K_i, dK_dz, dK_da = hub_pylon_factor(Z / W, inputs['alpha_F'][0])
        dz_dZ, dz_dW = 1.0 / W, -Z / W ** 2

        partials['Z_Wp', 'Z'], partials['Z_Wp', 'W_p'] = dz_dZ, dz_dW
        partials['K_i', 'Z'], partials['K_i', 'W_p'] = dK_dz * dz_dZ, dK_dz * dz_dW
        partials['K_i', 'alpha_F'] = dK_da
        partials['f_M', 'Z'] = dK_dz * dz_dZ * f
        partials['f_M', 'W_p'] = dK_dz * dz_dW * f
        partials['f_M', 'alpha_F'] = dK_da * f
        partials['f_M', 'f_hub'] = partials['f_M', 'f_shaft'] = 1.0 + K_i
