"""
TurboshaftRatingsComp -- uninstalled ratings of the example turboshaft.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Engine Performance": Figure 4.1 p. 274 (ratings against altitude,
standard and 95 deg F days, V = 0), Figure 4.2 p. 275 (forward speed), torque
limit p. 320. Kept to validate Chapter 4 on the example helicopter.

Figure 4.1 -- six exact straight lines (pixel digitization, rms 1-2 hp):

    P_day,r(h) = P0_day,r + s_day,r h

The 95 deg F day is isothermal (C4-4). At altitude h the two days are
blended linearly in the air temperature (T_std = T_0 (1 - L h)):

    w    = (T_air - T_std) / (T_95 - T_std)          day_temperature_comp.hot_day_weight
    P0_r = P_std,r + w (P_95,r - P_std,r)

Figure 4.2 -- the ram gain grows as V^2. One coefficient per rating, least
squares on both conditions of the figure (C4-6):

    k_r = 1 + K_r (gamma/2) M^2,   M = V / V_son

Limits: maximum continuous never above intermediate (Fig. 4.1 above
23,500 ft), then every rating capped by the torque limit, after the ram
gain (Fig. 4.2: intermediate reaches it near 165 kt). Both minima use the
quadratic fillet of _smooth.smoothmin.

    altitude, T_air, V, V_son (nn,), P_torque_limit --> P_eng (nn, 3)
"""

import warnings

import numpy as np
import openmdao.api as om

from prouty.hover.atmosphere_comp import T_0
from prouty.performance._smooth import smoothmin
from prouty.performance.day_temperature_comp import hot_day_weight
from prouty.performance.piston_power_lapse_comp import RATINGS

GAMMA = 1.4

# Figure 4.1 p. 274, order of RATINGS: takeoff, intermediate, max continuous
P0_STD = np.array([2129.0, 2002.0, 1598.0])                 # hp
SLOPE_STD = np.array([-40.0, -40.2, -23.0]) * 1e-3          # hp/ft
P0_95F = np.array([1862.0, 1722.0, 1256.0])
SLOPE_95F = np.array([-57.85, -57.8, -43.0]) * 1e-3

# Figure 4.2 p. 275
K_RAM = np.array([1.12, 0.65, 0.91])

_TO, _INT, _MCP = range(3)


class TurboshaftRatingsComp(om.ExplicitComponent):
    """Example turboshaft ratings against altitude, temperature and speed."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('blend_width', types=float, default=10.0,
                             desc='half width of the smooth minima, hp')

    def setup(self):
        nn = self.options['num_nodes']
        nr = len(RATINGS)

        self.add_input('altitude', val=np.zeros(nn), units='ft')
        self.add_input('T_air', val=np.full(nn, T_0), units='degR')
        self.add_input('V', val=np.zeros(nn), units='ft/s', desc='true airspeed')
        self.add_input('V_son', val=np.full(nn, 1116.4), units='ft/s')
        self.add_input('P_torque_limit', val=2080.0, units='hp', desc='per engine')
        self.add_output('P_eng', val=np.ones((nn, nr)), units='hp',
                        desc='uninstalled ratings per engine: ' + ', '.join(RATINGS))

        rows = np.arange(nn * nr)
        cols = np.repeat(np.arange(nn), nr)
        self.declare_partials('P_eng', ['altitude', 'T_air', 'V', 'V_son'], rows=rows, cols=cols)
        self.declare_partials('P_eng', 'P_torque_limit')

    def _evaluate(self, inputs):
        """P_eng (nn, 3) and its derivatives with respect to each input."""
        h, T = inputs['altitude'][:, None], inputs['T_air'][:, None]
        V, a = inputs['V'][:, None], inputs['V_son'][:, None]
        w_hp = self.options['blend_width']

        # Figure 4.1: two days, blended in temperature
        w, dw_dh, dw_dT = hot_day_weight(h, T)
        P_std = P0_STD + SLOPE_STD * h
        dP_day = (P0_95F + SLOPE_95F * h) - P_std
        P0 = P_std + w * dP_day

        dP0_dh = SLOPE_STD + w * (SLOPE_95F - SLOPE_STD) + dw_dh * dP_day
        dP0_dT = dw_dT * dP_day

        # Figure 4.2: ram gain
        M = V / a
        k = 1.0 + K_RAM * 0.5 * GAMMA * M ** 2
        dk_dM = K_RAM * GAMMA * M
        P = P0 * k
        d = {'altitude': dP0_dh * k, 'T_air': dP0_dT * k,
             'V': P0 * dk_dM / a, 'V_son': -P0 * dk_dM * M / a}

        # maximum continuous <= intermediate
        P_mcp, s_mcp, s_int = smoothmin(P[:, _MCP], P[:, _INT], w_hp)
        for dd in d.values():
            dd[:, _MCP] = s_mcp * dd[:, _MCP] + s_int * dd[:, _INT]
        P[:, _MCP] = P_mcp

        # torque limit
        P, s_P, s_lim = smoothmin(P, inputs['P_torque_limit'][0], w_hp)
        for name in d:
            d[name] = s_P * d[name]
        d['P_torque_limit'] = s_lim
        return P, d

    def compute(self, inputs, outputs):
        w = hot_day_weight(np.real(inputs['altitude']), np.real(inputs['T_air']))[0]
        if np.any(w < -1e-6) or np.any(w > 1.0 + 1e-6):
            warnings.warn('T_air is outside the standard and 95 deg F days of '
                          'Figure 4.1; the ratings are extrapolated.', stacklevel=2)
        outputs['P_eng'] = self._evaluate(inputs)[0]

    def compute_partials(self, inputs, partials):
        d = self._evaluate(inputs)[1]
        for name in ('altitude', 'T_air', 'V', 'V_son'):
            partials['P_eng', name] = d[name].ravel()
        partials['P_eng', 'P_torque_limit'] = d['P_torque_limit'].reshape(-1, 1)
