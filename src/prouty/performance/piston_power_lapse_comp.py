"""
PistonPowerLapseComp -- uninstalled piston engine ratings.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Engine Performance", p. 274: three ratings (takeoff, intermediate,
maximum continuous), limited on reciprocating engines by intake manifold
pressure and rpm, and functions of altitude and temperature.

Prouty gives no lapse law, only the turboshaft charts of Figures 4.1-4.3. The
law below is external to the book (C4-5). At full manifold pressure and fixed
rpm the power follows the mass of air per cycle, taken as

    phi = (p/p_0) sqrt(T_0/T) = sigma sqrt(T/T_0)       since p/p_0 = sigma T/T_0

Normally aspirated:

    P_k = P_SL,k  phi

Supercharged: manifold pressure is held up to the critical altitude h_c of the
standard day, where phi_c = theta_c ** (5.2559 - 0.5), theta_c = 1 - L h_c.
Above it the engine is aspirated again:

    P_k = P_SL,k  smoothmin(1, phi/phi_c)

The minimum is the quadratic fillet of _smooth.smoothmin over
|phi/phi_c - 1| < w: C1 and monotone in altitude for the Newton solves of the
ceilings, and w/4 below 1 at h_c. Below h_c on a hot
day phi/phi_c can exceed 1: the supercharger then makes up the temperature
loss too, and the engine is flat rated.

    density_ratio, T_air (nn,), P_SL (3,), [h_crit] --> P_eng (nn, 3)
"""

import numpy as np
import openmdao.api as om

from prouty.hover.atmosphere_comp import EXP, LAPSE, T_0
from prouty.performance._smooth import smoothmin

RATINGS = ('takeoff', 'intermediate', 'max_continuous')


class PistonPowerLapseComp(om.ExplicitComponent):
    """Piston engine ratings against density ratio and temperature."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('supercharged', types=bool, default=False)
        self.options.declare('blend_width', types=float, default=0.02,
                             desc='half width of the smoothmin, in phi/phi_c')

    def setup(self):
        nn = self.options['num_nodes']
        nr = len(RATINGS)

        self.add_input('density_ratio', val=np.ones(nn))
        self.add_input('T_air', val=np.full(nn, T_0), units='degR')
        self.add_input('P_SL', val=np.ones(nr), units='hp',
                       desc='sea level standard ratings per engine: ' + ', '.join(RATINGS))
        self.add_output('P_eng', val=np.ones((nn, nr)), units='hp',
                        desc='uninstalled ratings per engine')

        node_rows = np.arange(nn * nr)
        node_cols = np.repeat(np.arange(nn), nr)
        self.declare_partials('P_eng', ['density_ratio', 'T_air'],
                              rows=node_rows, cols=node_cols)
        self.declare_partials('P_eng', 'P_SL', rows=node_rows,
                              cols=np.tile(np.arange(nr), nn))

        if self.options['supercharged']:
            self.add_input('h_crit', val=0.0, units='ft',
                           desc='critical altitude, standard day')
            self.declare_partials('P_eng', 'h_crit')

    def _lapse(self, inputs):
        """Lapse factor g, dg/dphi and dg/dh_crit, each (nn,)."""
        phi = inputs['density_ratio'] * np.sqrt(inputs['T_air'] / T_0)
        if not self.options['supercharged']:
            return phi, np.ones_like(phi), np.zeros_like(phi)

        theta_c = 1.0 - LAPSE * inputs['h_crit'][0]
        phi_c = theta_c ** (EXP - 0.5)
        dphic_dh = -(EXP - 0.5) * LAPSE * theta_c ** (EXP - 1.5)

        g, dg_dr, _ = smoothmin(phi / phi_c, 1.0, self.options['blend_width'])
        return g, dg_dr / phi_c, -dg_dr * phi / phi_c ** 2 * dphic_dh

    def compute(self, inputs, outputs):
        g = self._lapse(inputs)[0]
        outputs['P_eng'] = np.outer(g, inputs['P_SL'])

    def compute_partials(self, inputs, partials):
        g, dg_dphi, dg_dh = self._lapse(inputs)
        P_SL = inputs['P_SL']
        sigma, T = inputs['density_ratio'], inputs['T_air']

        dphi_dsigma = np.sqrt(T / T_0)
        dphi_dT = 0.5 * sigma / np.sqrt(T * T_0)

        partials['P_eng', 'density_ratio'] = np.outer(dg_dphi * dphi_dsigma, P_SL).ravel()
        partials['P_eng', 'T_air'] = np.outer(dg_dphi * dphi_dT, P_SL).ravel()
        partials['P_eng', 'P_SL'] = np.repeat(g, len(P_SL))
        if self.options['supercharged']:
            partials['P_eng', 'h_crit'] = np.outer(dg_dh, P_SL).reshape(-1, 1)
