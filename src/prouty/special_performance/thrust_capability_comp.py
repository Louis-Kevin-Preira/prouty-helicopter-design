"""
ThrustCapabilityComp -- G1, rotor thrust capability, Figure 5.2.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Producing Maximum Load Factors" pp. 344-346, Figure 5.2 p. 345.

Each boundary of Fig. 5.2 is a shaded band; both edges are digitized
(p. 345, grid-overlay reading on the scan). A band_fraction input
places the boundary inside the band: 0 = upper edge, 1 = lower edge.
For 'level' the upper edge is the "Low Drag" line, the lower edge "High Drag".
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

MU = np.array([0., .05, .10, .15, .20, .25, .30, .35, .40, .45, .50])

FIG_5_2 = {
    #               upper edge                                                          lower edge
    'transient':   (np.array([.172, .173, .174, .174, .174, .174, .173, .171, .169, .165, .162]),
                    np.array([.155, .158, .161, .163, .163, .162, .161, .159, .158, .154, .150])),
    'steady_turn': (np.array([.170, .165, .160, .158, .153, .147, .141, .134, .127, .121, .113]),
                    np.array([.155, .154, .151, .151, .147, .141, .135, .129, .119, .113, .101])),
    'level':       (np.array([.165, .162, .155, .148, .140, .132, .123, .113, .102, .089, .077]),
                    np.array([.153, .147, .138, .128, .118, .109, .097, .084, .068, .052, .038])),
}


class ThrustCapabilityComp(om.ExplicitComponent):
    """Maximum C_T/sigma(mu) from Fig. 5.2 and the load factor it allows.

    CT_sigma_max = (1 - f) * upper(mu) + f * lower(mu)
    n_max        = CT_sigma_max / CW_sigma          (n = T/GW, p. 340)
    n_margin     = n_max - n                        (>= 0 as optimization constraint)
    Thrust-weighted solidity is to be used for non-constant chord (p. 344).
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('boundary', default='steady_turn', values=tuple(FIG_5_2))

    def setup(self):
        nn = self.options['num_nodes']
        up, lo = FIG_5_2[self.options['boundary']]
        self._up = InterpND(method='akima', points=(MU,), values=up, extrapolate=True)
        self._lo = InterpND(method='akima', points=(MU,), values=lo, extrapolate=True)

        self.add_input('mu', val=np.zeros(nn))
        self.add_input('band_fraction', val=0.5 * np.ones(nn))
        self.add_input('CW_sigma', val=0.08 * np.ones(nn))
        self.add_input('n', val=np.ones(nn))
        self.add_output('CT_sigma_max', val=0.15 * np.ones(nn))
        self.add_output('n_max', val=np.ones(nn))
        self.add_output('n_margin', val=np.zeros(nn))

        ar = np.arange(nn)
        self.declare_partials(['CT_sigma_max', 'n_max', 'n_margin'], ['mu', 'band_fraction'], rows=ar, cols=ar)
        self.declare_partials(['n_max', 'n_margin'], 'CW_sigma', rows=ar, cols=ar)
        self.declare_partials('n_margin', 'n', rows=ar, cols=ar, val=-1.)

    def _edges(self, mu):
        x = mu[:, np.newaxis]
        up, dup = self._up.interpolate(x, compute_derivative=True)
        lo, dlo = self._lo.interpolate(x, compute_derivative=True)
        return up, lo, dup[:, 0], dlo[:, 0]

    def compute(self, inputs, outputs):
        up, lo, _, _ = self._edges(inputs['mu'])
        f = inputs['band_fraction']
        ct = (1. - f) * up + f * lo
        outputs['CT_sigma_max'] = ct
        outputs['n_max'] = ct / inputs['CW_sigma']
        outputs['n_margin'] = outputs['n_max'] - inputs['n']

    def compute_partials(self, inputs, J):
        up, lo, dup, dlo = self._edges(inputs['mu'])
        f, cw = inputs['band_fraction'], inputs['CW_sigma']
        ct = (1. - f) * up + f * lo
        dct_dmu = (1. - f) * dup + f * dlo
        dct_df = lo - up
        J['CT_sigma_max', 'mu'] = dct_dmu
        J['CT_sigma_max', 'band_fraction'] = dct_df
        for out in ('n_max', 'n_margin'):
            J[out, 'mu'] = dct_dmu / cw
            J[out, 'band_fraction'] = dct_df / cw
            J[out, 'CW_sigma'] = -ct / cw ** 2
