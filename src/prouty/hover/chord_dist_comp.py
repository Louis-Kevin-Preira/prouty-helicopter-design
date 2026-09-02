"""
ChordDistComp -- blade chord distribution, geometric solidity and
thrust-weighted solidity.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, "Blade Element Method", p. 17; used at step 3 of the combined
momentum and blade element method, p. 69.

Planform: constant chord c_root out to the taper start x_t, then a linear
taper to the tip chord c_tip. Setting c_tip = c_root (the default) or x_t = 1
gives a constant-chord blade, so no separate code path is needed.

    sigma   = b * c_bar / (pi R)      c_bar = mean chord           (p. 17)
    sigma_T = b * c_e   / (pi R)      c_e   = thrust-weighted chord (p. 17)

The thrust-weighted chord is the chord averaged with the (r/R)^2 loading of
the thrust integral, c_e = 3 * int_0^1 (c/R)(r/R)^2 d(r/R). Carrying out the
integral for this planform and cancelling the removable singularity at
x_t = 1 gives the closed forms used below.

    r_R --> ChordDistComp --> c_R (nn,), sigma, sigma_T
"""

import numpy as np
import openmdao.api as om

TINY = 1e-12


class ChordDistComp(om.ExplicitComponent):
    """Local chord ratio c/R and the two solidities."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
        self.add_input('b', val=4.0, desc='number of blades')
        self.add_input('c_root_R', val=0.0668, desc='inboard chord ratio, c/R')
        self.add_input('c_tip_R', val=0.0668, desc='tip chord ratio, c/R')
        self.add_input('x_t', val=1.0, desc='taper start station, r/R')

        self.add_output('c_R', shape=(nn,), desc='local chord ratio, c/R')
        self.add_output('sigma', val=0.085, desc='geometric solidity')
        self.add_output('sigma_T', val=0.085, desc='thrust-weighted solidity')

        ar = np.arange(nn)
        self.declare_partials('c_R', 'r_R', rows=ar, cols=ar)
        self.declare_partials('c_R', ['c_root_R', 'c_tip_R', 'x_t'])
        self.declare_partials(['sigma', 'sigma_T'],
                              ['b', 'c_root_R', 'c_tip_R', 'x_t'])

    def _taper(self, inputs):
        """Taper fraction t at each station, and the tapered-region mask."""
        x_t = inputs['x_t']
        den = 1.0 - x_t
        if np.real(den) < TINY:                      # untapered blade
            return np.zeros_like(inputs['r_R']), np.zeros_like(inputs['r_R'], bool)
        mask = np.real(inputs['r_R']) > np.real(x_t)
        t = np.where(mask, (inputs['r_R'] - x_t) / den, 0.0)
        return t, mask

    def compute(self, inputs, outputs):
        c_r, c_t, x_t, b = (inputs['c_root_R'], inputs['c_tip_R'],
                            inputs['x_t'], inputs['b'])
        dc = c_t - c_r
        t, _ = self._taper(inputs)

        outputs['c_R'] = c_r + dc * t
        outputs['sigma'] = b / np.pi * (c_r + dc * (1.0 - x_t) / 2.0)
        outputs['sigma_T'] = b / np.pi * (c_r + dc * self._g(x_t))

    @staticmethod
    def _g(x_t):
        """Thrust weighting of the taper increment; (3 - x - x^2 - x^3) / 4."""
        return (3.0 - x_t - x_t ** 2 - x_t ** 3) / 4.0

    def compute_partials(self, inputs, partials):
        c_r, c_t, x_t, b = (inputs['c_root_R'], inputs['c_tip_R'],
                            inputs['x_t'], inputs['b'])
        dc = c_t - c_r
        t, mask = self._taper(inputs)
        den = 1.0 - x_t
        safe_den = den if np.real(den) > TINY else 1.0

        partials['c_R', 'r_R'] = np.where(mask, dc / safe_den, 0.0)
        partials['c_R', 'c_root_R'] = 1.0 - t
        partials['c_R', 'c_tip_R'] = t
        partials['c_R', 'x_t'] = np.where(
            mask, dc * (inputs['r_R'] - 1.0) / safe_den ** 2, 0.0)

        g = self._g(x_t)
        dg = -(1.0 + 2.0 * x_t + 3.0 * x_t ** 2) / 4.0

        partials['sigma', 'b'] = (c_r + dc * den / 2.0) / np.pi
        partials['sigma', 'c_root_R'] = b / np.pi * (1.0 + x_t) / 2.0
        partials['sigma', 'c_tip_R'] = b / np.pi * den / 2.0
        partials['sigma', 'x_t'] = -b / np.pi * dc / 2.0

        partials['sigma_T', 'b'] = (c_r + dc * g) / np.pi
        partials['sigma_T', 'c_root_R'] = b / np.pi * (1.0 - g)
        partials['sigma_T', 'c_tip_R'] = b / np.pi * g
        partials['sigma_T', 'x_t'] = b / np.pi * dc * dg
