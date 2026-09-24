"""
VortexRingBoundariesComp -- G5, boundaries of the vortex ring state.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "States of Flow" p. 95; "Characteristics of the Vortex Ring State"
pp. 102-107.

Rates of descent, as fractions of v_1hov:

    classical vortex ring state        0     to 2 v_1hov                   p. 95
    vibration in flight (ref. 2.3)     0.23  to 1.25 v_1hov (rough_bounds)  p. 102
    maximum wake instability (2.7)     0.707 v_1hov, tip vortices in plane  p. 105

Forward speed clears the vortex ring state: about 10 kt on a small tandem
(ref. 2.3), 25 kt on a larger single rotor helicopter (ref. 2.4), p. 102;
option V_escape_kt, returned as an output so that a design can constrain it.

vrs_margin is a smooth indicator of the rough region for optimization (not in
the book): with x = V_D_bar and bounds (x_lo, x_hi),

    vrs_margin = (x - x_lo)(x_hi - x) / ((x_hi - x_lo)/2)^2

1 at mid-region, 0 on the boundaries, negative outside: vrs_margin <= 0 keeps
a vertical descent out of the rough region.

Figures 2.6-2.8 (thrust fluctuations, power settling, twist) are wind tunnel
data on model rotors and are not digitized (decision, Sept 2026).

    v_hov, V_D_bar (nn,) --> V_D_rough_low, V_D_rough_high, V_D_max_instability,
                             V_D_classic_high, vrs_margin (nn,), V_escape
"""

import numpy as np
import openmdao.api as om

CLASSIC_HIGH = 2.0          # p. 95
MAX_INSTABILITY = 0.707     # p. 105, ref. 2.7


class VortexRingBoundariesComp(om.ExplicitComponent):
    """Vortex ring state boundaries and a smooth margin, pp. 95, 102-105."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('rough_bounds', types=tuple, default=(0.23, 1.25),
                             desc='V_D / v_1hov bounds of vortex ring vibration, p. 102')
        self.options.declare('V_escape_kt', default=25.0,
                             desc='forward speed clearing the vortex ring state, p. 102')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('v_hov', val=np.ones(nn), units='ft/s', desc='hover induced velocity')
        self.add_input('V_D_bar', val=np.zeros(nn), desc='V_D / v_1hov, descent > 0')

        self._ratios = {'V_D_rough_low': self.options['rough_bounds'][0],
                        'V_D_rough_high': self.options['rough_bounds'][1],
                        'V_D_max_instability': MAX_INSTABILITY,
                        'V_D_classic_high': CLASSIC_HIGH}
        for name, ratio in self._ratios.items():
            self.add_output(name, val=ratio * np.ones(nn), units='ft/s')
            self.declare_partials(name, 'v_hov', rows=ar, cols=ar, val=ratio)
        self.add_output('vrs_margin', val=np.zeros(nn), desc='> 0 inside the rough region')
        self.declare_partials('vrs_margin', 'V_D_bar', rows=ar, cols=ar)
        self.add_output('V_escape', val=self.options['V_escape_kt'], units='kn')

    def compute(self, inputs, outputs):
        for name, ratio in self._ratios.items():
            outputs[name] = ratio * inputs['v_hov']
        lo, hi = self.options['rough_bounds']
        x = inputs['V_D_bar']
        outputs['vrs_margin'] = (x - lo) * (hi - x) / (0.5 * (hi - lo)) ** 2
        outputs['V_escape'] = self.options['V_escape_kt']

    def compute_partials(self, inputs, partials):
        lo, hi = self.options['rough_bounds']
        partials['vrs_margin', 'V_D_bar'] = (lo + hi - 2.0 * inputs['V_D_bar']) / (
            0.5 * (hi - lo)) ** 2
