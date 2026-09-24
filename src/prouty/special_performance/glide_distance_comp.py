"""
GlideDistanceComp -- G2c, glide distance in autorotation.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Steady Rate of Descent in Autorotation" p. 351 and
"Glide Distance" p. 352, Figure 5.6.

Glide ratio L/D = V_fwd / V_descent (p. 351):

    Delta_d = 60 Delta_h V_1 / (R/D)   [R/D in ft/min]  = Delta_h V_1 / (R/D)
    d       = (h_0 + Delta_h) V_1 / (R/D)

R/D: steady rate of descent in autorotation at V_1 (G2b).

    V_1 (nn,), RD (nn,), delta_h (nn,), h_0 --> delta_d (nn,), glide_distance (nn,)
"""

import numpy as np
import openmdao.api as om


class GlideDistanceComp(om.ExplicitComponent):
    """Extra glide distance due to the zoom, and total glide distance, pp. 351-352."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('V_1', val=150.0 * np.ones(nn), units='ft/s')
        self.add_input('RD', val=25.0 * np.ones(nn), units='ft/s')
        self.add_input('delta_h', val=np.zeros(nn), units='ft')
        self.add_input('h_0', val=0.0, units='ft', desc='height at the power failure')
        self.add_output('delta_d', val=np.zeros(nn), units='ft')
        self.add_output('glide_distance', val=np.zeros(nn), units='ft')
        self.declare_partials('delta_d', ['V_1', 'RD', 'delta_h'], rows=ar, cols=ar)
        self.declare_partials('glide_distance', ['V_1', 'RD', 'delta_h'], rows=ar, cols=ar)
        self.declare_partials('glide_distance', 'h_0', rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        ld = inputs['V_1'] / inputs['RD']
        outputs['delta_d'] = inputs['delta_h'] * ld
        outputs['glide_distance'] = (inputs['h_0'] + inputs['delta_h']) * ld

    def compute_partials(self, inputs, J):
        V, RD, dh, h0 = inputs['V_1'], inputs['RD'], inputs['delta_h'], inputs['h_0']
        ld = V / RD
        for out, h in (('delta_d', dh), ('glide_distance', h0 + dh)):
            J[out, 'V_1'] = h / RD
            J[out, 'RD'] = -h * ld / RD
            J[out, 'delta_h'] = ld
        J['glide_distance', 'h_0'] = ld
