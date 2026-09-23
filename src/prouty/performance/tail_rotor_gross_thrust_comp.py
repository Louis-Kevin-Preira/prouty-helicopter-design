"""
TailRotorGrossThrustComp -- tail rotor thrust needed to overcome the fin force.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Tail Rotor-Fin Interference in Hover", p. 283 (equation) and
Figure 4.9 p. 286; example p. 286 and p. 309.

The fin force F opposes the rotor thrust, T_net = T_gross (1 - F/T), so the
rotor must produce, p. 283:

    T_gross = T_req / (1 - F/T)

T_req is the net antitorque thrust required by the yaw balance (G5, p. 309).

The example uses T_gross = 1.125 T_net for F/T = 0.125 (pp. 286, 309), i.e.
1 + F/T, the first order expansion; the equation of p. 283 gives 1.143 and is
the one implemented (C4-1).

    T_req (nn,), F_T --> T_gross (nn,)
"""

import numpy as np
import openmdao.api as om


class TailRotorGrossThrustComp(om.ExplicitComponent):
    """Gross tail rotor thrust, p. 283."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('T_req', val=np.zeros(nn), units='lbf', desc='net tail rotor thrust required')
        self.add_input('F_T', val=0.0, desc='fin force over tail rotor thrust, Figure 4.9')
        self.add_output('T_gross', val=np.zeros(nn), units='lbf', desc='tail rotor thrust')

        self.declare_partials('T_gross', 'T_req', rows=ar, cols=ar)
        self.declare_partials('T_gross', 'F_T')

    def compute(self, inputs, outputs):
        outputs['T_gross'] = inputs['T_req'] / (1.0 - inputs['F_T'])

    def compute_partials(self, inputs, partials):
        gap = 1.0 - inputs['F_T'][0]
        partials['T_gross', 'T_req'] = np.full(self.options['num_nodes'], 1.0 / gap)
        partials['T_gross', 'F_T'] = (inputs['T_req'] / gap ** 2).reshape(-1, 1)
