"""
TailRotorFinPowerComp -- tail rotor power with the fin installed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Tail Rotor-Fin Interference in Hover", pp. 285-286, Figure 4.10
p. 287; example p. 286.

Besides the thrust loss, the fin changes the flow through the tail rotor: a
pseudo ground effect for a tractor, a pseudo ceiling effect for a pusher
(p. 285). On the Lockheed 286 pusher tail rotor (Figure 4.10, F/T = 0.13) the
power measured with the fin on was about 94 % of the fin-off power at the
gross thrust. From this one test the book suggests, p. 286:

    P_TR = (1 - (F/T)/2) P_TR,isolated(T_gross)

with P_TR,isolated the power of the isolated tail rotor (Chapter 1) at
T_gross (TailRotorGrossThrustComp). Example helicopter: F/T = 0.125, factor
0.94. Applied to tractor and pusher installations alike (C4-15).

P_TR is the tail rotor power expected by PowerLossesGroup (G1).

    P_TR_iso (nn,), F_T --> P_TR (nn,)
"""

import numpy as np
import openmdao.api as om


class TailRotorFinPowerComp(om.ExplicitComponent):
    """Empirical fin effect on tail rotor power, p. 286."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('P_TR_iso', val=np.zeros(nn), units='hp',
                       desc='isolated tail rotor power at the gross thrust')
        self.add_input('F_T', val=0.0, desc='fin force over tail rotor thrust, Figure 4.9')
        self.add_output('P_TR', val=np.zeros(nn), units='hp', desc='tail rotor power, fin installed')

        self.declare_partials('P_TR', 'P_TR_iso', rows=ar, cols=ar)
        self.declare_partials('P_TR', 'F_T')

    def compute(self, inputs, outputs):
        outputs['P_TR'] = (1.0 - 0.5 * inputs['F_T']) * inputs['P_TR_iso']

    def compute_partials(self, inputs, partials):
        partials['P_TR', 'P_TR_iso'] = np.full(self.options['num_nodes'],
                                              1.0 - 0.5 * inputs['F_T'][0])
        partials['P_TR', 'F_T'] = (-0.5 * inputs['P_TR_iso']).reshape(-1, 1)
