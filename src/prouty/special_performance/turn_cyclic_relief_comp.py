"""
TurnCyclicReliefComp -- G1, cyclic pitch relief due to pitch rate.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Turns and Pullups" p. 342; Chapter 7 p. 473.
"""

import numpy as np
import openmdao.api as om


class TurnCyclicReliefComp(om.ExplicitComponent):
    """Cyclic change required to precess the rotor at a pitch rate.

    delta_B1 = -(16/gamma) theta_dot / Omega      p. 342 (from Ch. 7, p. 473, hover, e = 0)
    Negative value = less retreating-blade pitch (stall relief).
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('theta_dot', val=np.zeros(nn), units='rad/s')
        self.add_input('lock_number', val=8.1)
        self.add_input('Omega', val=21.67, units='rad/s')
        self.add_output('delta_B1', val=np.zeros(nn), units='rad')
        ar = np.arange(nn)
        self.declare_partials('delta_B1', 'theta_dot', rows=ar, cols=ar)
        self.declare_partials('delta_B1', ['lock_number', 'Omega'], rows=ar, cols=np.zeros(nn, int))

    def compute(self, inputs, outputs):
        outputs['delta_B1'] = -16. * inputs['theta_dot'] / (inputs['lock_number'] * inputs['Omega'])

    def compute_partials(self, inputs, J):
        q, gam, Om = inputs['theta_dot'], inputs['lock_number'], inputs['Omega']
        J['delta_B1', 'theta_dot'] = -16. / (gam * Om) * np.ones_like(q)
        J['delta_B1', 'lock_number'] = 16. * q / (gam ** 2 * Om)
        J['delta_B1', 'Omega'] = 16. * q / (gam * Om ** 2)
