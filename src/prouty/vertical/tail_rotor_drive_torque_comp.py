"""
TailRotorDriveTorqueComp -- G8, design torque of the tail rotor drive system.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Effect of Rapid Pitch Changes" pp. 115-116, Figure 2.15.

A rapid pitch increase overshoots the steady thrust, up to twice the final
value in Figure 2.15, and the torque with it. The main rotor slows down and
absorbs the transient; the tail rotor cannot, and demands whatever torque its
drive gives. Stopping a hover turn in the torque direction with full opposite
pedal can reach a 30 deg change of angle of attack and fully stalled blades.
Current practice (p. 116) designs the drive for a moderate multiple of the
maximum hover torque, e.g. twice, and asks pilots to avoid extreme turn
reversals:

    Q_T_design = k_design Q_T_hov_max,      k_design = 2 by default

    Q_T_hov_max (nn,) --> Q_T_design (nn,)
"""

import numpy as np
import openmdao.api as om


class TailRotorDriveTorqueComp(om.ExplicitComponent):
    """Design torque of the tail rotor drive, p. 116."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('k_design', default=2.0,
                             desc='multiple of the maximum hover torque, p. 116')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('Q_T_hov_max', val=np.ones(nn), units='ft*lbf',
                       desc='maximum hover torque of the tail rotor')
        self.add_output('Q_T_design', val=np.ones(nn), units='ft*lbf',
                        desc='tail rotor drive design torque')
        self.declare_partials('Q_T_design', 'Q_T_hov_max', rows=ar, cols=ar,
                              val=self.options['k_design'])

    def compute(self, inputs, outputs):
        outputs['Q_T_design'] = self.options['k_design'] * inputs['Q_T_hov_max']
