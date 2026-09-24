"""
RotorSpeedDecayGroup -- G2a, rotor speed decay following a power failure.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Rotor Speed Decay" pp. 348-350.

    inertia   DriveInertiaComp        J
    t_ke      KineticEnergyTimeComp   t_KE   (Omega_0 = Omega_M)
    decay     RotorSpeedDecayComp     Omega/Omega_0 at the times t
"""

import openmdao.api as om

from prouty.special_performance.drive_inertia_comp import DriveInertiaComp
from prouty.special_performance.kinetic_energy_time_comp import KineticEnergyTimeComp
from prouty.special_performance.rotor_speed_decay_comp import RotorSpeedDecayComp


class RotorSpeedDecayGroup(om.Group):
    """J -> t_KE -> Omega/Omega_0(t), pp. 348-350."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1,
                             desc='number of times t')
        self.options.declare('inertia_ratio', default='coherent', values=('coherent', 'book'))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('inertia', DriveInertiaComp(inertia_ratio=self.options['inertia_ratio']),
                           promotes=['*'])
        self.add_subsystem('t_ke', KineticEnergyTimeComp(),
                           promotes_inputs=['J', ('Omega_0', 'Omega_M'), 'P_0'],
                           promotes_outputs=['t_KE'])
        self.add_subsystem('decay', RotorSpeedDecayComp(num_nodes=nn), promotes=['*'])
