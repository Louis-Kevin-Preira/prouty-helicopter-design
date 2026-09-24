"""
TurnsPullupsGroup -- G1, turns and pullups.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Turns and Pullups" pp. 340-346.

Power required in a steady turn (p. 343) is not a component here: it is the
level-flight power of the Chapter 4 chain at the effective weight n*GW: see
SteadyTurnPowerGroup, kept separate because it carries a full trim.
"""
import openmdao.api as om

from prouty.special_performance.load_factor_comp import LoadFactorComp
from prouty.special_performance.thrust_capability_comp import ThrustCapabilityComp
from prouty.special_performance.turn_cyclic_relief_comp import TurnCyclicReliefComp
from prouty.special_performance.turn_energy_power_comp import TurnEnergyPowerComp
from prouty.special_performance.turn_kinematics_comp import TurnKinematicsComp

_TURN_MODES = ('bank', 'turn_rate', 'pitch_rate')


class TurnsPullupsGroup(om.Group):
    """Load factor -> turn kinematics -> cyclic relief, energy power, thrust margin.

    Turn modes ('bank', 'turn_rate', 'pitch_rate') include the steady-turn
    kinematics and the 180 deg energy-trade power; for 'pullup' and 'pushover'
    theta_dot is a promoted input.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('load_factor_mode', default='bank',
                             values=_TURN_MODES + ('pullup', 'pushover'))
        self.options.declare('boundary', default='steady_turn',
                             values=('transient', 'steady_turn', 'level'))
        self.options.declare('turn_time', default='coherent', values=('coherent', 'book'))

    def setup(self):
        nn = self.options['num_nodes']
        mode = self.options['load_factor_mode']

        self.add_subsystem('load_factor', LoadFactorComp(num_nodes=nn, mode=mode),
                           promotes_inputs=['*'], promotes_outputs=['n'])

        if mode in _TURN_MODES:
            kin_out = ['R_turn', 'phi', 'theta_dot']
            kin_out += [] if mode == 'turn_rate' else ['omega']
            if mode == 'bank':
                kin_out.remove('phi')
            if mode == 'pitch_rate':
                kin_out.remove('theta_dot')
            self.add_subsystem('kinematics', TurnKinematicsComp(num_nodes=nn),
                               promotes_inputs=['V', 'n'], promotes_outputs=kin_out)
            self.add_subsystem('energy_power',
                               TurnEnergyPowerComp(num_nodes=nn, turn_time=self.options['turn_time']),
                               promotes=['*'])

        self.add_subsystem('cyclic_relief', TurnCyclicReliefComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('thrust_capability',
                           ThrustCapabilityComp(num_nodes=nn, boundary=self.options['boundary']),
                           promotes=['*'])
