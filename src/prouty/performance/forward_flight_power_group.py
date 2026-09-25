"""
ForwardFlightPowerGroup -- engine power required in level forward flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Forward Flight Performance" pp. 317-319 and Figure 4.38;
Chapter 3 trim pp. 192-199 and Table 3.5 case 2 p. 233; power losses p. 277.

    advance_ratio  AdvanceRatioComp        mu = V / (Omega R)
    trim           TrimConditionsGroup     hp_M, hp_T, alpha_F, theta_0, ...   Chapter 3
    mach           TipMachComp             M_tip, M_190
    drag_rise      DragRiseMachComp        M_dr3, M_ratio
    comp           CompressibilityTorqueComp  dCQ_sigma_comp                   Fig. 3.43
    comp_power     RotorPowerComp          hp_comp
    stall          StallTorqueIncrementComp  dCQ_sigma_stall (stall=True, C5-6)  pp. 230, 258-266
    stall_power    RotorPowerComp          hp_stall
    main_power     P_MR = hp_M + hp_comp + hp_stall
    losses         PowerLossesGroup (G1)   P_req                                p. 277-278

Compressibility is left out of the trim loop, where Chapter 3 found it puts
hp_M 10 % high at mu = 0.45, and its penalty is added afterwards, the way
Table 3.5 case 2 does it. Figure 4.38 separates it the same way, as
"additional compressibility losses" on its own scale.

The rotor model is the closed form of Chapter 3: a blade element integration
at every speed, weight and altitude of a range integral would cost far more
than it buys here. rotor_options and trim_options pass anything else through.

    V, GW, f, altitude (through rho, V_son), rotor and tail rotor geometry
        --> mu, hp_M, hp_T, hp_comp, P_MR, P_TR, P_req, P_loss, trim angles
"""

import numpy as np
import openmdao.api as om

from prouty.forward_flight import (AdvanceRatioComp, CompressibilityTorqueComp,
                                   DragRiseMachComp, RotorPowerComp, TipMachComp,
                                   TrimConditionsGroup)
from prouty.performance.gearbox_loss_comp import EXAMPLE_GEARBOXES, _check
from prouty.performance.power_losses_group import PowerLossesGroup
from prouty.performance.stall_torque_increment_comp import StallTorqueIncrementComp


class ForwardFlightPowerGroup(om.Group):
    """Chapter 3 level trim plus the compressibility penalty and the drive losses."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('compressibility', types=bool, default=True,
                             desc='add the Figure 3.43 penalty after the trim')
        self.options.declare('stall', types=bool, default=False,
                             desc='add the chart-calibrated stall torque increment (C5-6)')
        self.options.declare('trim_options', types=dict, default={},
                             desc='passed to TrimConditionsGroup')
        self.options.declare('gearboxes', types=dict, default=EXAMPLE_GEARBOXES,
                             check_valid=lambda name, value: _check(value))
        self.options.declare('density_scaled_losses', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        trim_options = dict(mode='level', rotor='closed_form',
                            rotor_options={'compressibility': False})
        trim_options.update(self.options['trim_options'])

        self.add_subsystem('advance_ratio', AdvanceRatioComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('trim', TrimConditionsGroup(num_nodes=nn, **trim_options),
                           promotes=['*'])

        if self.options['compressibility']:
            self.add_subsystem('mach', TipMachComp(num_nodes=nn), promotes=['*'])
            self.add_subsystem('drag_rise', DragRiseMachComp(num_nodes=nn), promotes=['*'])
            self.add_subsystem('comp', CompressibilityTorqueComp(num_nodes=nn), promotes=['*'])
            self.add_subsystem('comp_power', RotorPowerComp(num_nodes=nn),
                               promotes_inputs=[('CQ_sigma', 'dCQ_sigma_comp'), 'rho',
                                                'A_b', 'V_tip'],
                               promotes_outputs=[('hp', 'hp_comp')])
        else:
            self.add_subsystem('no_comp', om.IndepVarComp('hp_comp', val=np.zeros(nn),
                                                          units='hp'), promotes=['*'])

        if self.options['stall']:
            self.add_subsystem('stall', StallTorqueIncrementComp(num_nodes=nn), promotes=['*'])
            self.add_subsystem('stall_power', RotorPowerComp(num_nodes=nn),
                               promotes_inputs=[('CQ_sigma', 'dCQ_sigma_stall'), 'rho',
                                                'A_b', 'V_tip'],
                               promotes_outputs=[('hp', 'hp_stall')])
        else:
            self.add_subsystem('no_stall', om.IndepVarComp('hp_stall', val=np.zeros(nn),
                                                           units='hp'), promotes=['*'])

        self.add_subsystem('main_power', om.ExecComp(
            'P_MR = hp_M + hp_comp + hp_stall', P_MR={'units': 'hp', 'shape': nn},
            hp_M={'units': 'hp', 'shape': nn}, hp_comp={'units': 'hp', 'shape': nn},
            hp_stall={'units': 'hp', 'shape': nn}), promotes=['*'])
        self.add_subsystem('tail_power', om.ExecComp(
            'P_TR = hp_T', P_TR={'units': 'hp', 'shape': nn},
            hp_T={'units': 'hp', 'shape': nn}), promotes=['*'])
        self.add_subsystem('losses', PowerLossesGroup(
            num_nodes=nn, gearboxes=self.options['gearboxes'],
            density_scaled_losses=self.options['density_scaled_losses']), promotes=['*'])
