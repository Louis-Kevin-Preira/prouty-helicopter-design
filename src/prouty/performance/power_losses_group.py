"""
PowerLossesGroup -- G1, power required losses.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Power Losses", pp. 275-278; engine power in hover p. 311.

    load_elec, flow_hyd, p_hyd, eta_gen, eta_hyd, P_other
        AccessoryLossComp          P_loss_acc                       p. 278
    P_MR, P_TR, P_loss_acc, P_design_<gearbox>, [density_ratio]
        EnginePowerRequiredComp    P_req, P_loss                    pp. 277-278, 311

P_req is the power required from the engines, all engines. Promoted next to
EngineGroup (G0) it feeds the fuel flow directly, so this group must run
before EngineGroup. With density_scaled_losses=True, density_ratio must come
from an AtmosphereGroup placed upstream of both:

    AtmosphereGroup -> PowerLossesGroup -> EngineGroup(atmosphere=False)

GearboxLossComp is not in the group: EnginePowerRequiredComp already
contains the gearbox losses, with the loop on P_req closed exactly (C4-9).
"""

import openmdao.api as om

from prouty.performance.accessory_loss_comp import AccessoryLossComp
from prouty.performance.engine_power_required_comp import EnginePowerRequiredComp
from prouty.performance.gearbox_loss_comp import EXAMPLE_GEARBOXES, _check


class PowerLossesGroup(om.Group):
    """Accessory and gearbox losses added to the rotor powers."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('gearboxes', types=dict, default=EXAMPLE_GEARBOXES,
                             check_valid=lambda name, value: _check(value))
        self.options.declare('density_scaled_losses', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('accessories', AccessoryLossComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('engine_power', EnginePowerRequiredComp(
            num_nodes=nn, gearboxes=self.options['gearboxes'],
            density_scaled_losses=self.options['density_scaled_losses']), promotes=['*'])
