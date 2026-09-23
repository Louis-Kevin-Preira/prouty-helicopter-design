"""
CruisePerformanceGroup -- G7, cruise performance at a flight point.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Forward Flight Performance" pp. 317-331, Figures 4.38 and 4.42
to 4.47.

    atmosphere   AtmosphereGroup             rho, T_air, V_son            App. C
    power        ForwardFlightPowerGroup     P_req                        pp. 317-319
    engine       EngineGroup                 P_avail, FF                  pp. 274-278
    range        SpecificRangeComp           SR, V_ground                 p. 323
    endurance    SpecificEnduranceComp       SE                           p. 330

One flight point: speed, gross weight, altitude and wind in, specific range
and specific endurance out. The mission quantities are built on top of it:
BestRangeSpeedBalance and CruiseSpeedBalance find the speeds (through
FuelFlowSlopeComp, which runs this same group as a sub-problem, see
cruise_problem below), and MissionIntegralComp integrates SR or SE over the
gross weights between takeoff and landing.

The chain is explicit apart from the trim inside ForwardFlightPowerGroup, so a
point costs one trim solve.

    V, GW, altitude, V_wind, rotor and airframe inputs --> P_req, FF, SR, SE
"""

import numpy as np
import openmdao.api as om

from prouty.performance.atmosphere_group import DAYS, AtmosphereGroup
from prouty.performance.engine_group import ENGINE_TYPES, EngineGroup
from prouty.performance.forward_flight_power_group import ForwardFlightPowerGroup
from prouty.performance.gearbox_loss_comp import EXAMPLE_GEARBOXES, _check
from prouty.performance.specific_endurance_comp import SpecificEnduranceComp
from prouty.performance.specific_range_comp import SpecificRangeComp


class CruisePerformanceGroup(om.Group):
    """Specific range and endurance at one flight point, pp. 317-331."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('day', default='standard', values=DAYS)
        self.options.declare('engine_type', default='turboshaft', values=ENGINE_TYPES)
        self.options.declare('compressibility', types=bool, default=True)
        self.options.declare('trim_options', types=dict, default={})
        self.options.declare('gearboxes', types=dict, default=EXAMPLE_GEARBOXES,
                             check_valid=lambda name, value: _check(value))
        self.options.declare('density_scaled_losses', types=bool, default=False)

    def setup(self):
        nn, opt = self.options['num_nodes'], self.options

        self.add_subsystem('atmosphere', AtmosphereGroup(num_nodes=nn, day=opt['day']),
                           promotes=['*'])
        self.add_subsystem('power', ForwardFlightPowerGroup(
            num_nodes=nn, compressibility=opt['compressibility'],
            trim_options=opt['trim_options'], gearboxes=opt['gearboxes'],
            density_scaled_losses=opt['density_scaled_losses']), promotes=['*'])
        self.add_subsystem('engine', EngineGroup(num_nodes=nn, engine_type=opt['engine_type'],
                                                 atmosphere=False), promotes=['*'])
        self.add_subsystem('range', SpecificRangeComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('endurance', SpecificEnduranceComp(num_nodes=nn), promotes=['*'])

        self.set_input_defaults('V', np.full(nn, 100.0), units='kn')


def cruise_problem(inputs=None, **group_options):
    """A Problem holding one CruisePerformanceGroup, ready for FuelFlowSlopeComp."""
    prob = om.Problem()
    prob.model.add_subsystem('cruise', CruisePerformanceGroup(**group_options), promotes=['*'])
    if inputs:
        prob.setup()
        for name, val in inputs.items():
            prob.set_val(name, val)
        prob.final_setup()
    return prob
