"""
EngineGroup -- G0, engine performance.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Engine Performance" and "Engine Installation Losses",
pp. 274-277; hot day of Figures 4.1-4.3 (C4-4).

    altitude, [dT_offset | T_day]
        AtmosphereGroup           rho, density_ratio, T_air, V_son   atmosphere=True
    ratings, per engine
        PistonPowerLapseComp      P_eng    engine_type = 'piston_na' | 'piston_turbo'
        TurboshaftRatingsComp     P_eng    engine_type = 'turboshaft'   (Figs 4.1-4.2)
        InstalledPowerComp        P_avail  losses, n_eng, [P_trans]
    fuel, all operating engines
        PistonFuelFlowComp        FF       piston types
        TurboshaftFuelFlowComp    FF       turboshaft               (Fig. 4.3)

P_avail and FF are separate outputs: power available is matched against
power required by the balances of G5-G8, and FF is evaluated at the engine
power required P_req coming from G1.

In a model combining several Chapter 4 groups, place one AtmosphereGroup
first and build this group with atmosphere=False, after PowerLossesGroup.
"""

import openmdao.api as om

from prouty.performance.atmosphere_group import DAYS, AtmosphereGroup
from prouty.performance.installed_power_comp import InstalledPowerComp
from prouty.performance.piston_fuel_flow_comp import PistonFuelFlowComp
from prouty.performance.piston_power_lapse_comp import PistonPowerLapseComp
from prouty.performance.turboshaft_fuel_flow_comp import TurboshaftFuelFlowComp
from prouty.performance.turboshaft_ratings_comp import TurboshaftRatingsComp

ENGINE_TYPES = ('piston_na', 'piston_turbo', 'turboshaft')


class EngineGroup(om.Group):
    """Atmosphere, engine ratings, installed power and fuel flow."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('engine_type', default='piston_na', values=ENGINE_TYPES)
        self.options.declare('day', default='standard', values=DAYS)
        self.options.declare('atmosphere', types=bool, default=True,
                             desc='embed an AtmosphereGroup (False when one runs upstream)')
        self.options.declare('transmission_limit', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        engine = self.options['engine_type']

        if self.options['atmosphere']:
            self.add_subsystem('atmosphere', AtmosphereGroup(num_nodes=nn, day=self.options['day']),
                               promotes=['*'])

        if engine == 'turboshaft':
            ratings = TurboshaftRatingsComp(num_nodes=nn)
            fuel = TurboshaftFuelFlowComp(num_nodes=nn)
        else:
            ratings = PistonPowerLapseComp(num_nodes=nn, supercharged=engine == 'piston_turbo')
            fuel = PistonFuelFlowComp(num_nodes=nn)

        self.add_subsystem('ratings', ratings, promotes=['*'])
        self.add_subsystem('installed', InstalledPowerComp(
            num_nodes=nn, transmission_limit=self.options['transmission_limit']),
            promotes=['*'])
        self.add_subsystem('fuel_flow', fuel, promotes=['*'])
