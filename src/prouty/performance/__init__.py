"""Chapter 4 -- Performance analysis, p. 273-338.

        AtmosphereGroup, reference day and atmosphere, runs first
    G0  EngineGroup(engine_type=...), engine performance         p. 274-277
        DayTemperatureComp, PistonPowerLapseComp, TurboshaftRatingsComp,
        InstalledPowerComp, PistonFuelFlowComp, TurboshaftFuelFlowComp
    G1  PowerLossesGroup, power required losses                   p. 275-278
        GearboxLossComp, AccessoryLossComp, EnginePowerRequiredComp
    G2  VerticalDragGroup, vertical drag in hover                 p. 278-285
        WakeDynamicPressureComp, InducedVelocityRatioComp, VerticalDragComp,
        PseudoGroundEffectComp (with hover.GroundEffectComp, Figure 1.41),
        GroundProximityDownloadComp
    G3  TailRotorFinInterferenceGroup, tail rotor-fin in hover    p. 283-287
        FinInterferenceRatioComp, TailRotorGrossThrustComp, TailRotorFinPowerComp
    G4  ParasiteDragGroup, parasite drag in forward flight        p. 287-308
        FuselageDragComp, NacelleDragComp, RotorHubDragComp, RotorShaftDragComp,
        HubPylonInterferenceComp, LandingGearDragComp, StabilizerDragComp,
        RotorFuselageInterferenceComp, ExhaustDragComp, TotalParasiteDragComp
    G5  hover performance                                         p. 308-312
        HoverPerformanceGroup: TailRotorThrustComp, RotorLoadingMatchComp,
        HoverCeilingBalance, MainRotorHoverPowerComp, RatingSelectComp
    G6  VerticalClimbGroup, vertical climb                        p. 313-317
    G7  forward flight performance                                p. 317-331
        ForwardFlightPowerGroup
        ClimbInducedVelocityComp, VerticalClimbPowerComp, VerticalClimbBalance
"""

from prouty.performance.day_temperature_comp import DayTemperatureComp
from prouty.performance.piston_power_lapse_comp import PistonPowerLapseComp
from prouty.performance.turboshaft_ratings_comp import TurboshaftRatingsComp
from prouty.performance.installed_power_comp import InstalledPowerComp
from prouty.performance.piston_fuel_flow_comp import PistonFuelFlowComp
from prouty.performance.turboshaft_fuel_flow_comp import TurboshaftFuelFlowComp
from prouty.performance.engine_group import ENGINE_TYPES, EngineGroup
from prouty.performance.gearbox_loss_comp import EXAMPLE_GEARBOXES, GearboxLossComp
from prouty.performance.accessory_loss_comp import AccessoryLossComp
from prouty.performance.engine_power_required_comp import EnginePowerRequiredComp
from prouty.performance.power_losses_group import PowerLossesGroup
from prouty.performance.atmosphere_group import AtmosphereGroup
from prouty.performance.wake_dynamic_pressure_comp import WakeDynamicPressureComp
from prouty.performance.induced_velocity_ratio_comp import InducedVelocityRatioComp
from prouty.performance.vertical_drag_comp import VerticalDragComp
from prouty.performance.pseudo_ground_effect_comp import PseudoGroundEffectComp
from prouty.performance.ground_proximity_download_comp import GroundProximityDownloadComp
from prouty.performance.vertical_drag_group import VerticalDragGroup
from prouty.performance.fin_interference_ratio_comp import FinInterferenceRatioComp
from prouty.performance.tail_rotor_gross_thrust_comp import TailRotorGrossThrustComp
from prouty.performance.tail_rotor_fin_power_comp import TailRotorFinPowerComp
from prouty.performance.tail_rotor_fin_interference_group import TailRotorFinInterferenceGroup
from prouty.performance.fuselage_drag_comp import FuselageDragComp
from prouty.performance.nacelle_drag_comp import NacelleDragComp
from prouty.performance.rotor_hub_drag_comp import RotorHubDragComp
from prouty.performance.rotor_shaft_drag_comp import RotorShaftDragComp
from prouty.performance.hub_pylon_interference_comp import HubPylonInterferenceComp
from prouty.performance.landing_gear_drag_comp import LandingGearDragComp
from prouty.performance.stabilizer_drag_comp import StabilizerDragComp
from prouty.performance.rotor_fuselage_interference_comp import RotorFuselageInterferenceComp
from prouty.performance.exhaust_drag_comp import ExhaustDragComp
from prouty.performance.total_parasite_drag_comp import TotalParasiteDragComp
from prouty.performance.parasite_drag_group import ParasiteDragGroup
from prouty.performance.tail_rotor_thrust_comp import TailRotorThrustComp
from prouty.performance.rotor_loading_match_comp import RotorLoadingMatchComp
from prouty.performance.hover_ceiling_balance import HoverCeilingBalance
from prouty.performance.main_rotor_hover_power_comp import MainRotorHoverPowerComp
from prouty.performance.rating_select_comp import RatingSelectComp
from prouty.performance.hover_performance_group import HoverPerformanceGroup
from prouty.performance.climb_induced_velocity_comp import ClimbInducedVelocityComp
from prouty.performance.vertical_climb_power_comp import VerticalClimbPowerComp
from prouty.performance.vertical_climb_balance import VerticalClimbBalance
from prouty.performance.vertical_climb_group import VerticalClimbGroup
from prouty.performance.forward_flight_power_group import ForwardFlightPowerGroup
from prouty.performance.max_speed_balance import MaxSpeedBalance
from prouty.performance.equivalent_rotor_ld_comp import EquivalentRotorLDComp
from prouty.performance.specific_range_comp import SpecificRangeComp
from prouty.performance.best_range_speed_balance import (BestRangeSpeedBalance,
                                                          TangencyProductComp)
from prouty.performance.fuel_flow_slope_comp import FuelFlowSlopeComp
from prouty.performance.cruise_speed_balance import CruiseSpeedBalance, CruiseSpeedComp
from prouty.performance.specific_endurance_comp import (BestEnduranceSpeedBalance,
                                                          SpecificEnduranceComp)
from prouty.performance.mission_integral_comp import MissionIntegralComp
from prouty.performance.payload_range_comp import PayloadRangeComp
from prouty.performance.ferry_reserves_comp import FerryReservesComp
from prouty.performance.cruise_performance_group import CruisePerformanceGroup, cruise_problem
from prouty.performance.climb_flat_plate_comp import ClimbFlatPlateComp
from prouty.performance.forward_climb_balance import ForwardClimbBalance
from prouty.performance.climb_time_distance_comp import ClimbTimeDistanceComp
from prouty.performance.climb_ceiling_balance import ClimbCeilingBalance
from prouty.performance.forward_climb_group import ForwardClimbGroup
from prouty.performance.military_mission_group import (MilitaryMissionGroup,
                                                       MissionSegmentFuelComp,
                                                       MissionSummaryComp, MissionWeightsComp)
