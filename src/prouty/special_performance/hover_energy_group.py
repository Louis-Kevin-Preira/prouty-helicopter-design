"""
HoverEnergyGroup -- G2e/G2f linked to Chapter 4: equivalent hover time and
flare time from the hover chain.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, pp. 361 and 363; Chapter 4 hover performance pp. 308-312
(HoverPerformanceGroup) and the main rotor maximum of Figure 4.28.

    hover       HoverPerformanceGroup     P_req, out of ground effect (engine power)
    weight_coef ExecComp                  C_W/sigma at the normal tip speed
    t_equiv     EquivalentHoverTimeComp   t/k                              p. 363
    flare_time  FlareTimeComp             Delta_t of the flare             p. 361

hp_OGE is the engine power required in hover: after a power failure the
rotor energy also drives the tail rotor and the drive losses (C5-9).
(C_T/sigma)_max is the main rotor maximum of the hover chain, CT_sigma_max_M
(Figure 4.28), shared by both times. J and Omega are inputs (G2a).
The hover inputs are set through hover.* (GW is promoted).
"""

import openmdao.api as om

from prouty.performance import HoverPerformanceGroup
from prouty.special_performance.equivalent_hover_time_comp import EquivalentHoverTimeComp
from prouty.special_performance.flare_time_comp import FlareTimeComp


class HoverEnergyGroup(om.Group):
    """t_equiv and flare Delta_t with hp_OGE and (C_T/sigma)_max from Chapter 4."""

    def initialize(self):
        self.options.declare('hover_options', types=dict, default={'num_segments': 21})

    def setup(self):
        self.add_subsystem('hover', HoverPerformanceGroup(**self.options['hover_options']),
                           promotes_inputs=['GW', ('CT_sigma_max_M', 'CT_sigma_max')])
        self.add_subsystem('weight_coef', om.ExecComp(
            'CW_sigma = GW / (rho * A_b * V_tip**2)',
            GW={'val': 20000.0, 'units': 'lbf'}, rho={'val': 0.002377, 'units': 'slug/ft**3'},
            A_b={'val': 240.0, 'units': 'ft**2'}, V_tip={'val': 650.0, 'units': 'ft/s'}),
            promotes_inputs=['GW', 'rho', 'A_b', 'V_tip'], promotes_outputs=['CW_sigma'])
        self.add_subsystem('t_equiv', EquivalentHoverTimeComp(),
                           promotes_inputs=['J', 'Omega', 'CT_sigma_max'],
                           promotes_outputs=['t_equiv'])
        self.add_subsystem('flare_time', FlareTimeComp(),
                           promotes_inputs=['J', ('Omega_0', 'Omega'), 'CT_sigma_max'],
                           promotes_outputs=['dt'])
        self.connect('CW_sigma', ['t_equiv.CW_sigma', 'flare_time.CW_sigma'])
        self.connect('hover.P_req', ['t_equiv.P_OGE', 'flare_time.P_OGE'])
        self.set_input_defaults('CT_sigma_max', val=0.167)
        self.set_input_defaults('GW', val=20000.0, units='lbf')
