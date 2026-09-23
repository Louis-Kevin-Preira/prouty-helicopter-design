"""
ParasiteDragGroup -- G4, parasite drag in forward flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Parasite Drag in Forward Flight" pp. 287-308; the group follows
the estimating procedure of pp. 306-308 item by item.

    fuselage        FuselageDragComp               f_F     Fig. 4.17
    nacelles        NacelleDragComp                f_N     Fig. 4.19
    hub_main        RotorHubDragComp               f_hub_M Table 4.2, Fig. 4.22
    shaft           RotorShaftDragComp             f_s     Fig. 4.23
    hub_pylon       HubPylonInterferenceComp       f_M     Fig. 4.24
    hub_tail        RotorHubDragComp               f_T     Table 4.2, Fig. 4.22
    gear_main       LandingGearDragComp            f_MLG   Fig. 4.26
    gear_nose       LandingGearDragComp            f_NLG   Fig. 4.26
    stab_h, stab_v  StabilizerDragComp             f_H, f_V Figs 4.12, 4.21
    interference    RotorFuselageInterferenceComp  f_int   Fig. 4.25
    exhaust         ExhaustDragComp                f_ex    p. 304
    total           TotalParasiteDragComp          f_total, f_design

No loop: everything is explicit, at one reference speed. rho and T_air come
from an AtmosphereGroup placed upstream; V is the reference speed of the drag
estimate (115 kt in the book's example).

Names. Each item keeps the inputs of its component, prefixed where two
instances share them: A_hub_M and A_hub_T, C_D0_M and C_D0_T, A_MLG and
A_NLG, A_H and A_V, and so on. alpha_F feeds the hub-pylon factor; the
rotor-fuselage interference has its own alpha_F_int, zero by default, because
the procedure reads Figure 4.25 at zero angle of attack (p. 308) while it
reads Figure 4.24 at the trimmed angle. Connect the two to follow the angle.

    geometry, rho, T_air, V --> f_F ... f_ex, f_total, f_design
"""

import openmdao.api as om

from prouty.performance.exhaust_drag_comp import ExhaustDragComp
from prouty.performance.fuselage_drag_comp import REFERENCE_AIRCRAFT, FuselageDragComp
from prouty.performance.hub_pylon_interference_comp import HubPylonInterferenceComp
from prouty.performance.landing_gear_drag_comp import LandingGearDragComp
from prouty.performance.nacelle_drag_comp import NacelleDragComp
from prouty.performance.rotor_fuselage_interference_comp import RotorFuselageInterferenceComp
from prouty.performance.rotor_hub_drag_comp import RotorHubDragComp
from prouty.performance.rotor_shaft_drag_comp import RotorShaftDragComp
from prouty.performance.stabilizer_drag_comp import StabilizerDragComp
from prouty.performance.total_parasite_drag_comp import TotalParasiteDragComp

ITEMS = ('F', 'N', 'M', 'T', 'MLG', 'NLG', 'H', 'V', 'int', 'ex')


class ParasiteDragGroup(om.Group):
    """Equivalent flat plate area of the whole helicopter, procedure pp. 306-308."""

    def initialize(self):
        self.options.declare('reference_aircraft', default='L-286',
                             values=tuple(REFERENCE_AIRCRAFT), desc='cleanliness level, Fig. 4.17')
        self.options.declare('num_nacelles', types=int, default=2)
        self.options.declare('hub_fairing', default='unfaired', values=('unfaired', 'faired'))
        self.options.declare('tail_hub_fairing', default='unfaired', values=('unfaired', 'faired'))
        self.options.declare('nose_gear', types=bool, default=True,
                             desc='nose or tail wheel read on the e/d curve of Fig. 4.26')
        self.options.declare('nose_gear_strut', default='round', values=('round', 'faired'))
        self.options.declare('exhaust_mode', default='thrust', values=('thrust', 'momentum'))
        self.options.declare('transition', default='turbulent', values=('turbulent', 'natural'))

    def setup(self):
        opt = self.options

        self.add_subsystem('fuselage', FuselageDragComp(reference=opt['reference_aircraft']),
                           promotes=['*'])
        self.add_subsystem('nacelles', NacelleDragComp(num_nacelles=opt['num_nacelles']),
                           promotes=['*'])

        self.add_subsystem('hub_main', RotorHubDragComp(fairing=opt['hub_fairing']),
                           promotes_inputs=[('A_hub', 'A_hub_M'), ('C_D0', 'C_D0_M'),
                                            'alpha_s', 'rpm_pct'],
                           promotes_outputs=[('f_hub', 'f_hub_M'), ('C_D', 'C_D_hub_M'),
                                             ('DR', 'DR_M')])
        self.add_subsystem('shaft', RotorShaftDragComp(),
                           promotes_inputs=['rho', 'T_air', 'V', 'D_s', 'l_s'],
                           promotes_outputs=[('f_s', 'f_s'), ('RN', 'RN_s'), ('C_D', 'C_D_s')])
        self.add_subsystem('hub_pylon', HubPylonInterferenceComp(),
                           promotes_inputs=['Z', 'W_p', 'alpha_F', ('f_hub', 'f_hub_M'),
                                            ('f_shaft', 'f_s')],
                           promotes_outputs=['K_i', 'Z_Wp', ('f_M', 'f_M')])
        self.add_subsystem('hub_tail', RotorHubDragComp(fairing=opt['tail_hub_fairing']),
                           promotes_inputs=[('A_hub', 'A_hub_T'), ('C_D0', 'C_D0_T'),
                                            ('alpha_s', 'alpha_s_T'), ('rpm_pct', 'rpm_pct_T')],
                           promotes_outputs=[('f_hub', 'f_T'), ('C_D', 'C_D_hub_T'),
                                             ('DR', 'DR_T')])

        self.add_subsystem('gear_main', LandingGearDragComp(mode='catalog'),
                           promotes_inputs=[('A_gear', 'A_MLG'), ('C_D', 'C_D_MLG')],
                           promotes_outputs=[('f_gear', 'f_MLG')])
        nose_mode = 'nose_wheel' if opt['nose_gear'] else 'catalog'
        nose_inputs = [('A_gear', 'A_NLG')] + ([('e', 'e_NLG'), ('d', 'd_NLG')]
                                               if opt['nose_gear'] else [('C_D', 'C_D_NLG')])
        self.add_subsystem('gear_nose', LandingGearDragComp(mode=nose_mode,
                                                            strut=opt['nose_gear_strut']),
                           promotes_inputs=nose_inputs,
                           promotes_outputs=[('f_gear', 'f_NLG')]
                           + ([('C_D', 'C_D_NLG')] if opt['nose_gear'] else []))

        for tag, junctions in (('H', 2), ('V', 0)):
            self.add_subsystem(f'stab_{tag.lower()}',
                               StabilizerDragComp(num_junctions=junctions,
                                                  transition=opt['transition']),
                               promotes_inputs=['rho', 'T_air', 'V', 'delta',
                                                ('A', f'A_{tag}'), ('b', f'b_{tag}'),
                                                ('MAC', f'MAC_{tag}'), ('t_c', f't_c_{tag}'),
                                                ('C_L', f'C_L_{tag}'),
                                                ('q_ratio', f'q_ratio_{tag}')],
                               promotes_outputs=[('f_stab', f'f_{tag}'), ('RN', f'RN_{tag}'),
                                                 ('C_D', f'C_D_{tag}'), ('C_F', f'C_F_{tag}')])

        self.add_subsystem('interference', RotorFuselageInterferenceComp(),
                           promotes_inputs=['A_F', ('alpha_F', 'alpha_F_int')],
                           promotes_outputs=['dC_D', 'f_int'])
        self.add_subsystem('exhaust', ExhaustDragComp(mode=opt['exhaust_mode']), promotes=['*'])
        self.add_subsystem('total', TotalParasiteDragComp(items=ITEMS), promotes=['*'])
