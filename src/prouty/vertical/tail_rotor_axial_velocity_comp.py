"""
TailRotorAxialVelocityComp -- axial "descent" velocity of the tail rotor.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "The Tail Rotor and the Vortex Ring State" pp. 107-109, p. 116.

A tail rotor meets the vortex ring state in sideward flight or in a hover turn
(p. 107): moving in the direction of its own thrust, it sees an axial flow
against its induced velocity, the analogue of the main rotor in vertical
descent (p. 116). With V_y_T the sideward speed and r_yaw the yaw rate, both
positive when they carry the tail rotor in the direction of its thrust:

    V_D_T     = V_y_T + r_yaw l_T
    V_D_bar_T = V_D_T / v_1hov_T

p. 108: the UH-1 tail rotor (v_1hov_T = 48 ft/s) should meet the maximum vortex
ring effect at 70 % of it, about 20 kt; in flight it comes earlier, the main
rotor wake interacting with the tail rotor.

    V_y_T, r_yaw, v_hov_T (nn,), l_T --> V_D_T, V_D_bar_T (nn,)
"""

import numpy as np
import openmdao.api as om


class TailRotorAxialVelocityComp(om.ExplicitComponent):
    """Axial velocity through the tail rotor in sideward flight and hover turns, pp. 107-108."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('V_y_T', val=np.zeros(nn), units='ft/s',
                       desc='sideward speed, > 0 in the tail rotor thrust direction')
        self.add_input('r_yaw', val=np.zeros(nn), units='rad/s',
                       desc='yaw rate, > 0 when it moves the tail rotor along its thrust')
        self.add_input('v_hov_T', val=np.ones(nn), units='ft/s',
                       desc='tail rotor hover induced velocity')
        self.add_input('l_T', val=1.0, units='ft', desc='main to tail rotor distance')
        self.add_output('V_D_T', val=np.zeros(nn), units='ft/s', desc='axial descent velocity')
        self.add_output('V_D_bar_T', val=np.zeros(nn), desc='V_D_T / v_1hov_T')
        self.declare_partials(['V_D_T', 'V_D_bar_T'], ['V_y_T', 'r_yaw'], rows=ar, cols=ar)
        self.declare_partials('V_D_bar_T', 'v_hov_T', rows=ar, cols=ar)
        self.declare_partials(['V_D_T', 'V_D_bar_T'], 'l_T')

    def compute(self, inputs, outputs):
        V = inputs['V_y_T'] + inputs['r_yaw'] * inputs['l_T'][0]
        outputs['V_D_T'] = V
        outputs['V_D_bar_T'] = V / inputs['v_hov_T']

    def compute_partials(self, inputs, partials):
        l, vh, r = inputs['l_T'][0], inputs['v_hov_T'], inputs['r_yaw']
        V = inputs['V_y_T'] + r * l
        partials['V_D_T', 'V_y_T'] = np.ones_like(vh)
        partials['V_D_T', 'r_yaw'] = np.full_like(vh, l)
        partials['V_D_T', 'l_T'] = r.reshape(-1, 1)
        partials['V_D_bar_T', 'V_y_T'] = 1.0 / vh
        partials['V_D_bar_T', 'r_yaw'] = l / vh
        partials['V_D_bar_T', 'l_T'] = (r / vh).reshape(-1, 1)
        partials['V_D_bar_T', 'v_hov_T'] = -V / vh ** 2
