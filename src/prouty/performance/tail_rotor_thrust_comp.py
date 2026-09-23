"""
TailRotorThrustComp -- net tail rotor thrust that balances the main rotor torque.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Hover Performance" p. 309.

The tail rotor balances the main rotor torque about the shaft, with the moment
arm l_T between the two shafts:

    T_T_req = 550 h.p._M R_M / [(Omega R)_M l_T]

Example helicopter, p. 309: T_T_req = 0.69 h.p._M, i.e. R_M = 30 ft,
(Omega R)_M = 650 ft/s and l_T = 36.8 ft.

T_req feeds TailRotorGrossThrustComp (G3), which adds the fin interference.

    P_MR (nn,), R_M, V_tip_M, l_T --> T_req (nn,)
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class TailRotorThrustComp(om.ExplicitComponent):
    """Yaw balance in hover, p. 309."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('P_MR', val=np.zeros(nn), units='hp', desc='main rotor power')
        self.add_input('R_M', val=1.0, units='ft', desc='main rotor radius')
        self.add_input('V_tip_M', val=1.0, units='ft/s', desc='main rotor tip speed')
        self.add_input('l_T', val=1.0, units='ft', desc='main to tail rotor shaft distance')
        self.add_output('T_req', val=np.zeros(nn), units='lbf', desc='net tail rotor thrust')

        self.declare_partials('T_req', 'P_MR', rows=ar, cols=ar)
        self.declare_partials('T_req', ['R_M', 'V_tip_M', 'l_T'])

    def compute(self, inputs, outputs):
        outputs['T_req'] = (HP_TO_FT_LBF_PER_S * inputs['P_MR'] * inputs['R_M'][0]
                            / (inputs['V_tip_M'][0] * inputs['l_T'][0]))

    def compute_partials(self, inputs, partials):
        P, R, V, l = (inputs[k] for k in ('P_MR', 'R_M', 'V_tip_M', 'l_T'))
        R, V, l = R[0], V[0], l[0]
        T = HP_TO_FT_LBF_PER_S * P * R / (V * l)

        partials['T_req', 'P_MR'] = np.full(P.size, HP_TO_FT_LBF_PER_S * R / (V * l))
        partials['T_req', 'R_M'] = (T / R).reshape(-1, 1)
        partials['T_req', 'V_tip_M'] = (-T / V).reshape(-1, 1)
        partials['T_req', 'l_T'] = (-T / l).reshape(-1, 1)
