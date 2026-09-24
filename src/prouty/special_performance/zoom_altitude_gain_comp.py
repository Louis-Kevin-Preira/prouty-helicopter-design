"""
ZoomAltitudeGainComp -- G2c, altitude gained by zooming after a power failure.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Glide Distance" p. 352, Figure 5.6 p. 353.

Kinetic energy from V_0 down to V_1, less the average level flight power
dissipated while gravity decelerates the helicopter along the climb path:

    Delta_h = (V_0^2 - V_1^2)/(2g)
              - 550 (hp_0 + hp_1)/(2 G.W.) (V_0 - V_1)/(g sin gamma_c)

Rotor kinetic energy is deliberately excluded (p. 352).
hp_0, hp_1: level flight power at V_0 and V_1 (Chapter 4 chain).

    V_0, P_0, V_1 (nn,), P_1 (nn,), GW, gamma_c --> delta_h (nn,)
"""

import numpy as np
import openmdao.api as om

G = 32.2                    # ft/s^2, as printed in Chapter 5
HP_TO_FT_LBF_PER_S = 550.0


class ZoomAltitudeGainComp(om.ExplicitComponent):
    """Zoom altitude gain for autorotation speeds V_1 <= V_0, p. 352."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1,
                             desc='number of autorotation speeds V_1')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('V_0', val=270.0, units='ft/s')
        self.add_input('P_0', val=4000.0, units='hp')
        self.add_input('V_1', val=150.0 * np.ones(nn), units='ft/s')
        self.add_input('P_1', val=1200.0 * np.ones(nn), units='hp')
        self.add_input('GW', val=20000.0, units='lbf')
        self.add_input('gamma_c', val=0.8, units='rad')
        self.add_output('delta_h', val=np.zeros(nn), units='ft')
        self.declare_partials('delta_h', ['V_1', 'P_1'], rows=ar, cols=ar)
        self.declare_partials('delta_h', ['V_0', 'P_0', 'GW', 'gamma_c'], rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        V0, V1 = inputs['V_0'], inputs['V_1']
        k = HP_TO_FT_LBF_PER_S / (2.0 * inputs['GW'] * G * np.sin(inputs['gamma_c']))
        outputs['delta_h'] = ((V0 ** 2 - V1 ** 2) / (2.0 * G)
                              - k * (inputs['P_0'] + inputs['P_1']) * (V0 - V1))

    def compute_partials(self, inputs, J):
        V0, V1, W, gc = inputs['V_0'], inputs['V_1'], inputs['GW'], inputs['gamma_c']
        P = inputs['P_0'] + inputs['P_1']
        k = HP_TO_FT_LBF_PER_S / (2.0 * W * G * np.sin(gc))
        loss = k * P * (V0 - V1)
        J['delta_h', 'V_0'] = V0 / G - k * P
        J['delta_h', 'V_1'] = -V1 / G + k * P
        J['delta_h', 'P_0'] = -k * (V0 - V1)
        J['delta_h', 'P_1'] = -k * (V0 - V1)
        J['delta_h', 'GW'] = loss / W
        J['delta_h', 'gamma_c'] = loss / np.tan(gc)
