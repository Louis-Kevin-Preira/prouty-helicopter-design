"""
ClimbFlatPlateComp -- flat plate area that carries the weight up the flight path.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Climb in Forward Flight" pp. 332-334, Figures 4.48 and 4.49.

"The power required to climb at a given flight path angle, gamma, can be
determined by using the charts of Chapter 3 just as for level flight but
modifying the flat plate area to account for the rearward component of gross
weight along the flight path", p. 333:

    f_climb = f + G.W. sin(gamma) / q

so a climb costs exactly what the same drag increment would cost in level
flight, and the whole forward flight chain is reused unchanged.

The angle follows from the rate of climb and the speed:

    angle_convention = 'path'        sin(gamma) = V_c / V     V along the path
                     = 'horizontal'  tan(gamma) = V_c / V     V horizontal

p. 333 reads the one engine case as "1,100 ft/min at 48 knots, which gives a
climb angle of 12.7 degrees": that is the arc tangent, so 'horizontal' is the
book's own convention here (C4-34), and it is the default. The difference
reaches 0.4 deg at that point and grows with the angle.

    f, GW, q, V, V_c (nn,) --> gamma, sin_gamma, f_climb (nn,)
"""

import numpy as np
import openmdao.api as om


class ClimbFlatPlateComp(om.ExplicitComponent):
    """Equivalent flat plate area of a climb, p. 333."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('angle_convention', default='horizontal',
                             values=('horizontal', 'path'))

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('f', val=np.zeros(nn), units='ft**2', desc='level flight flat plate area')
        self.add_input('GW', val=np.zeros(nn), units='lbf', desc='gross weight')
        self.add_input('q', val=np.ones(nn), units='lbf/ft**2', desc='dynamic pressure')
        self.add_input('V', val=np.ones(nn), units='ft/s', desc='flight speed')
        self.add_input('V_c', val=np.zeros(nn), units='ft/s', desc='rate of climb')

        self.add_output('gamma', val=np.zeros(nn), units='rad', desc='flight path angle')
        self.add_output('sin_gamma', val=np.zeros(nn), desc='sine of the flight path angle')
        self.add_output('f_climb', val=np.zeros(nn), units='ft**2',
                        desc='flat plate area to use in the level flight chain')

        self.declare_partials(['gamma', 'sin_gamma', 'f_climb'], ['V', 'V_c'], rows=ar, cols=ar)
        self.declare_partials('f_climb', ['f', 'GW', 'q'], rows=ar, cols=ar)

    def _angle(self, inputs):
        """gamma, dgamma/dV_c, dgamma/dV."""
        ratio = inputs['V_c'] / inputs['V']
        if self.options['angle_convention'] == 'path':
            gamma = np.arcsin(ratio)
            d = 1.0 / np.sqrt(1.0 - ratio ** 2)
        else:
            gamma = np.arctan(ratio)
            d = 1.0 / (1.0 + ratio ** 2)
        return gamma, d / inputs['V'], -d * ratio / inputs['V']

    def compute(self, inputs, outputs):
        gamma = self._angle(inputs)[0]
        sin_gamma = np.sin(gamma)
        outputs['gamma'] = gamma
        outputs['sin_gamma'] = sin_gamma
        outputs['f_climb'] = inputs['f'] + inputs['GW'] * sin_gamma / inputs['q']

    def compute_partials(self, inputs, partials):
        GW, q = inputs['GW'], inputs['q']
        gamma, dg_dVc, dg_dV = self._angle(inputs)
        sin_gamma, cos_gamma = np.sin(gamma), np.cos(gamma)

        partials['gamma', 'V_c'], partials['gamma', 'V'] = dg_dVc, dg_dV
        partials['sin_gamma', 'V_c'] = cos_gamma * dg_dVc
        partials['sin_gamma', 'V'] = cos_gamma * dg_dV
        partials['f_climb', 'V_c'] = GW / q * cos_gamma * dg_dVc
        partials['f_climb', 'V'] = GW / q * cos_gamma * dg_dV
        partials['f_climb', 'f'] = np.ones_like(GW)
        partials['f_climb', 'GW'] = sin_gamma / q
        partials['f_climb', 'q'] = -GW * sin_gamma / q ** 2
