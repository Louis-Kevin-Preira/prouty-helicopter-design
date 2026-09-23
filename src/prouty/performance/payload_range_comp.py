"""
PayloadRangeComp -- weights behind a payload-range point.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Payload-Range" pp. 326-328, Figure 4.45.

The payload-range curve rests on two equations, p. 326. The range is the area
under the specific range curve between the landing and takeoff weights
(MissionIntegralComp); this component carries the weights:

    G.W._ldng = G.W._T.O. - (expended fuel + WUTO fuel)
    Payload   = (G.W._T.O. - G.W._min.OP)
                - (expended fuel + WUTO fuel + reserves + aux. fuel tank wt.)

WUTO is warmup and takeoff. The auxiliary tank weight appears only when fuel
is carried beyond the internal tanks, and p. 328 takes it as 10 % of the fuel
in those tanks; here it is an input, so a cabin tank or an external tank can be
weighed as the study requires.

Example helicopter, p. 326: taking off at 20,000 lb it carries its design
payload of 6,600 lb, thirty passengers and baggage at 220 lb each, for 330
n.mi.; its minimum operating weight is 11,261 lb (p. 329).

    GW_TO, fuel_expended, fuel_WUTO, fuel_reserves, w_aux_tank, GW_min_OP
        --> GW_ldng, payload, fuel_total (nn,)
"""

import numpy as np
import openmdao.api as om


class PayloadRangeComp(om.ExplicitComponent):
    """Landing weight and payload of a payload-range point, p. 326."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('GW_TO', val=np.zeros(nn), units='lbf', desc='takeoff gross weight')
        self.add_input('fuel_expended', val=np.zeros(nn), units='lbf',
                       desc='fuel burnt in the cruise legs')
        self.add_input('fuel_WUTO', val=0.0, units='lbf', desc='warmup and takeoff fuel')
        self.add_input('fuel_reserves', val=0.0, units='lbf', desc='fuel kept in reserve')
        self.add_input('w_aux_tank', val=0.0, units='lbf', desc='auxiliary tank weight')
        self.add_input('GW_min_OP', val=0.0, units='lbf', desc='minimum operating weight')

        self.add_output('GW_ldng', val=np.zeros(nn), units='lbf', desc='landing gross weight')
        self.add_output('payload', val=np.zeros(nn), units='lbf')
        self.add_output('fuel_total', val=np.zeros(nn), units='lbf',
                        desc='expended + WUTO + reserves')

        self.declare_partials('GW_ldng', ['GW_TO', 'fuel_expended'], rows=ar, cols=ar)
        self.declare_partials('GW_ldng', 'fuel_WUTO')
        self.declare_partials('payload', ['GW_TO', 'fuel_expended'], rows=ar, cols=ar)
        self.declare_partials('payload', ['fuel_WUTO', 'fuel_reserves', 'w_aux_tank',
                                          'GW_min_OP'])
        self.declare_partials('fuel_total', 'fuel_expended', rows=ar, cols=ar, val=1.0)
        self.declare_partials('fuel_total', ['fuel_WUTO', 'fuel_reserves'])

    def compute(self, inputs, outputs):
        burnt = inputs['fuel_expended'] + inputs['fuel_WUTO'][0]
        outputs['GW_ldng'] = inputs['GW_TO'] - burnt
        outputs['fuel_total'] = burnt + inputs['fuel_reserves'][0]
        outputs['payload'] = ((inputs['GW_TO'] - inputs['GW_min_OP'][0])
                              - (burnt + inputs['fuel_reserves'][0] + inputs['w_aux_tank'][0]))

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        ones, col = np.ones(nn), np.ones((nn, 1))

        partials['GW_ldng', 'GW_TO'] = ones
        partials['GW_ldng', 'fuel_expended'] = -ones
        partials['GW_ldng', 'fuel_WUTO'] = -col

        partials['payload', 'GW_TO'] = ones
        partials['payload', 'fuel_expended'] = -ones
        for name in ('fuel_WUTO', 'fuel_reserves', 'w_aux_tank', 'GW_min_OP'):
            partials['payload', name] = -col

        partials['fuel_total', 'fuel_WUTO'] = col
        partials['fuel_total', 'fuel_reserves'] = col
