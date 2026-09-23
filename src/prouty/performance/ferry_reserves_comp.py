"""
FerryReservesComp -- reserves and start weights of a ferry mission.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Maximum Ferry Range" pp. 328-330, Figure 4.46.

The maximum range is integrated between the mission start weight, "the takeoff
gross weight minus fuel for warmup and takeoff", and the landing weight, "the
minimum operating weight plus reserves", p. 329. The reserves are two amounts:

    Reserve no. 1 = fuel for 45 minutes at the minimum operating weight
    Reserve no. 2 = (0.1/1.1) [total fuel - (fuel for the first 3 hr
                                             + WUTO + reserve no. 1)]

Reserve no. 2 is the usual 10 % allowance, written on the fuel that is not
already accounted for and taken out of its own gross amount, hence the 1.1.

Takeoff weight is the minimum operating weight, which already carries the
cabin auxiliary tank, plus the external tanks and all the usable fuel. The
external tanks are dropped when empty, which the specific range integral of
MissionIntegralComp sees as a step in the drag area.

Example helicopter, p. 329: minimum operating weight 11,261 lb, usable fuel
15,961 lb in five tanks, external tanks 409 lb each. Reserve no. 1 is 420 lb
at 11,261 lb, 20,000 ft and 120 kt; the first three hours burn 4,400 lb and
WUTO 56 lb, so reserve no. 2 is 1,008 lb and the landing weight 12,689 lb.

    fuel_total, fuel_first_period, fuel_WUTO, FF_reserve, t_reserve,
    reserve_fraction, GW_min_OP, w_tanks --> reserve_1, reserve_2,
                                             reserves, GW_ldng, GW_TO, GW_start
"""

import numpy as np
import openmdao.api as om


class FerryReservesComp(om.ExplicitComponent):
    """The two reserves and the weights they bracket, p. 329."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('fuel_total', val=np.zeros(nn), units='lbf', desc='usable fuel')
        self.add_input('fuel_first_period', val=np.zeros(nn), units='lbf',
                       desc='fuel burnt in the first hours, 3 hr in the book')
        self.add_input('fuel_WUTO', val=0.0, units='lbf', desc='warmup and takeoff fuel')
        self.add_input('FF_reserve', val=0.0, units='lbm/h',
                       desc='fuel flow at the minimum operating weight')
        self.add_input('t_reserve', val=0.75, units='h', desc='reserve time, 45 min in the book')
        self.add_input('reserve_fraction', val=0.10, desc='fraction kept as reserve no. 2')
        self.add_input('GW_min_OP', val=0.0, units='lbf', desc='minimum operating weight')
        self.add_input('w_tanks', val=0.0, units='lbf', desc='droppable tanks, empty')

        for name, desc in (('reserve_1', 'fuel for the reserve time'),
                           ('reserve_2', 'percentage reserve on the rest'),
                           ('reserves', 'reserve no. 1 + no. 2'),
                           ('GW_ldng', 'minimum operating weight plus reserves'),
                           ('GW_TO', 'takeoff gross weight'),
                           ('GW_start', 'takeoff weight less the WUTO fuel')):
            self.add_output(name, val=np.zeros(nn), units='lbf', desc=desc)

        self.declare_partials('reserve_1', ['FF_reserve', 't_reserve'])
        self.declare_partials(['reserve_2', 'reserves', 'GW_ldng'],
                              ['fuel_total', 'fuel_first_period'], rows=ar, cols=ar)
        self.declare_partials(['reserve_2', 'reserves', 'GW_ldng'],
                              ['fuel_WUTO', 'FF_reserve', 't_reserve', 'reserve_fraction'])
        self.declare_partials(['reserves', 'GW_ldng'], ['FF_reserve', 't_reserve'])
        self.declare_partials('GW_ldng', 'GW_min_OP')
        self.declare_partials(['GW_TO', 'GW_start'], 'fuel_total', rows=ar, cols=ar, val=1.0)
        self.declare_partials(['GW_TO', 'GW_start'], ['GW_min_OP', 'w_tanks'])
        self.declare_partials('GW_start', 'fuel_WUTO')

    def _values(self, inputs):
        k = inputs['reserve_fraction'][0]
        reserve_1 = inputs['FF_reserve'][0] * inputs['t_reserve'][0]
        rest = inputs['fuel_total'] - (inputs['fuel_first_period'] + inputs['fuel_WUTO'][0]
                                       + reserve_1)
        reserve_2 = k / (1.0 + k) * rest
        return reserve_1, reserve_2, k

    def compute(self, inputs, outputs):
        reserve_1, reserve_2, _ = self._values(inputs)
        reserves = reserve_1 + reserve_2
        outputs['reserve_1'] = reserve_1
        outputs['reserve_2'] = reserve_2
        outputs['reserves'] = reserves
        outputs['GW_ldng'] = inputs['GW_min_OP'][0] + reserves
        outputs['GW_TO'] = inputs['GW_min_OP'][0] + inputs['w_tanks'][0] + inputs['fuel_total']
        outputs['GW_start'] = outputs['GW_TO'] - inputs['fuel_WUTO'][0]

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        reserve_1, reserve_2, k = self._values(inputs)
        rest = reserve_2 * (1.0 + k) / k if k else np.zeros(nn)
        c = k / (1.0 + k)
        ones, col = np.ones(nn), np.ones((nn, 1))
        FF, t = inputs['FF_reserve'][0], inputs['t_reserve'][0]

        partials['reserve_1', 'FF_reserve'] = t * col
        partials['reserve_1', 't_reserve'] = FF * col

        for name, val in (('fuel_total', c * ones), ('fuel_first_period', -c * ones)):
            partials['reserve_2', name] = val
            partials['reserves', name] = val
            partials['GW_ldng', name] = val
        for name, val in (('fuel_WUTO', -c * col), ('FF_reserve', -c * t * col),
                          ('t_reserve', -c * FF * col)):
            partials['reserve_2', name] = val
            partials['reserves', name] = val + (t * col if name == 'FF_reserve' else
                                                (FF * col if name == 't_reserve' else 0.0))
            partials['GW_ldng', name] = partials['reserves', name]
        d_frac = (rest / (1.0 + k) ** 2).reshape(-1, 1)
        partials['reserve_2', 'reserve_fraction'] = d_frac
        partials['reserves', 'reserve_fraction'] = d_frac
        partials['GW_ldng', 'reserve_fraction'] = d_frac
        partials['GW_ldng', 'GW_min_OP'] = col

        partials['GW_TO', 'GW_min_OP'] = col
        partials['GW_TO', 'w_tanks'] = col
        partials['GW_start', 'GW_min_OP'] = col
        partials['GW_start', 'w_tanks'] = col
        partials['GW_start', 'fuel_WUTO'] = -col
