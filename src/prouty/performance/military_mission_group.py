"""
MilitaryMissionGroup -- G9, analysis of a military-type mission.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Analysis of the Military-Type Mission" pp. 336-337, Table 4.4.

"Since the fuel required for the mission is not known, the takeoff gross
weight cannot be initially determined. The landing weight, however, can be
closely estimated since it is simply the minimum operating weight plus a small
fuel reserve. For this reason, it is convenient to analyse the mission in
reverse as on Table 4.4."

    segments   MissionSegmentFuelComp   fuel of each segment
    weights    MissionWeightsComp       gross weight at each segment boundary,
                                        sweeping backwards from the landing weight
    summary    MissionSummaryComp       mission fuel, total fuel, reserve, T.O.G.W.

Each segment burns its fuel in one of three ways, which is what the columns of
Table 4.4 hold:

    'time'       fuel = fuel flow * time                  warmup, climb, dash, hover
    'distance'   fuel = distance / specific range         cruise legs
    'endurance'  fuel = time / specific endurance         loiter

Payload dropped at the end of a segment reappears in the backward sweep: the
helicopter was heavier before it, which is the 6,600 lb step at the hover of
segment 6.

The reserve closes the loop at the end, p. 337: the mission fuel is what the
segments burn, the total fuel is mission fuel / (1 - reserve fraction), and
the takeoff weight is the minimum operating weight plus payload plus total
fuel. The landing weight the sweep started from used an estimated reserve, so
the two takeoff weights differ by that estimate error: 19,705 lb estimated
against 19,669 lb computed in the book, "36 lb too high due to initial reserve
estimate".

    fuel flows, times, distances, specific ranges, payload drops
        --> GW at each boundary, mission fuel, total fuel, reserve, T.O.G.W.
"""

import numpy as np
import openmdao.api as om

MODES = ('time', 'distance', 'endurance')


class MissionSegmentFuelComp(om.ExplicitComponent):
    """Fuel burnt by each segment, Table 4.4."""

    def initialize(self):
        self.options.declare('modes', types=(list, tuple), default=('time',),
                             desc='one of time, distance or endurance per segment')

    def setup(self):
        modes = tuple(self.options['modes'])
        if any(m not in MODES for m in modes):
            raise ValueError(f'each segment mode must be one of {MODES}')
        self._modes = modes
        n = len(modes)
        ar = np.arange(n)

        self.add_input('FF', val=np.zeros(n), units='lbm/h', desc='fuel flow, all engines')
        self.add_input('time', val=np.zeros(n), units='h', desc='segment time')
        self.add_input('distance', val=np.zeros(n), units='NM', desc='segment distance')
        self.add_input('SR', val=np.ones(n), units='NM/lbm', desc='specific range')
        self.add_input('SE', val=np.ones(n), units='h/lbm', desc='specific endurance')
        self.add_output('fuel', val=np.zeros(n), units='lbf', desc='fuel for each segment')

        self.declare_partials('fuel', ['FF', 'time', 'distance', 'SR', 'SE'],
                              rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        fuel = np.zeros_like(inputs['FF'])
        for i, mode in enumerate(self._modes):
            if mode == 'time':
                fuel[i] = inputs['FF'][i] * inputs['time'][i]
            elif mode == 'distance':
                fuel[i] = inputs['distance'][i] / inputs['SR'][i]
            else:
                fuel[i] = inputs['time'][i] / inputs['SE'][i]
        outputs['fuel'] = fuel

    def compute_partials(self, inputs, partials):
        for name in ('FF', 'time', 'distance', 'SR', 'SE'):
            partials['fuel', name] = np.zeros_like(inputs['FF'])
        for i, mode in enumerate(self._modes):
            if mode == 'time':
                partials['fuel', 'FF'][i] = inputs['time'][i]
                partials['fuel', 'time'][i] = inputs['FF'][i]
            elif mode == 'distance':
                partials['fuel', 'distance'][i] = 1.0 / inputs['SR'][i]
                partials['fuel', 'SR'][i] = -inputs['distance'][i] / inputs['SR'][i] ** 2
            else:
                partials['fuel', 'time'][i] = 1.0 / inputs['SE'][i]
                partials['fuel', 'SE'][i] = -inputs['time'][i] / inputs['SE'][i] ** 2


class MissionWeightsComp(om.ExplicitComponent):
    """Backward sweep of the gross weights, Table 4.4."""

    def initialize(self):
        self.options.declare('num_segments', types=int, default=1,
                             desc='segments in flight order, 1 to n')

    def setup(self):
        n = self.options['num_segments']

        self.add_input('GW_ldng', val=0.0, units='lbf',
                       desc='landing weight, minimum operating weight plus estimated reserve')
        self.add_input('fuel', val=np.zeros(n), units='lbf', desc='fuel of each segment')
        self.add_input('payload_drop', val=np.zeros(n), units='lbf',
                       desc='payload left behind at the end of each segment')

        self.add_output('GW_end', val=np.zeros(n), units='lbf',
                        desc='gross weight at the end of each segment')
        self.add_output('GW_avg', val=np.zeros(n), units='lbf',
                        desc='average gross weight of each segment')
        self.add_output('GW_TO_est', val=0.0, units='lbf',
                        desc='takeoff weight from the sweep')

        self.declare_partials('*', ['GW_ldng', 'fuel', 'payload_drop'])

    def compute(self, inputs, outputs):
        fuel, drop = inputs['fuel'], inputs['payload_drop']
        n = fuel.size
        GW_end = np.zeros_like(fuel)            # complex safe under complex step
        running = inputs['GW_ldng'][0]
        for i in range(n - 1, -1, -1):          # last segment first
            GW_end[i] = running + drop[i]       # the end weight still carries the payload
            running = GW_end[i] + fuel[i]
        outputs['GW_end'] = GW_end
        outputs['GW_avg'] = GW_end + 0.5 * fuel
        outputs['GW_TO_est'] = running

    def compute_partials(self, inputs, partials):
        n = self.options['num_segments']
        later = np.triu(np.ones((n, n)), 1)     # end weight sees the segments after it
        with_own_drop = later + np.eye(n)       # and the payload dropped at its own end

        partials['GW_end', 'GW_ldng'] = np.ones((n, 1))
        partials['GW_end', 'fuel'] = later
        partials['GW_end', 'payload_drop'] = with_own_drop
        partials['GW_avg', 'GW_ldng'] = np.ones((n, 1))
        partials['GW_avg', 'fuel'] = later + 0.5 * np.eye(n)
        partials['GW_avg', 'payload_drop'] = with_own_drop
        partials['GW_TO_est', 'GW_ldng'] = 1.0
        partials['GW_TO_est', 'fuel'] = np.ones((1, n))
        partials['GW_TO_est', 'payload_drop'] = np.ones((1, n))


class MissionSummaryComp(om.ExplicitComponent):
    """Mission fuel, reserve and takeoff weight, p. 337."""

    def initialize(self):
        self.options.declare('num_segments', types=int, default=1)

    def setup(self):
        n = self.options['num_segments']

        self.add_input('fuel', val=np.zeros(n), units='lbf')
        self.add_input('reserve_fraction', val=0.10, desc='reserve as a fraction of total fuel')
        self.add_input('GW_min_OP', val=0.0, units='lbf')
        self.add_input('payload', val=0.0, units='lbf')

        self.add_output('fuel_mission', val=0.0, units='lbf')
        self.add_output('fuel_total', val=0.0, units='lbf')
        self.add_output('reserve', val=0.0, units='lbf')
        self.add_output('GW_TO', val=0.0, units='lbf')

        self.declare_partials(['fuel_mission', 'fuel_total', 'reserve', 'GW_TO'], 'fuel')
        self.declare_partials(['fuel_total', 'reserve', 'GW_TO'], 'reserve_fraction')
        self.declare_partials('GW_TO', ['GW_min_OP', 'payload'])

    def compute(self, inputs, outputs):
        mission = np.sum(inputs['fuel'])
        total = mission / (1.0 - inputs['reserve_fraction'][0])
        outputs['fuel_mission'] = mission
        outputs['fuel_total'] = total
        outputs['reserve'] = total - mission
        outputs['GW_TO'] = inputs['GW_min_OP'][0] + inputs['payload'][0] + total

    def compute_partials(self, inputs, partials):
        n = self.options['num_segments']
        k = inputs['reserve_fraction'][0]
        mission = np.sum(inputs['fuel'])
        ones = np.ones((1, n))

        partials['fuel_mission', 'fuel'] = ones
        partials['fuel_total', 'fuel'] = ones / (1.0 - k)
        partials['reserve', 'fuel'] = ones * (1.0 / (1.0 - k) - 1.0)
        partials['GW_TO', 'fuel'] = ones / (1.0 - k)
        d_total = mission / (1.0 - k) ** 2
        partials['fuel_total', 'reserve_fraction'] = d_total
        partials['reserve', 'reserve_fraction'] = d_total
        partials['GW_TO', 'reserve_fraction'] = d_total
        partials['GW_TO', 'GW_min_OP'] = 1.0
        partials['GW_TO', 'payload'] = 1.0


class MilitaryMissionGroup(om.Group):
    """Reverse analysis of a military mission, Table 4.4."""

    def initialize(self):
        self.options.declare('modes', types=(list, tuple), default=('time',))

    def setup(self):
        modes = tuple(self.options['modes'])
        n = len(modes)
        self.add_subsystem('segments', MissionSegmentFuelComp(modes=modes), promotes=['*'])
        self.add_subsystem('weights', MissionWeightsComp(num_segments=n), promotes=['*'])
        self.add_subsystem('summary', MissionSummaryComp(num_segments=n), promotes=['*'])
