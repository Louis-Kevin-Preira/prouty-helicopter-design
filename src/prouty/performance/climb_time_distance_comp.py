"""
ClimbTimeDistanceComp -- time and distance to climb to an altitude.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Time and Distance to Climb" p. 334, Figures 4.49 and 4.50.

"The minimum time to climb to a given altitude can be calculated using the
maximum rates of climb on Figure 4.49 to determine the time required to climb
in 1,000-ft increments. The summation of this process is shown on Figure 4.50.
Knowing the speed corresponding to the climb rates makes it possible to find
the distance required to climb."

    time     = int dh / (R/C)
    distance = int V dh / (R/C)          V the speed flown at each altitude

Both are cumulative, so the outputs are the running sums at each altitude
node, not only the totals: Figure 4.50 plots them against altitude, and a
mission analysis needs the leg values. The book sums 1,000 ft increments,
which is the trapezoidal rule on a regular grid; the nodes here may be
anywhere and may be design variables, so their derivatives are carried.

Example helicopter, Figure 4.50 at intermediate power on a standard day:
about 10 min and 14 n.mi. to reach 20,000 ft at 20,000 lb.

    altitude, V_c, V (nn,) --> time, distance (nn,), time_total, distance_total
"""

import numpy as np
import openmdao.api as om


class ClimbTimeDistanceComp(om.ExplicitComponent):
    """Cumulative time and distance of a climb, p. 334."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=2)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('altitude', val=np.linspace(0.0, 1.0, nn), units='ft',
                       desc='climb altitudes, increasing')
        self.add_input('V_c', val=np.ones(nn), units='ft/min', desc='rate of climb at each')
        self.add_input('V', val=np.ones(nn), units='kn', desc='speed flown at each')

        self.add_output('time', val=np.zeros(nn), units='min', desc='time from the first node')
        self.add_output('distance', val=np.zeros(nn), units='NM', desc='ground distance')
        self.add_output('time_total', val=0.0, units='min')
        self.add_output('distance_total', val=0.0, units='NM')

        self.declare_partials(['time', 'distance', 'time_total', 'distance_total'],
                              ['altitude', 'V_c'])
        self.declare_partials(['distance', 'distance_total'], 'V')

    def _integrands(self, inputs):
        """dt/dh in min/ft and dx/dh in n.mi./ft at each node."""
        dt_dh = 1.0 / inputs['V_c']
        dx_dh = inputs['V'] / (60.0 * inputs['V_c'])          # kn * min/ft = n.mi./ft / 60
        return dt_dh, dx_dh

    def compute(self, inputs, outputs):
        h = inputs['altitude']
        dt_dh, dx_dh = self._integrands(inputs)
        dh = np.diff(h)

        time = np.concatenate([[0.0], np.cumsum(0.5 * dh * (dt_dh[:-1] + dt_dh[1:]))])
        distance = np.concatenate([[0.0], np.cumsum(0.5 * dh * (dx_dh[:-1] + dx_dh[1:]))])
        outputs['time'], outputs['distance'] = time, distance
        outputs['time_total'], outputs['distance_total'] = time[-1], distance[-1]

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        h, V_c, V = inputs['altitude'], inputs['V_c'], inputs['V']
        dt_dh, dx_dh = self._integrands(inputs)
        dh = np.diff(h)

        # d(segment)/d(node quantities), then accumulated by the cumulative sum
        seg_t = np.zeros((nn - 1, nn))
        seg_x = np.zeros((nn - 1, nn))
        seg_t_h = np.zeros((nn - 1, nn))
        seg_x_h = np.zeros((nn - 1, nn))
        seg_x_V = np.zeros((nn - 1, nn))
        for i in range(nn - 1):
            seg_t[i, i] = seg_t[i, i + 1] = 0.5 * dh[i]
            seg_x[i, i] = seg_x[i, i + 1] = 0.5 * dh[i]
            seg_t_h[i, i] = -0.5 * (dt_dh[i] + dt_dh[i + 1])
            seg_t_h[i, i + 1] = 0.5 * (dt_dh[i] + dt_dh[i + 1])
            seg_x_h[i, i] = -0.5 * (dx_dh[i] + dx_dh[i + 1])
            seg_x_h[i, i + 1] = 0.5 * (dx_dh[i] + dx_dh[i + 1])
            seg_x_V[i, i] = 0.5 * dh[i] / (60.0 * V_c[i])
            seg_x_V[i, i + 1] = 0.5 * dh[i] / (60.0 * V_c[i + 1])

        accumulate = np.tril(np.ones((nn, nn - 1)), -1)
        d_dt_dVc = -1.0 / V_c ** 2
        d_dx_dVc = -V / (60.0 * V_c ** 2)

        partials['time', 'V_c'] = accumulate @ (seg_t * d_dt_dVc)
        partials['time', 'altitude'] = accumulate @ seg_t_h
        partials['distance', 'V_c'] = accumulate @ (seg_x * d_dx_dVc)
        partials['distance', 'altitude'] = accumulate @ seg_x_h
        partials['distance', 'V'] = accumulate @ seg_x_V
        for name in ('V_c', 'altitude'):
            partials['time_total', name] = partials['time', name][-1]
            partials['distance_total', name] = partials['distance', name][-1]
        partials['distance_total', 'V'] = partials['distance', 'V'][-1]
