"""
DayTemperatureComp -- temperature offset of the reference day.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4: the "95 deg F day" of Figure 4.1 p. 274, Figure 4.2 p. 275,
Figures 4.33-4.35 pp. 314-316; Appendix C, Figure C.1 p. 703.

AtmosphereComp (Chapter 1) takes the temperature as an offset dT from the
standard day, constant with altitude. Prouty's hot day is not that: it is
isothermal, 95 deg F at every altitude. Figure 4.33 prints its altitudes for
four density ratios, and only the isothermal day reproduces them (C4-4):

    rho/rho_0     book 95 F     dT = +36 F     isothermal 95 F
    0.367         23,800 ft     28,268 ft      23,710 ft
    0.533         14,700        17,727         14,753
    0.862          2,200         2,714          2,235

The offset that turns AtmosphereComp into any of the three days is

    standard     dT = 0
    offset       dT = dT_offset
    isothermal   dT = T_day - T_0 (1 - L h)

    altitude, [dT_offset | T_day] --> dT --> AtmosphereComp
"""

import numpy as np
import openmdao.api as om

from prouty.hover.atmosphere_comp import LAPSE, T_0

T_95F = 95.0 + 459.67       # deg R, Prouty's hot day


def hot_day_weight(h, T):
    """
    Position of T between the standard and 95 deg F days at altitude h.

    w = (T - T_std) / (T_95 - T_std), 0 on the standard day, 1 on the
    isothermal hot day (C4-4). Returns w, dw/dh (1/ft), dw/dT (1/degR).
    Used to blend the two days of Figures 4.1 and 4.3.
    """
    T_std = T_0 * (1.0 - LAPSE * h)
    span = T_95F - T_std
    w = (T - T_std) / span
    return w, (T - T_95F) / span ** 2 * (-T_0 * LAPSE), 1.0 / span


class DayTemperatureComp(om.ExplicitComponent):
    """Offset from the standard day for a standard, offset or isothermal day."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('day', default='standard',
                             values=('standard', 'offset', 'isothermal'))

    def setup(self):
        nn, day = self.options['num_nodes'], self.options['day']
        ar = np.arange(nn)

        self.add_input('altitude', val=np.zeros(nn), units='ft', desc='pressure altitude')
        self.add_output('dT', val=np.zeros(nn), units='degR',
                        desc='temperature offset from the standard day')

        if day == 'offset':
            self.add_input('dT_offset', val=np.zeros(nn), units='degR')
            self.declare_partials('dT', 'dT_offset', rows=ar, cols=ar, val=1.0)
        elif day == 'isothermal':
            self.add_input('T_day', val=np.full(nn, T_95F), units='degR',
                           desc='ambient temperature, the same at every altitude')
            self.declare_partials('dT', 'T_day', rows=ar, cols=ar, val=1.0)
            self.declare_partials('dT', 'altitude', rows=ar, cols=ar, val=T_0 * LAPSE)

    def compute(self, inputs, outputs):
        day = self.options['day']
        if day == 'standard':
            outputs['dT'] = 0.0
        elif day == 'offset':
            outputs['dT'] = inputs['dT_offset']
        else:
            outputs['dT'] = inputs['T_day'] - T_0 * (1.0 - LAPSE * inputs['altitude'])
