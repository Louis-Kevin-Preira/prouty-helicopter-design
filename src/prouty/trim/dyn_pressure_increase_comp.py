"""Dynamic pressure increase at the horizontal stabiliser at low speed.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 493, Figure 8.10, from Blake and
Alansky, "Stability and Control of the YUH-61A", JAHS 22-1, 1977.

Digitised by tracking the printed curve column by column from its peak
outward. The track passes through every measured point the figure plots:
0.80 at V/v1 = 1.15, 1.11 at 1.9, 0.35 at 2.7 and zero at 3.7, with the
low-speed zero at 0.5.
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_V_V1 = np.arange(0.45, 3.8001, 0.05)

# (q_H - q)/D.L. of Figure 8.10 p. 493
_DQ_DL = np.array([
      0.00000,   0.02827,   0.12079,   0.20608,   0.28460,   0.35684,   0.42326,   0.48433,
      0.54053,   0.59231,   0.64015,   0.68453,   0.72591,   0.76476,   0.80156,   0.83677,
      0.87086,   0.90424,   0.93675,   0.96792,   0.99727,   1.02435,   1.04866,   1.06975,
      1.08715,   1.10037,   1.10896,   1.11243,   1.11032,   1.10216,   1.08748,   1.06580,
      1.03666,   1.00009,   0.95710,   0.90883,   0.85640,   0.80097,   0.74365,   0.68559,
      0.62790,   0.57139,   0.51654,   0.46382,   0.41370,   0.36664,   0.32312,   0.28360,
      0.24848,   0.21766,   0.19072,   0.16726,   0.14688,   0.12917,   0.11374,   0.10017,
      0.08806,   0.07701,   0.06662,   0.05648,   0.04619,   0.03534,   0.02353,   0.01036,
      0.00000,   0.00000,   0.00000,   0.00000,
])



class DynPressureIncreaseComp(om.ExplicitComponent):
    """Local dynamic pressure at the stabiliser at low speed, p. 493::

        q_H = (q_H/q) q + [(q_H - q)/D.L.] D.L.

    Two effects are in play at the stabiliser and they pull opposite ways.
    The fuselage and rotor hub take momentum out of the flow, which is the
    ``q_H/q`` of Figure 8.9 p. 492, about 0.6 for the example helicopter. But
    at low speed the rotor wake sweeps across the stabiliser and puts energy
    back in, and Figure 8.10 measures that as a fraction of disc loading.

    The increment peaks at 1.11 disc loadings around ``V/v1_hover = 1.8`` and
    is zero below 0.5 and above 3.7. At 20,000 lb and 2,827 sq ft that peak
    is 7.9 lb/sq ft, against a free stream ``q`` of 6.6 at the same speed:
    the stabiliser sees more than twice the dynamic pressure the airframe
    does. Ignoring it understates the stabiliser download through transition,
    which is where a helicopter is most likely to pitch unexpectedly.

    Outside the digitised span the increment is zero, which is what the
    figure shows rather than an extrapolation: the wake has missed the
    stabiliser entirely by 3.7 and has not reached it below 0.5.

    Notes
    -----
    ``V_v1`` is flight speed over *hover* induced velocity, not over the
    induced velocity at the flight condition.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('V_v1', shape=(nn,), val=2.0,
                       desc='flight speed over hover induced velocity')
        self.add_input('q', shape=(nn,), val=1.0, units='lbf/ft**2',
                       desc='free stream dynamic pressure')
        self.add_input('qH_q_momentum', shape=(nn,), val=1.0,
                       desc='dynamic pressure ratio from Figure 8.9')
        self.add_input('DL', shape=(nn,), val=0.0, units='lbf/ft**2',
                       desc='main rotor disc loading')

        self.add_output('dq_DL', shape=(nn,),
                        desc='(q_H - q)/D.L., Figure 8.10')
        self.add_output('q_H', shape=(nn,), units='lbf/ft**2',
                        desc='dynamic pressure at the stabiliser')
        self.add_output('qH_q', shape=(nn,),
                        desc='q_H over q, for the stabiliser polar')

        self._chart = InterpND(method='akima', points=_V_V1, values=_DQ_DL,
                               extrapolate=True)
        self.declare_partials('dq_DL', 'V_v1', rows=ar, cols=ar)
        for name in ('q_H', 'qH_q'):
            self.declare_partials(name, ['V_v1', 'q', 'qH_q_momentum', 'DL'],
                                  rows=ar, cols=ar)

    def _chart_value(self, V_v1):
        inside = (V_v1 > _V_V1[0]) & (V_v1 < _V_V1[-1])
        v, dv = self._chart.interpolate(np.clip(V_v1, _V_V1[0], _V_V1[-1]),
                                        compute_derivative=True)
        return np.where(inside, v, 0.0), np.where(inside, dv.ravel(), 0.0)

    def compute(self, inputs, outputs):
        v, _ = self._chart_value(inputs['V_v1'])
        outputs['dq_DL'] = v
        outputs['q_H'] = (inputs['qH_q_momentum'] * inputs['q']
                          + v * inputs['DL'])
        outputs['qH_q'] = outputs['q_H'] / inputs['q']

    def compute_partials(self, inputs, J):
        v, dv = self._chart_value(inputs['V_v1'])
        J['dq_DL', 'V_v1'] = dv
        J['q_H', 'V_v1'] = dv * inputs['DL']
        J['q_H', 'q'] = inputs['qH_q_momentum']
        J['q_H', 'qH_q_momentum'] = inputs['q']
        J['q_H', 'DL'] = v

        q, DL = inputs['q'], inputs['DL']
        J['qH_q', 'V_v1'] = dv * DL / q
        J['qH_q', 'q'] = -v * DL / q ** 2
        J['qH_q', 'qH_q_momentum'] = np.ones_like(q)
        J['qH_q', 'DL'] = v / q
