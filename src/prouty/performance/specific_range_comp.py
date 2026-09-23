"""
SpecificRangeComp -- distance flown per pound of fuel.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Cruising Flight" p. 323, Figures 4.42 and 4.43.

    S.R. = V_ground / fuel flow          n.mi. per lb

with the ground speed reduced by a headwind and increased by a tailwind:

    V_ground = V - V_wind                V_wind positive as a headwind

Wind moves the best cruise speed as well as the range: p. 323 gives 114 kt in
still air for the example, 127 kt into a 40 kt headwind and 104 kt with a
40 kt tailwind. The speed for best range is the tangent from the origin to the
fuel flow curve, which BestRangeSpeedBalance solves; this component only
states the definition.

    V, V_wind, FF (nn,) --> V_ground, SR (nn,)
"""

import numpy as np
import openmdao.api as om


class SpecificRangeComp(om.ExplicitComponent):
    """Specific range in still air or in wind, p. 323."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('V', val=np.ones(nn), units='kn', desc='true airspeed')
        self.add_input('V_wind', val=0.0, units='kn', desc='headwind, negative for a tailwind')
        self.add_input('FF', val=np.ones(nn), units='lbm/h', desc='fuel flow, all engines')
        self.add_output('V_ground', val=np.ones(nn), units='kn', desc='ground speed')
        self.add_output('SR', val=np.zeros(nn), units='NM/lbm', desc='specific range')

        self.declare_partials('V_ground', 'V', rows=ar, cols=ar, val=1.0)
        self.declare_partials('V_ground', 'V_wind', val=-np.ones((nn, 1)))
        self.declare_partials('SR', ['V', 'FF'], rows=ar, cols=ar)
        self.declare_partials('SR', 'V_wind')

    def compute(self, inputs, outputs):
        V_ground = inputs['V'] - inputs['V_wind'][0]
        outputs['V_ground'] = V_ground
        outputs['SR'] = V_ground / inputs['FF']

    def compute_partials(self, inputs, partials):
        FF = inputs['FF']
        V_ground = inputs['V'] - inputs['V_wind'][0]

        partials['SR', 'V'] = 1.0 / FF
        partials['SR', 'V_wind'] = (-1.0 / FF).reshape(-1, 1)
        partials['SR', 'FF'] = -V_ground / FF ** 2
