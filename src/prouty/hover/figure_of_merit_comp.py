"""
FigureOfMeritComp -- hover efficiency measures.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, p. 9 and p. 22.

    F.M. = C_T^1.5 / (sqrt(2) C_Q)          ideal power over actual power
    P.L. = T / hp                            power loading, lb per horsepower

The figure of merit compares the actual torque against the momentum theory
minimum for the same thrust, so it isolates how well the rotor is doing at a
given operating point. Power loading is the same information in engineering
units, and it is what sizes an engine.

Neither is a design objective on its own. Figure of merit rises with disc
loading for a fixed rotor, so a small rotor worked hard can show a good F.M.
while burning more power than a large one; p. 22 makes the point that the two
must be read together with the disc loading.

    CT, CQ, T, power_hp --> FigureOfMeritComp --> FM, power_loading
"""

import numpy as np
import openmdao.api as om

SQRT2 = np.sqrt(2.0)
TINY = 1e-30


class FigureOfMeritComp(om.ExplicitComponent):
    """Figure of merit and power loading."""

    def setup(self):
        self.add_input('CT', val=0.00719, desc='thrust coefficient')
        self.add_input('CQ', val=6.07e-4, desc='torque coefficient')
        self.add_input('T', val=20400.0, units='lbf', desc='rotor thrust')
        self.add_input('power_hp', val=1990.0, units='hp', desc='rotor power')

        self.add_output('FM', val=0.7, desc='figure of merit')
        self.add_output('power_loading', val=10.0, units='lbf/hp',
                        desc='thrust per horsepower')

        self.declare_partials('FM', ['CT', 'CQ'])
        self.declare_partials('power_loading', ['T', 'power_hp'])

    def compute(self, inputs, outputs):
        ct = np.where(np.real(inputs['CT']) > TINY, inputs['CT'], TINY)
        outputs['FM'] = ct ** 1.5 / (SQRT2 * inputs['CQ'])
        outputs['power_loading'] = inputs['T'] / inputs['power_hp']

    def compute_partials(self, inputs, partials):
        ct = np.where(np.real(inputs['CT']) > TINY, inputs['CT'], TINY)
        cq, hp = inputs['CQ'], inputs['power_hp']

        partials['FM', 'CT'] = 1.5 * np.sqrt(ct) / (SQRT2 * cq)
        partials['FM', 'CQ'] = -ct ** 1.5 / (SQRT2 * cq ** 2)
        partials['power_loading', 'T'] = 1.0 / hp
        partials['power_loading', 'power_hp'] = -inputs['T'] / hp ** 2
