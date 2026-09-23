"""
DimensionalPerfComp -- thrust, power and shaft torque.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 21, p. 72:

    T  = rho A (Omega R)^2 C_T                       [lb]
    hp = rho A (Omega R)^3 C_Q / 550                  [hp]

Shaft torque follows from the power and the rotational speed, Q = P / Omega
with Omega = V_tip / R, which reduces to rho A V_tip^2 R C_Q. It is not in the
book's step list but it sizes the transmission, so it is produced here rather
than recomputed by hand later.

This is the only component that turns the whole calculation back into pounds
and horsepower. Everything upstream is non-dimensional apart from the disc
loading of step 17, which exists solely to enter Figure 1.34.

For the example helicopter at sea level, p. 76: T = 20,400 lb and hp = 1,990.

    rho, A, V_tip, R, CT, CQ --> DimensionalPerfComp --> T, power_hp, Q
"""

import openmdao.api as om

FT_LB_PER_HP = 550.0


class DimensionalPerfComp(om.ExplicitComponent):
    """Dimensional rotor thrust, power and torque."""

    def setup(self):
        self.add_input('rho', val=0.002377, units='slug/ft**3', desc='density')
        self.add_input('A', val=2827.4, units='ft**2', desc='disc area')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed')
        self.add_input('R', val=30.0, units='ft', desc='rotor radius')
        self.add_input('CT', val=0.00719, desc='thrust coefficient')
        self.add_input('CQ', val=6.07e-4, desc='torque coefficient')

        self.add_output('T', val=20400.0, units='lbf', desc='rotor thrust')
        self.add_output('power_hp', val=1990.0, units='hp', desc='rotor power')
        self.add_output('Q', val=5.0e4, units='ft*lbf', desc='shaft torque')

        self.declare_partials('T', ['rho', 'A', 'V_tip', 'CT'])
        self.declare_partials('power_hp', ['rho', 'A', 'V_tip', 'CQ'])
        self.declare_partials('Q', ['rho', 'A', 'V_tip', 'R', 'CQ'])

    def compute(self, inputs, outputs):
        rho, A, v = inputs['rho'], inputs['A'], inputs['V_tip']
        ct, cq = inputs['CT'], inputs['CQ']

        outputs['T'] = rho * A * v ** 2 * ct
        outputs['power_hp'] = rho * A * v ** 3 * cq / FT_LB_PER_HP
        outputs['Q'] = rho * A * v ** 2 * inputs['R'] * cq

    def compute_partials(self, inputs, partials):
        rho, A, v, R = inputs['rho'], inputs['A'], inputs['V_tip'], inputs['R']
        ct, cq = inputs['CT'], inputs['CQ']

        partials['T', 'rho'] = A * v ** 2 * ct
        partials['T', 'A'] = rho * v ** 2 * ct
        partials['T', 'V_tip'] = 2.0 * rho * A * v * ct
        partials['T', 'CT'] = rho * A * v ** 2

        k = 1.0 / FT_LB_PER_HP
        partials['power_hp', 'rho'] = A * v ** 3 * cq * k
        partials['power_hp', 'A'] = rho * v ** 3 * cq * k
        partials['power_hp', 'V_tip'] = 3.0 * rho * A * v ** 2 * cq * k
        partials['power_hp', 'CQ'] = rho * A * v ** 3 * k

        partials['Q', 'rho'] = A * v ** 2 * R * cq
        partials['Q', 'A'] = rho * v ** 2 * R * cq
        partials['Q', 'V_tip'] = 2.0 * rho * A * v * R * cq
        partials['Q', 'R'] = rho * A * v ** 2 * cq
        partials['Q', 'CQ'] = rho * A * v ** 2 * R
