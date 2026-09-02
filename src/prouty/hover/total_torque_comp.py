"""
TotalTorqueComp -- total torque coefficient with the empirical correction.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 20, p. 72:

    C_Q = (C_Q0 + C_Qi + dC_Qi) x (measured power / calculated power)

Three losses are summed before the correction is applied: profile drag from
step 13, thrust induced drag from step 15, and the swirl left in the wake from
step 16. The empirical factor of step 19 then scales the whole sum, not just
the induced part, because Figure 1.34 was built by comparing measured shaft
power against a calculation of the same three terms.

C_Q/sigma is produced as well. It is the torque analogue of the blade loading,
and it is what Figure 1.45 quotes, at 0.0070 for the example helicopter.

    CQ0, CQi, CQi_rot, power_factor, sigma
        --> TotalTorqueComp --> CQ_uncorrected, CQ, CQ_sigma
"""

import openmdao.api as om


class TotalTorqueComp(om.ExplicitComponent):
    """Sum of the torque contributions, scaled by the empirical factor."""

    def setup(self):
        self.add_input('CQ0', val=1.11e-4, desc='profile torque, step 13')
        self.add_input('CQi', val=4.59e-4, desc='induced torque, step 15')
        self.add_input('CQi_rot', val=7.8e-6, desc='wake rotation, step 16')
        self.add_input('power_factor', val=1.05,
                       desc='measured over calculated power, step 19')
        self.add_input('sigma', val=0.085, desc='solidity')

        self.add_output('CQ_uncorrected', val=5.78e-4,
                        desc='sum of the three torque terms')
        self.add_output('CQ', val=6.07e-4, desc='total torque coefficient')
        self.add_output('CQ_sigma', val=0.0071, desc='C_Q / sigma')

        self.declare_partials('CQ_uncorrected', ['CQ0', 'CQi', 'CQi_rot'],
                              val=1.0)
        self.declare_partials('CQ', ['CQ0', 'CQi', 'CQi_rot', 'power_factor'])
        self.declare_partials('CQ_sigma', ['CQ0', 'CQi', 'CQi_rot',
                                           'power_factor', 'sigma'])

    def compute(self, inputs, outputs):
        total = inputs['CQ0'] + inputs['CQi'] + inputs['CQi_rot']
        cq = total * inputs['power_factor']

        outputs['CQ_uncorrected'] = total
        outputs['CQ'] = cq
        outputs['CQ_sigma'] = cq / inputs['sigma']

    def compute_partials(self, inputs, partials):
        total = inputs['CQ0'] + inputs['CQi'] + inputs['CQi_rot']
        k, s = inputs['power_factor'], inputs['sigma']

        for name in ('CQ0', 'CQi', 'CQi_rot'):
            partials['CQ', name] = k
            partials['CQ_sigma', name] = k / s

        partials['CQ', 'power_factor'] = total
        partials['CQ_sigma', 'power_factor'] = total / s
        partials['CQ_sigma', 'sigma'] = -total * k / s ** 2
