"""
LinearAccelerationComp -- G5, speed at which the linear acceleration law of
the takeoff reaches zero.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Optimum Takeoff Procedure at High Gross Weights" p. 367
("the acceleration capability is linear with speed", Figure 5.14).

The line through x_ddot_HIGE at hover and the G3 capability acc_ref at V_ref:

    V_max = V_ref acc_0 / (acc_0 - acc_ref)

    acc_0, acc_ref, V_ref --> V_max
"""

import openmdao.api as om


class LinearAccelerationComp(om.ExplicitComponent):
    """x_dot_max of the linear law, p. 367. Requires acc_0 > acc_ref."""

    def setup(self):
        self.add_input('acc_0', val=17.0, units='ft/s**2')
        self.add_input('acc_ref', val=12.0, units='ft/s**2')
        self.add_input('V_ref', val=60.0 * 1.6878, units='ft/s')
        self.add_output('V_max', val=300.0, units='ft/s')
        self.declare_partials('V_max', '*')

    def compute(self, inputs, outputs):
        a0, ar = inputs['acc_0'], inputs['acc_ref']
        outputs['V_max'] = inputs['V_ref'] * a0 / (a0 - ar)

    def compute_partials(self, inputs, J):
        a0, ar, V = inputs['acc_0'], inputs['acc_ref'], inputs['V_ref']
        d = a0 - ar
        J['V_max', 'V_ref'] = a0 / d
        J['V_max', 'acc_0'] = -V * ar / d ** 2
        J['V_max', 'acc_ref'] = V * a0 / d ** 2
