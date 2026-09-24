"""
TakeoffAccelerationDistanceComp -- G5, ground-effect acceleration distance to
the rotation speed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Optimum Takeoff Procedure at High Gross Weights" p. 367.

Acceleration linear with speed between hover (IGE) and x_dot_max:

    x_ddot = x_ddot_HIGE + (d x_ddot/d x_dot) x_dot,   d x_ddot/d x_dot = -x_ddot_HIGE/x_dot_max
    x_acc  = x_dot_max [ln(1 - u) + u] / (d x_ddot/d x_dot),   u = x_dot_rot / x_dot_max
           = x_dot_max^2 [-ln(1 - u) - u] / x_ddot_HIGE

    acc_0, V_max, V_rot (nn,) --> x_acc (nn,), t_acc (nn,)
"""

import numpy as np
import openmdao.api as om


class TakeoffAccelerationDistanceComp(om.ExplicitComponent):
    """Distance and time to accelerate from hover to V_rot, p. 367. Requires V_rot < V_max."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('acc_0', val=10.0, units='ft/s**2', desc='x_ddot_HIGE')
        self.add_input('V_max', val=250.0, units='ft/s', desc='x_dot_max, zero acceleration')
        self.add_input('V_rot', val=40.0 * np.ones(nn), units='ft/s')
        self.add_output('x_acc', val=np.ones(nn), units='ft')
        self.add_output('t_acc', val=np.ones(nn), units='s')
        for out in ('x_acc', 't_acc'):
            self.declare_partials(out, 'V_rot', rows=ar, cols=ar)
            self.declare_partials(out, ['acc_0', 'V_max'], rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        a, Vm, V = inputs['acc_0'], inputs['V_max'], inputs['V_rot']
        u = V / Vm
        L = -np.log(1.0 - u)
        outputs['x_acc'] = Vm ** 2 * (L - u) / a
        outputs['t_acc'] = Vm * L / a            # x_dot = x_dot_max (1 - exp(-t a/x_dot_max))

    def compute_partials(self, inputs, J):
        a, Vm, V = inputs['acc_0'], inputs['V_max'], inputs['V_rot']
        u = V / Vm
        L = -np.log(1.0 - u)
        dL_du = 1.0 / (1.0 - u)
        x = Vm ** 2 * (L - u) / a
        dx_du = Vm ** 2 * (dL_du - 1.0) / a
        J['x_acc', 'V_rot'] = dx_du / Vm
        J['x_acc', 'acc_0'] = -x / a
        J['x_acc', 'V_max'] = 2.0 * x / Vm - dx_du * u / Vm
        t = Vm * L / a
        J['t_acc', 'V_rot'] = dL_du / a
        J['t_acc', 'acc_0'] = -t / a
        J['t_acc', 'V_max'] = L / a - dL_du * u / a
