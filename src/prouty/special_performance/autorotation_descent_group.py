"""
AutorotationDescentGroup -- G2b, steady rate of descent in autorotation.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Steady Rate of Descent in Autorotation" pp. 350-351, Figure 5.5;
Chapter 3 trim in autorotation p. 196-198.

    speed     ExecComp               mu = V / V_tip
    trim      TrimConditionsGroup    mode='autorotation' (C_Q/sigma = losses)
    descent   ExecComp               R/D = -R/C,  L/D = V / (R/D)   (p. 351)

Forward speeds from about 40 kt (the high-speed induced velocity of the trim
is singular at mu = 0; the value at V = 0 is the vertical autorotation of
Chapter 2). trim_options are passed to TrimConditionsGroup (e.g. rotor).
"""

import numpy as np
import openmdao.api as om

from prouty.forward_flight.trim_conditions_group import TrimConditionsGroup


class AutorotationDescentGroup(om.Group):
    """Rate of descent and equivalent L/D at the speeds V, pp. 350-351."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('trim_options', types=dict, default={})

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('speed', om.ExecComp(
            'mu = V / V_tip', mu=np.zeros(nn),
            V={'val': 100.0 * np.ones(nn), 'units': 'ft/s'},
            V_tip={'val': 650.0, 'units': 'ft/s'}), promotes=['*'])
        trim = TrimConditionsGroup(mode='autorotation', num_nodes=nn,
                                   **self.options['trim_options'])
        self.add_subsystem('trim', trim, promotes=['*'])
        self.add_subsystem('descent', om.ExecComp(
            ['RD = -R_C', 'LD = -V / R_C'],
            RD={'val': np.ones(nn), 'units': 'ft/s'},
            R_C={'val': -np.ones(nn), 'units': 'ft/s'},
            V={'val': 100.0 * np.ones(nn), 'units': 'ft/s'},
            LD=np.ones(nn), has_diag_partials=True), promotes=['*'])
        self.set_input_defaults('V_tip', val=650.0, units='ft/s')
