"""
ClimboutDistanceComp -- G5, distance to climb over the obstacle and total
takeoff distance.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Optimum Takeoff Procedure at High Gross Weights" pp. 367-368.

    R/C   = 33,000 (hp_avail - hp_level(x_dot_rot)) / G.W.      ft/min (momentum)
    x_CL  = 60 h x_dot_rot / (R/C)  = h x_dot_rot G.W. / (550 Delta_hp)
    x_tot = x_acc + x_CL

    P_avail, P_level (nn,), GW, h, V_rot (nn,), x_acc (nn,) --> R_C, x_cl, x_tot (nn,)
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class ClimboutDistanceComp(om.ExplicitComponent):
    """Climb-out distance at constant rotation speed, p. 368. Requires P_avail > P_level."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('P_avail', val=4000.0, units='hp')
        self.add_input('P_level', val=3500.0 * np.ones(nn), units='hp')
        self.add_input('GW', val=20000.0, units='lbf')
        self.add_input('h', val=50.0, units='ft', desc='obstacle height')
        self.add_input('V_rot', val=40.0 * np.ones(nn), units='ft/s')
        self.add_input('x_acc', val=np.zeros(nn), units='ft')
        self.add_output('R_C', val=np.ones(nn), units='ft/s')
        self.add_output('x_cl', val=np.ones(nn), units='ft')
        self.add_output('x_tot', val=np.ones(nn), units='ft')
        self.declare_partials('R_C', 'P_level', rows=ar, cols=ar)
        self.declare_partials('R_C', ['P_avail', 'GW'], rows=ar, cols=zero)
        for out in ('x_cl', 'x_tot'):
            self.declare_partials(out, ['P_level', 'V_rot'], rows=ar, cols=ar)
            self.declare_partials(out, ['P_avail', 'GW', 'h'], rows=ar, cols=zero)
        self.declare_partials('x_tot', 'x_acc', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        dP = inputs['P_avail'] - inputs['P_level']
        W = inputs['GW']
        outputs['R_C'] = HP_TO_FT_LBF_PER_S * dP / W
        outputs['x_cl'] = inputs['h'] * inputs['V_rot'] * W / (HP_TO_FT_LBF_PER_S * dP)
        outputs['x_tot'] = inputs['x_acc'] + outputs['x_cl']

    def compute_partials(self, inputs, J):
        dP = inputs['P_avail'] - inputs['P_level']
        W, h, V = inputs['GW'], inputs['h'], inputs['V_rot']
        J['R_C', 'P_avail'] = HP_TO_FT_LBF_PER_S / W
        J['R_C', 'P_level'] = -HP_TO_FT_LBF_PER_S / W
        J['R_C', 'GW'] = -HP_TO_FT_LBF_PER_S * dP / W ** 2
        x = h * V * W / (HP_TO_FT_LBF_PER_S * dP)
        for out in ('x_cl', 'x_tot'):
            J[out, 'P_avail'] = -x / dP
            J[out, 'P_level'] = x / dP
            J[out, 'GW'] = x / W
            J[out, 'h'] = x / h
            J[out, 'V_rot'] = x / V
