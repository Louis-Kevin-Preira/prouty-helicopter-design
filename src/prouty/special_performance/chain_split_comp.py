"""
ChainSplitComp -- G2c chain, splits the node results back to V_1 and V_0.

    RD_nodes, P_nodes (nn+1,) --> RD (nn,), P_1 (nn,), P_0
"""

import numpy as np
import openmdao.api as om


class ChainSplitComp(om.ExplicitComponent):
    """R/D and power at V_1; power at V_0."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('RD_nodes', val=np.ones(nn + 1), units='ft/s')
        self.add_input('P_nodes', val=np.ones(nn + 1), units='hp')
        self.add_output('RD', val=np.ones(nn), units='ft/s')
        self.add_output('P_1', val=np.ones(nn), units='hp')
        self.add_output('P_0', val=1.0, units='hp')
        self.declare_partials('RD', 'RD_nodes', rows=ar, cols=ar, val=1.0)
        self.declare_partials('P_1', 'P_nodes', rows=ar, cols=ar, val=1.0)
        self.declare_partials('P_0', 'P_nodes', rows=[0], cols=[nn], val=1.0)

    def compute(self, inputs, outputs):
        outputs['RD'] = inputs['RD_nodes'][:-1]
        outputs['P_1'] = inputs['P_nodes'][:-1]
        outputs['P_0'] = inputs['P_nodes'][-1]
