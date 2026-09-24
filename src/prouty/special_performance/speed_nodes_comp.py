"""
SpeedNodesComp -- G2c chain, speeds [V_1 ..., V_0] for the trims.

    V_1 (nn,), V_0 --> V_nodes (nn+1,)
"""

import numpy as np
import openmdao.api as om


class SpeedNodesComp(om.ExplicitComponent):
    """Stacks the autorotation speeds and the failure speed."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('V_1', val=150.0 * np.ones(nn), units='ft/s')
        self.add_input('V_0', val=270.0, units='ft/s')
        self.add_output('V_nodes', val=np.ones(nn + 1), units='ft/s')
        self.declare_partials('V_nodes', 'V_1', rows=np.arange(nn), cols=np.arange(nn), val=1.0)
        self.declare_partials('V_nodes', 'V_0', rows=[nn], cols=[0], val=1.0)

    def compute(self, inputs, outputs):
        outputs['V_nodes'] = np.concatenate((inputs['V_1'], inputs['V_0']))
