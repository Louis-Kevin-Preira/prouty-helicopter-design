"""
LocalMachComp -- local blade element Mach number.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 3, p. 69:

    M = (r/R) (Omega R) / V_son

In hover the blade element sees only its own rotational speed, so the Mach
number is linear in r/R and independent of the collective. That is what breaks
the apparent cycle between the inflow and the airfoil model: the lift curve
slope a = f(M) is fixed as soon as the geometry and the tip speed are known.

The compressibility tip relief of step 7, p. 70, is a correction applied to the
outer 10% of the blade and is handled separately by TipReliefComp, so the Mach
number produced here is the geometric one.

    r_R, V_tip, V_son --> LocalMachComp --> M (nn,)
"""

import numpy as np
import openmdao.api as om


class LocalMachComp(om.ExplicitComponent):
    """Rotational Mach number at each blade station."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed, Omega R')
        self.add_input('V_son', val=1116.45, units='ft/s', desc='speed of sound')
        self.add_output('M', shape=(nn,), desc='local Mach number')

        ar = np.arange(nn)
        zeros = np.zeros(nn, int)
        self.declare_partials('M', 'r_R', rows=ar, cols=ar)
        self.declare_partials('M', ['V_tip', 'V_son'], rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        outputs['M'] = inputs['r_R'] * inputs['V_tip'] / inputs['V_son']

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        r_R, V_tip, V_son = inputs['r_R'], inputs['V_tip'], inputs['V_son']

        partials['M', 'r_R'] = np.full(nn, V_tip / V_son)
        partials['M', 'V_tip'] = r_R / V_son
        partials['M', 'V_son'] = -r_R * V_tip / V_son ** 2
