"""
MeanAngleOfAttackComp -- mean blade lift coefficient and its angle of attack.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rate of Descent in Vertical Autorotation" p. 112 (c_l bar);
Chapter 6, linear lift range of the NACA 0012 model, p. 428.

    cl_bar    = 6 C_T/sigma
    alpha_bar = cl_bar / a_sec          a_sec in 1/deg at the section Mach number

The drag of p. 112 is the section drag at cl_bar (decision: Chapter 6 model).
The linear range holds up to alpha_L; beyond it a warning is issued, the
post-stall lift decrement being outside what the autorotation estimate covers.

    CT_sigma, a_sec, alpha_L (nn,) --> cl_bar, alpha_bar (nn,)
"""

import warnings

import numpy as np
import openmdao.api as om


class MeanAngleOfAttackComp(om.ExplicitComponent):
    """Angle of attack giving the mean lift coefficient 6 C_T/sigma, pp. 112, 428."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('CT_sigma', val=np.full(nn, 0.085), desc='thrust coefficient over solidity')
        self.add_input('a_sec', val=np.full(nn, 0.1), units='1/deg', desc='section lift slope')
        self.add_input('alpha_L', val=np.full(nn, 8.0), units='deg', desc='stall onset angle')
        self.add_output('cl_bar', val=np.full(nn, 0.5), desc='mean lift coefficient')
        self.add_output('alpha_bar', val=np.full(nn, 5.0), units='deg')
        self.declare_partials('cl_bar', 'CT_sigma', rows=ar, cols=ar, val=6.0)
        self.declare_partials('alpha_bar', ['CT_sigma', 'a_sec'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['cl_bar'] = 6.0 * inputs['CT_sigma']
        outputs['alpha_bar'] = outputs['cl_bar'] / inputs['a_sec']
        if np.any(np.real(outputs['alpha_bar']) > np.real(inputs['alpha_L'])):
            warnings.warn('MeanAngleOfAttackComp: alpha_bar above alpha_L, blades near stall (p. 97)')

    def compute_partials(self, inputs, partials):
        a = inputs['a_sec']
        partials['alpha_bar', 'CT_sigma'] = 6.0 / a
        partials['alpha_bar', 'a_sec'] = -6.0 * inputs['CT_sigma'] / a ** 2
