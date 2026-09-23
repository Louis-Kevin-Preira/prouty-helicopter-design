"""
PitchComp -- local blade pitch from the collective and the built-in twist.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 4, p. 69; twist laws p. 14 (Figure 1.7) and p. 19.

Two twist laws, because they combine with the collective differently:

  'linear'  theta = theta_0 + d_theta          additive, p. 69
            theta_0 is the pitch at the reference station of TwistDistComp.

  'ideal'   theta = theta_0 / (r/R)            multiplicative, p. 19
            theta_0 is the pitch at the tip, since theta = theta_0 at r/R = 1.
            An ideally twisted blade cannot be written as theta_0 + d_theta
            with a d_theta independent of theta_0, which is why the ideal law
            lives here and not in TwistDistComp.

Step 4 carries a constraint rather than an equation: "choose minimum value of
theta_0 so that theta is always positive". It is checked and warned about here,
not enforced, so that an optimiser is free to pass through the region.

Deferred extension -- zero lift angle. The pitch produced here is geometric,
measured from the chord line, which is what step 6 needs to look up cl and cd.
The inflow equation of step 5, p. 70, instead assumes dL proportional to
(theta - phi) with theta measured from the zero lift line, so a cambered
airfoil would need a second output theta_zl = theta - alpha_L0 feeding
InflowRatioComp only. Left out for now because the airfoil model of Chapter 6
covers the symmetric NACA 0012 alone, for which alpha_L0 = 0; to be added when
a cambered section enters the model.

    theta_0, d_theta (or r_R) --> PitchComp --> theta (nn,) [deg]
"""

import warnings

import numpy as np
import openmdao.api as om


class PitchComp(om.ExplicitComponent):
    """Local blade pitch, linear or ideal twist."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)
        self.options.declare('twist_law', values=('linear', 'ideal'),
                             default='linear')

    def setup(self):
        nn = self.options['num_nodes']
        ideal = self.options['twist_law'] == 'ideal'
        self._warned = False

        self.add_input('theta_0', val=8.0 if ideal else 17.5, units='deg',
                       desc='tip pitch (ideal) or collective pitch (linear)')
        self.add_output('theta', shape=(nn,), units='deg', desc='local pitch')

        ar = np.arange(nn)
        zeros = np.zeros(nn, int)
        self.declare_partials('theta', 'theta_0', rows=ar, cols=zeros)

        if ideal:
            self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
            self.declare_partials('theta', 'r_R', rows=ar, cols=ar)
        else:
            self.add_input('d_theta', shape=(nn,), units='deg',
                           desc='twist increment')
            self.declare_partials('theta', 'd_theta', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        if self.options['twist_law'] == 'ideal':
            theta = inputs['theta_0'] / inputs['r_R']
        else:
            theta = inputs['theta_0'] + inputs['d_theta']

        if not self._warned and np.any(np.real(theta) <= 0.0):
            self._warned = True
            warnings.warn('theta is not positive at every station; p. 69 asks '
                          'for a collective large enough to keep it positive.')
        outputs['theta'] = theta

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        if self.options['twist_law'] == 'ideal':
            partials['theta', 'theta_0'] = 1.0 / inputs['r_R']
            partials['theta', 'r_R'] = -inputs['theta_0'] / inputs['r_R'] ** 2
        else:
            partials['theta', 'theta_0'] = np.ones(nn)
