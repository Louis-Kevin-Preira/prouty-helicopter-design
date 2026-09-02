"""
AvgLiftCoefComp -- average blade element lift coefficient.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 168, and Chapter 1 for the hovering relation.

Assuming every blade element works at the same lift coefficient and
integrating the thrust gives, in forward flight (p. 168):

    C_T/sigma = (cl_bar / 6) (1 + (3/2) mu^2)
    cl_bar    = 6 (C_T/sigma) / (1 + (3/2) mu^2)          form = 'forward'

and in hover, where the mu term drops out:

    cl_bar    = 6 (C_T/sigma)                             form = 'hover'

What it is for. cl_bar is Prouty's cheap stall indicator: it says how much
lift the average blade element is being asked to produce, so comparing it
with the airfoil's cl_max tells how close the rotor runs to its limit.
Prouty is candid that its usefulness is limited in forward flight (p. 168),
since stall there is localised on the retreating side rather than spread
over the disc -- alpha_1,270 and the maximum profile torque of p. 228 are
the honest indicators. cl_bar remains useful as a design-sizing number and,
above all, as the reference condition for the mean drag coefficient.

Why both forms exist, and it is not a matter of taste. Having derived the
forward flight relation on p. 168, Prouty then prescribes on p. 175 that the
drag coefficient be evaluated "as a function of C_T/sigma as it was in hover"
-- that is, with the 'hover' form, no mu correction. The two differ by
1 + 1.5 mu^2, which is 12 % at mu = 0.3 and 30 % at mu = 0.45. So the
quantity feeding the drag model is deliberately NOT the mean lift coefficient
of the same chapter. Both forms are kept so the choice is explicit; p. 175 is
followed by default where the drag is concerned, and 'forward' is the right
one wherever cl_bar is wanted for its own sake.

    CT_sigma, mu --> AvgLiftCoefComp --> cl_bar (nn,)
"""

import numpy as np
import openmdao.api as om

CL_BAR_FACTOR = 6.0


class AvgLiftCoefComp(om.ExplicitComponent):
    """Average blade element lift coefficient, p. 168."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('form', values=('forward', 'hover'),
                             default='forward')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('CT_sigma', shape=(nn,), desc='C_T / sigma')
        self.add_output('cl_bar', shape=(nn,),
                        desc='average blade element lift coefficient')
        self.declare_partials('cl_bar', 'CT_sigma', rows=ar, cols=ar)

        if self.options['form'] == 'forward':
            self.add_input('mu', shape=(nn,), desc='tip speed ratio')
            self.declare_partials('cl_bar', 'mu', rows=ar, cols=ar)

    def _scale(self, inputs):
        if self.options['form'] == 'hover':
            return np.ones_like(inputs['CT_sigma'])
        return 1.0 + 1.5 * inputs['mu'] ** 2

    def compute(self, inputs, outputs):
        outputs['cl_bar'] = CL_BAR_FACTOR * inputs['CT_sigma'] / self._scale(inputs)

    def compute_partials(self, inputs, partials):
        scale = self._scale(inputs)
        partials['cl_bar', 'CT_sigma'] = CL_BAR_FACTOR / scale

        if self.options['form'] == 'forward':
            cl_bar = CL_BAR_FACTOR * inputs['CT_sigma'] / scale
            partials['cl_bar', 'mu'] = -cl_bar * 3.0 * inputs['mu'] / scale
