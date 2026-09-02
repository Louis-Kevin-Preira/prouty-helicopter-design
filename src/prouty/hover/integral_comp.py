"""
IntegralComp -- integral of a radial loading, given quadrature weights.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, steps 9 and 11 (thrust, p. 70-71) and steps 13 and 15 (profile and
induced torque, p. 71). All four are the same operation, the scalar product of
the weights produced by IntegrationWeightsComp with a loading tabulated at the
stations, so one component with configurable variable names serves them all:

    integral = w . loading

Two instances cover the thrust: one with the weights running to r/R = 1 for the
thrust without tip loss of step 9, one with the weights cut at B for the
corrected thrust of step 11. The loading itself is computed once and shared,
which is the whole point of keeping the grid fixed and moving the weights.

    w, loading --> IntegralComp --> integral
"""

import numpy as np
import openmdao.api as om


class IntegralComp(om.ExplicitComponent):
    """Scalar product of quadrature weights and a radial loading."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)
        self.options.declare('weights', types=str, default='w',
                             desc='name of the quadrature weights input')
        self.options.declare('integrand', types=str, default='dCT_dr',
                             desc='name of the loading input')
        self.options.declare('integral', types=str, default='CT',
                             desc='name of the scalar output')

    def setup(self):
        nn = self.options['num_nodes']
        self._w = self.options['weights']
        self._f = self.options['integrand']
        self._I = self.options['integral']

        self.add_input(self._w, shape=(nn,), desc='quadrature weights')
        self.add_input(self._f, shape=(nn,), desc='radial loading')
        self.add_output(self._I, val=0.0, desc='integrated coefficient')

        self.declare_partials(self._I, [self._w, self._f])

    def compute(self, inputs, outputs):
        outputs[self._I] = inputs[self._w] @ inputs[self._f]

    def compute_partials(self, inputs, partials):
        partials[self._I, self._w] = inputs[self._f]
        partials[self._I, self._f] = inputs[self._w]
