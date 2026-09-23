"""
IntegrationWeightsComp -- trapezoidal quadrature weights on a fixed radial grid,
with a variable upper limit.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, steps 9 and 11, p. 70-71. Step 9 integrates the thrust loading to
r/R = 1, step 11 repeats it to the tip loss factor B.

The grid stays fixed from the cutout to the tip; only the weights change. That
keeps the upstream blade element calculation independent of B and leaves the
chain acyclic, since B is itself a function of the step 9 result.

Given the loading f at the stations, the integral is w . f. The cell containing
x_max is cut in two and its contribution taken from a linear interpolation of f,
which gives, with s = x_max - x_i and t = s / h,

    w_i     += s - s t / 2          w_{i+1} += s t / 2

These reduce to the full trapezoid h/2, h/2 at t = 1, and the weights and their
derivatives are continuous as x_max crosses a station, so the quadrature stays
differentiable while B moves through the grid.

Instantiate it twice: once with x_max left at its default of 1.0 for step 9,
once with x_max connected to B for step 11.

One kink, at the tip only. When x_max crosses an interior station the weights
stay smooth, because the cell that stops being partial is replaced by the next
one starting partial. The last station has no next cell, so at x_max = r_R[-1]
exactly, which is the default configuration of the step 9 instance, the
derivative of w[-1] with respect to r_R[-1] is one sided: +h/2 from the full
branch, -h/2 from the partial one. It never propagates, because BladeGridComp
maps the tip to r/R = 1 for every value of the cutout, so that entry of the
jacobian is always multiplied by a structural zero. Total derivatives are
unaffected; only a component level check_partials on that instance reports it.

    r_R, x_max --> IntegrationWeightsComp --> w (nn,)
"""

import numpy as np
import openmdao.api as om


class IntegrationWeightsComp(om.ExplicitComponent):
    """Quadrature weights from the first station to a variable upper limit."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
        self.add_input('x_max', val=1.0, desc='upper limit of integration')
        self.add_output('w', shape=(nn,), desc='quadrature weights')

        self.declare_partials('w', ['r_R', 'x_max'])

    def _cells(self, inputs):
        """Yield (i, h, t) for every cell that contributes; t = 1 if full."""
        x, xm = inputs['r_R'], inputs['x_max'][0]
        for i in range(self.options['num_nodes'] - 1):
            h = x[i + 1] - x[i]
            if np.real(xm) >= np.real(x[i + 1]):
                yield i, h, None
            elif np.real(xm) > np.real(x[i]):
                yield i, h, (xm - x[i]) / h

    def compute(self, inputs, outputs):
        outputs['w'][:] = 0.0
        for i, h, t in self._cells(inputs):
            if t is None:
                outputs['w'][i] += 0.5 * h
                outputs['w'][i + 1] += 0.5 * h
            else:
                s = t * h
                outputs['w'][i] += s - 0.5 * s * t
                outputs['w'][i + 1] += 0.5 * s * t

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        dx = np.zeros((nn, nn))
        dm = np.zeros(nn)

        for i, h, t in self._cells(inputs):
            if t is None:
                dx[i, i] -= 0.5
                dx[i, i + 1] += 0.5
                dx[i + 1, i] -= 0.5
                dx[i + 1, i + 1] += 0.5
            else:
                t = np.real(t)
                dx[i, i] += -(1.0 - t) - 0.5 * t ** 2
                dx[i, i + 1] += 0.5 * t ** 2
                dx[i + 1, i] += -t + 0.5 * t ** 2
                dx[i + 1, i + 1] += -0.5 * t ** 2
                dm[i] += 1.0 - t
                dm[i + 1] += t

        partials['w', 'r_R'] = dx
        partials['w', 'x_max'] = dm.reshape(nn, 1)
