"""
NormalForceComp -- normal force coefficient of a blade element.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 209.

    c_N = c_l (U_T / U_B) + c_d (U_P / U_B)

This is where the numerical method parts company with the closed form. p. 209:
"the procedure used in deriving the closed-form equations will be modified
somewhat by resolving both the lift and drag forces into a normal -- or
thrust -- force, where before only lift was assumed to contribute to thrust."
The drag term is small in attached flow, U_P being two orders below U_T at the
tips, but it is not small near the reverse flow boundary, and it is what makes
the method valid there.

Two outputs, and the second is the one to use. c_N itself carries 1/U_B, which
reaches 110 on the disc at mu = 0.3 because U_B falls to 0.009 at the root on
the fore-aft line. Every loading that consumes c_N multiplies it back by
U_B^2:

    d(C_T/sigma)/d(r/R) = (U_B^2 / 2) c_N                    p. 209
    d(C_M/sigma)/d(r/R) = -(U_B^2 / 2)(r/R) cos(psi) c_N     p. 211
    d(C_R/sigma)/d(r/R) = -(U_B^2 / 2)(r/R) sin(psi) c_N     p. 211

so the product U_B^2 c_N = U_B (c_l U_T + c_d U_P) is what actually enters,
and it is regular: one power of U_B cancels analytically instead of being
divided out and multiplied back numerically. cN_UB2 is that product. c_N is
exported alongside because Prouty's equations are written in it and because it
is the natural quantity to plot, not because the loadings should be rebuilt
from it.

Sign conventions come for free. Prouty (p. 214): "when the airfoil
characteristics are used as a function of alpha with the signs as shown, the
normal and chordwise force equations will automatically resolve the lift and
drag components into forces with the correct orientation." Inside the reverse
flow circle U_T is negative and the quadrant rule of AlphaComp has already put
alpha in the third or fourth quadrant, so c_l is negative there; the product
c_l U_T comes out positive, which is correct -- a reversed element still
pushes the same way if its angle of attack has reversed with it.

    cl, cd, UT_bar, UP_bar, UB_bar --> NormalForceComp --> cN, cN_UB2
"""

import numpy as np
import openmdao.api as om


class NormalForceComp(om.ExplicitComponent):
    """Normal force coefficient and the regular product U_B^2 c_N, p. 209."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)
        rows = np.arange(nn * n_psi * n_r)

        for name in ('cl', 'cd', 'UT_bar', 'UP_bar', 'UB_bar'):
            self.add_input(name, shape=field)

        self.add_output('cN', shape=field, desc='normal force coefficient')
        self.add_output('cN_UB2', shape=field,
                        desc='U_B^2 c_N, the form the loadings use')

        for out in ('cN', 'cN_UB2'):
            for name in ('cl', 'cd', 'UT_bar', 'UP_bar', 'UB_bar'):
                self.declare_partials(out, name, rows=rows, cols=rows)

    def compute(self, inputs, outputs):
        force = inputs['cl'] * inputs['UT_bar'] + inputs['cd'] * inputs['UP_bar']

        outputs['cN'] = force / inputs['UB_bar']
        outputs['cN_UB2'] = force * inputs['UB_bar']

    def compute_partials(self, inputs, partials):
        cl, cd = inputs['cl'], inputs['cd']
        UT, UP, UB = inputs['UT_bar'], inputs['UP_bar'], inputs['UB_bar']
        force = cl * UT + cd * UP

        partials['cN', 'cl'] = (UT / UB).ravel()
        partials['cN', 'cd'] = (UP / UB).ravel()
        partials['cN', 'UT_bar'] = (cl / UB).ravel()
        partials['cN', 'UP_bar'] = (cd / UB).ravel()
        partials['cN', 'UB_bar'] = (-force / UB ** 2).ravel()

        partials['cN_UB2', 'cl'] = (UT * UB).ravel()
        partials['cN_UB2', 'cd'] = (UP * UB).ravel()
        partials['cN_UB2', 'UT_bar'] = (cl * UB).ravel()
        partials['cN_UB2', 'UP_bar'] = (cd * UB).ravel()
        partials['cN_UB2', 'UB_bar'] = force.ravel()
