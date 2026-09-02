"""
BladeElementGroup -- steps 4 to 7 of the combined momentum and blade element
method, with the inflow and the airfoil model coupled.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, p. 69-70; airfoil model Chapter 6, p. 426-433.

    InflowGroup       theta_0, a --> theta, v1_Or, phi, alpha   (steps 4-6)
    AirfoilHoverGroup M, alpha    --> a, cl, cd                 (step 7)

The two groups form a cycle: the inflow equation of step 5 needs the lift curve
slope, and the airfoil model needs the angle of attack. The cycle is weak,
because `a` is a function of the local Mach number alone and in hover the Mach
number does not depend on the collective. Gauss-Seidel therefore converges in
two useful passes from any starting point, and a third confirms it.

Tip relief is deliberately not applied. Step 7, p. 70, allows reducing the Mach
number of the outer 10% of the blade, but only for two-dimensional wind tunnel
data; it states that airfoil data synthesized from whirl tower or model rig
tests already include the effect. The Chapter 6 model is synthesized from the
whirl tower tests of reference 1.1, so applying the correction here would count
it twice.

Inputs  : theta_0; b, c_R, r_R, d_theta, M
Outputs : theta, v1_Or, phi, alpha, a, cl, cd  (nn,)
"""

import openmdao.api as om

from prouty.airfoil import AirfoilHoverGroup
from prouty.hover.inflow_group import InflowGroup


class BladeElementGroup(om.Group):
    """Coupled inflow and section coefficients at each blade station."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)
        self.options.declare('twist_law', values=('linear', 'ideal'),
                             default='linear')
        self.options.declare('maxiter', types=int, default=20)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('inflow', InflowGroup(
            num_nodes=nn, twist_law=self.options['twist_law']), promotes=['*'])
        self.add_subsystem('airfoil', AirfoilHoverGroup(num_nodes=nn),
                           promotes=['*'])

        self.nonlinear_solver = om.NonlinearBlockGS(
            maxiter=self.options['maxiter'], atol=1e-12, rtol=1e-12, iprint=0)
        self.linear_solver = om.DirectSolver()
