"""
HoverRotorGroup -- the complete combined momentum and blade element method.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, "Combined Momentum and Blade Element Theory with Empirical
Corrections", steps 1 to 21, p. 69-72.

    G0  RotorPreprocessGroup        steps 1 to 3
    G1  BladeElementGroup           steps 4 to 7   (inflow / airfoil cycle)
    G3  ThrustGroup                 steps 8 to 11
    G4  TorqueGroup                 steps 12 to 15
    G5  EmpiricalCorrectionsGroup   steps 16 to 20
    G6  RotorPerformanceGroup       step 21

Two modes.

  'analysis'  theta_0 is an input, thrust comes out. This is the book's own
              procedure and the default.

  'trim'      T_target is an input, theta_0 becomes an implicit state solved by
              a BalanceComp so that T matches. Useful for sweeps without an
              optimiser, for a tail rotor whose thrust is set by yaw balance,
              and whenever this rotor sits inside a larger model.

For gradient based design there is a third option that needs neither mode: keep
'analysis', declare theta_0 as a design variable and T = T_target as an equality
constraint, and let the optimiser resolve the trim in the same space as
everything else. That is usually faster than a Newton solver nested inside every
optimiser iteration.

Trim solver notes. Thrust is monotone in collective over the whole useful range
and dT/dtheta_0 is about 2,500 lb/deg for the example helicopter, so Newton
converges in three iterations from any sensible start.

Two limits found in validation, both worth knowing before trusting a sweep.

First, the starting collective and the lower bound must keep the blade pitch
positive at every station, which for linear twist about the centre means
theta_0 > -theta_1. Below that, the closed form inflow of step 5 has a negative
radicand, InflowRatioComp floors it, and the derivative it returns is zero;
Newton then cannot move and stalls at the starting point. The default bounds
are deliberately wide, so set theta_0_bounds for the rotor at hand. The default
start of 17.5 deg is inside the valid region for the example helicopter.

Second, check_totals with method='fd' does not reconverge the Newton when the
solver sits on a subgroup, so it reports the derivative at frozen theta_0 and
disagrees with the analytic one by a wide margin. The analytic totals are the
correct ones: a manual central difference over two fully retrimmed solutions
reproduces them to five figures. Verify trim derivatives that way, not with
check_totals.

The Newton sits above the NonlinearBlockGS that closes the inflow and airfoil
cycle inside G1; the inner solver is cheap because the lift curve slope depends
on Mach number alone.

    theta_0 or T_target, plus geometry and atmosphere
        --> HoverRotorGroup --> CT, CQ, T, power_hp, Q, FM, power_loading
"""

import openmdao.api as om

from prouty.hover.rotor_preprocess_group import RotorPreprocessGroup
from prouty.hover.blade_element_group import BladeElementGroup
from prouty.hover.thrust_group import ThrustGroup
from prouty.hover.torque_group import TorqueGroup
from prouty.hover.empirical_corrections_group import EmpiricalCorrectionsGroup
from prouty.hover.rotor_performance_group import RotorPerformanceGroup


class HoverRotorGroup(om.Group):
    """A rotor in hover, from dimensional geometry to thrust and power."""

    def initialize(self):
        self.options.declare('num_elements', types=int, default=10,
                             desc='blade elements; p. 69 recommends 5 to 15')
        self.options.declare('mode', values=('analysis', 'trim'),
                             default='analysis')
        self.options.declare('distribution', values=('uniform', 'cosine'),
                             default='uniform')
        self.options.declare('twist_law', values=('linear', 'ideal'),
                             default='linear')
        self.options.declare('twist_reference', values=('center', 'cutout'),
                             default='center')
        self.options.declare('tip_loss_model',
                             values=('effective_radius', 'general'),
                             default='effective_radius')
        self.options.declare('swirl_curve',
                             values=('approximate', 'wu', 'durand_glauert'),
                             default='approximate')
        self.options.declare('solidity',
                             values=('geometric', 'thrust_weighted'),
                             default='geometric')
        self.options.declare('theta_0_bounds', types=tuple, default=(0.0, 25.0),
                             desc='collective bounds used in trim mode, deg')

    def setup(self):
        ne = self.options['num_elements']
        nn = ne + 1

        self.add_subsystem('pre', RotorPreprocessGroup(
            num_elements=ne, distribution=self.options['distribution'],
            twist_reference=self.options['twist_reference']), promotes=['*'])

        self.add_subsystem('blade_element', BladeElementGroup(
            num_nodes=nn, twist_law=self.options['twist_law']), promotes=['*'])

        self.add_subsystem('thrust', ThrustGroup(
            num_nodes=nn,
            tip_loss_model=self.options['tip_loss_model']), promotes=['*'])

        self.add_subsystem('torque', TorqueGroup(num_nodes=nn), promotes=['*'])

        self.add_subsystem('empirical', EmpiricalCorrectionsGroup(
            swirl_curve=self.options['swirl_curve'],
            solidity=self.options['solidity']), promotes=['*'])

        self.add_subsystem('performance', RotorPerformanceGroup(), promotes=['*'])

        if self.options['mode'] == 'trim':
            self._add_trim()

    def _add_trim(self):
        """Solve theta_0 so that the rotor thrust matches T_target."""
        lower, upper = self.options['theta_0_bounds']

        balance = om.BalanceComp()
        balance.add_balance('theta_0', val=17.5, units='deg',
                            lhs_name='T', rhs_name='T_target',
                            eq_units='lbf', lower=lower, upper=upper)
        self.add_subsystem('trim', balance, promotes=['*'])

        self.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True, maxiter=30, atol=1e-8, rtol=1e-10, iprint=0)
        self.nonlinear_solver.linesearch = om.BoundsEnforceLS(
            bound_enforcement='vector', iprint=0)
        self.linear_solver = om.DirectSolver()
