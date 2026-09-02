"""
RotorChartGenerator -- reproduce the isolated rotor charts from G2.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 229-231, charts p. 254-271.

The charts plot, at one tip speed ratio per page,

    top     X = f/A_b + (sigma/mu^4)(C_T/sigma)^2   against C_T/sigma
    bottom  C_Q/sigma                               against C_T/sigma

both as families of constant collective, with the stall limit lines
alpha_1,270 = 12 and 16 deg and delta C_Q/sigma_0 = 0.004 and 0.008 drawn on
the lower one.

How the sweep runs. G2 takes lambda' and theta_0 and returns C_T/sigma, and
ChartParameterComp turns (lambda', C_T/sigma, mu) into X exactly. So a curve
of constant collective is traced by sweeping lambda', not by solving for
anything: each converged G2 point IS a chart point. Nothing is inverted.

Why this is a driver and not an OpenMDAO group. A chart page is about 180
points and G2 carries some fifty fields of num_azimuth x num_radial; running
them as one vectorised model would hand the DirectSolver eight million states.
Points are therefore run one at a time, reusing a single Problem and warm
starting each solve from the previous converged point, which is also what
makes the sweep fast: a neighbouring lambda' converges in a handful of
iterations where a cold start takes twenty.

Rotor parameters of the printed charts, p. 229:

    twist            -5 deg, linear
    airfoil          NACA 0012
    advancing tip Mach   0.7
    chord/radius     0.079
    tip loss factor  0.97

M_1,90 = 0.7 fixes the tip speed through (1 + mu) Omega R / a = 0.7, so
V_tip/V_son changes from page to page: 0.636 at mu = 0.10, 0.538 at mu = 0.30.
That is easy to get wrong -- the charts are NOT at constant tip speed.

Failures are recorded, not raised. A sweep that stops at the first
non-converged point is useless for exploring where the model breaks, so each
point carries a `converged` flag and the caller decides.
"""

import numpy as np
import openmdao.api as om

from .chart_parameter_comp import ChartParameterComp
from .disc_airfoil_group import DiscAirfoilGroup
from .numerical_rotor_group import NumericalRotorGroup

# p. 229
CHART_ROTOR = dict(theta_1=np.deg2rad(-5.0), c_R=0.079, B=0.97, x_0=0.0,
                   M_190=0.7, gamma=8.0, a=6.0, sigma=0.084883, R=30.0)

RECORDED = ('CT_sigma', 'CQ_sigma', 'CH_sigma', 'A_1', 'B_1', 'a0', 'vi_OR',
            'alpha_1270', 'CQ0_sigma_max')


class RotorChartGenerator:
    """Sweep G2 to reproduce one isolated rotor chart page."""

    def __init__(self, num_azimuth=12, num_radial=15, rotor=None,
                 group_options=None):
        self.rotor = {**CHART_ROTOR, **(rotor or {})}
        self.grid = dict(num_nodes=1, num_azimuth=num_azimuth,
                         num_radial=num_radial)

        problem = om.Problem()
        problem.model.add_subsystem(
            'g2', NumericalRotorGroup(airfoil=DiscAirfoilGroup(**self.grid),
                                      trim=True, inflow='momentum',
                                      **self.grid, **(group_options or {})),
            promotes=['*'])
        problem.model.add_subsystem(
            'chart', ChartParameterComp(mode='from_inflow'), promotes=['*'])
        problem.setup()
        problem.model.g2.nonlinear_solver.options['err_on_non_converge'] = False
        problem.final_setup()
        self.problem = problem

    def _tip_speed(self, mu):
        """M_1,90 = (1 + mu) Omega R / a fixes the tip speed at each mu."""
        return self.rotor['M_190'] / (1.0 + mu)

    def point(self, mu, theta_0, lambda_p, V_son=1116.0):
        """One chart point. theta_0 in degrees, lambda_p dimensionless."""
        p = self.problem
        for name, value in self.rotor.items():
            if name not in ('M_190',):
                p.set_val(name, value)
        p.set_val('mu', mu)
        p.set_val('V_son', V_son)
        p.set_val('V_tip', self._tip_speed(mu) * V_son)
        p.set_val('theta_0', np.deg2rad(theta_0))
        p.set_val('lambda_p', lambda_p)
        p.run_model()

        residual = max(abs(p.get_val(name)[0])
                       for name in ('res_CT', 'res_CM', 'res_CR'))
        result = {name: p.get_val(name)[0] for name in RECORDED}
        result.update(X_chart=p.get_val('X_chart')[0],
                      theta_0=theta_0, lambda_p=lambda_p, mu=mu,
                      converged=residual < 1e-8, residual=residual,
                      iterations=p.model.g2.nonlinear_solver._iter_count)
        return result

    def curve(self, mu, theta_0, lambda_p, restart=True):
        """One constant-collective curve, warm started along the sweep."""
        if restart:
            self._cold_start()
        return [self.point(mu, theta_0, value) for value in lambda_p]

    def page(self, mu, collectives, lambda_p):
        """A whole chart page: one curve per collective."""
        return {theta_0: self.curve(mu, theta_0, lambda_p)
                for theta_0 in collectives}

    def _cold_start(self):
        """Reset the solver states, so a curve does not inherit the previous
        collective's answer as its first guess."""
        p = self.problem
        p.set_val('CT_sigma_ref', 0.06)
        p.set_val('A_1', 0.0)
        p.set_val('B_1', 0.0)


def as_arrays(curve, *names):
    """Pull named fields out of a curve, keeping only converged points."""
    good = [point for point in curve if point['converged']]
    return tuple(np.array([point[name] for point in good]) for name in names)
