"""
FuselageAngleComp -- angle of attack of the fuselage.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 192.

    alpha_F = alpha_TPP - i_s - a_1s - delta_alpha_DW
    delta_alpha_DW = v1 / V                                    radians
    alpha_F = lambda'/mu - i_s - a_1s                          radians

The fuselage does not see the free stream. It sees it turned downward by the
rotor wake, and it is bolted to a shaft that is itself tilted relative to the
tip path plane. Three corrections therefore separate alpha_TPP from alpha_F:

  * i_s, the built-in shaft incidence, a fixed forward tilt of the mast
    relative to the fuselage reference line. Zero for Prouty's example
    helicopter (Appendix A p. 669), which is why it never shows in Table 3.3,
    but two to five degrees on most real machines.
  * a_1s, the longitudinal flapping, which tilts the tip path plane relative
    to the shaft. Zero in Table 3.2 by assumption, valid for performance work
    but not for stability and control.
  * the downwash angle. Prouty takes it as v1/V, citing the wind tunnel
    measurements of reference 3.31, which found the downwash at the fuselage
    to be about the momentum value at the rotor disc.

The three collapse neatly: alpha_TPP - v1/V is exactly lambda'/mu, so the
whole correction chain reduces to one division. That is the form coded here,
which also means alpha_F carries the inflow ratio's own accuracy rather than
compounding two separate estimates.

Singular at mu = 0, as the downwash angle must be: in hover the fuselage sees
pure downwash and its angle of attack is undefined. This component belongs to
the forward flight branch only.

Why it matters. alpha_F is what drives the Appendix A curves, so an error here
propagates straight into L_F and D_F and therefore back into alpha_TPP -- this
is the variable the trim loop of p. 193 actually iterates on. The lift slope
being 1.9 ft^2 per degree, one degree of alpha_F is 86 lb of fuselage lift at
115 knots.

Both directions are provided. The trim loop of p. 193 carries alpha_F as its
iteration variable, so it needs alpha_F -> lambda' on the way in and
lambda' -> alpha_F on the way out; having one component do both guarantees the
two stay exact inverses.

    lambda_p, mu, i_s, a1s --> 'from_inflow' --> alpha_F   (nn,)
    alpha_F,  mu, i_s, a1s --> 'to_inflow'   --> lambda_p  (nn,)
"""

import numpy as np
import openmdao.api as om


class FuselageAngleComp(om.ExplicitComponent):
    """Fuselage angle of attack, p. 192."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', values=('from_inflow', 'to_inflow'),
                             default='from_inflow')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.solve_for = ('alpha_F' if self.options['mode'] == 'from_inflow'
                          else 'lambda_p')
        self.given = 'lambda_p' if self.solve_for == 'alpha_F' else 'alpha_F'

        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('a1s', shape=(nn,), val=0.0, units='rad',
                       desc='longitudinal flapping')
        self.add_input('vi_OR', shape=(nn,), val=0.0, desc='v1 / (Omega R)')
        self.add_input('i_s', val=0.0, units='rad',
                       desc='built-in shaft incidence, zero in Appendix A')

        units = {'alpha_F': 'rad', 'lambda_p': None}
        self.add_input(self.given, shape=(nn,), units=units[self.given])
        self.add_output(self.solve_for, shape=(nn,), units=units[self.solve_for])
        self.add_output('alpha_DW', shape=(nn,), units='rad',
                        desc='downwash angle at the fuselage, v1/V')

        out = self.solve_for
        for name in (self.given, 'mu', 'a1s'):
            self.declare_partials(out, name, rows=ar, cols=ar)
        self.declare_partials(out, 'i_s', rows=ar, cols=np.zeros(nn, dtype=int))
        for name in ('vi_OR', 'mu'):
            self.declare_partials('alpha_DW', name, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        mu = inputs['mu']
        if np.any(np.real(mu) <= 0.0):
            raise om.AnalysisError('FuselageAngleComp: alpha_F is undefined at '
                                   f'mu = 0, got mu = {np.real(mu)}.')

        offset = inputs['i_s'][0] + inputs['a1s']
        if self.solve_for == 'alpha_F':
            outputs['alpha_F'] = inputs['lambda_p'] / mu - offset
        else:
            outputs['lambda_p'] = mu * (inputs['alpha_F'] + offset)
        outputs['alpha_DW'] = inputs['vi_OR'] / mu

    def compute_partials(self, inputs, partials):
        mu = inputs['mu']

        out = self.solve_for
        if out == 'alpha_F':
            partials[out, 'lambda_p'] = 1.0 / mu
            partials[out, 'mu'] = -inputs['lambda_p'] / mu ** 2
            partials[out, 'a1s'] = -np.ones_like(mu)
            partials[out, 'i_s'] = -np.ones_like(mu)
        else:
            offset = inputs['i_s'][0] + inputs['a1s']
            partials[out, 'alpha_F'] = mu
            partials[out, 'mu'] = inputs['alpha_F'] + offset
            partials[out, 'a1s'] = mu
            partials[out, 'i_s'] = mu

        partials['alpha_DW', 'vi_OR'] = 1.0 / mu
        partials['alpha_DW', 'mu'] = -inputs['vi_OR'] / mu ** 2
