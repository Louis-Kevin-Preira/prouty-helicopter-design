"""
MissionIntegralComp -- range or endurance by integrating over gross weight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4: range p. 325, Figures 4.44 and 4.45; endurance p. 331,
Figure 4.47.

"The distance traveled at the speed for 99 % maximum specific range while
burning a given amount of fuel can be found by integrating the area under the
specific range curves of Figure 4.44":

    Range     = int_{G.W._end}^{G.W._start} (S.R.) dG.W.        n.mi.
    Endurance = int_{G.W._end}^{G.W._start} (S.E.) dG.W.        hr

The helicopter gets lighter as it burns fuel, so the integrand is evaluated at
a set of gross weights between the landing and the takeoff weight and the area
is taken by the trapezoidal rule, or by Simpson's rule on an odd number of
evenly spaced nodes. The nodes themselves may be design variables, so the
derivatives with respect to G.W. are carried too.

kind
    'range'      SR in n.mi./lb  --> range in n.mi.
    'endurance'  SE in hr/lb     --> endurance in hr

    GW (nn,), SR or SE (nn,) --> range or endurance
"""

import numpy as np
import openmdao.api as om

KINDS = {'range': ('SR', 'NM/lbm', 'range', 'NM'),
         'endurance': ('SE', 'h/lbm', 'endurance', 'h')}


class MissionIntegralComp(om.ExplicitComponent):
    """Area under the specific range or specific endurance curve, pp. 325-331."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=3)
        self.options.declare('kind', default='range', values=tuple(KINDS))
        self.options.declare('rule', default='trapezoid', values=('trapezoid', 'simpson'))

    def setup(self):
        nn = self.options['num_nodes']
        y_name, y_units, out_name, out_units = KINDS[self.options['kind']]
        self._names = (y_name, out_name)

        if self.options['rule'] == 'simpson' and (nn < 3 or nn % 2 == 0):
            raise ValueError("Simpson's rule needs an odd number of at least 3 nodes")

        self.add_input('GW', val=np.linspace(1.0, 2.0, nn), units='lbm',
                       desc='gross weights from landing to takeoff weight')
        self.add_input(y_name, val=np.zeros(nn), units=y_units,
                       desc='integrand at each gross weight')
        self.add_output(out_name, val=0.0, units=out_units)
        self.declare_partials(out_name, ['GW', y_name])

    def _weights(self, GW):
        """Quadrature weights, and their derivatives with respect to the nodes."""
        nn = GW.size
        w = np.zeros_like(GW)                       # complex safe under complex step
        dw = np.zeros((nn, nn))                     # dw_i / dGW_j
        if self.options['rule'] == 'trapezoid':
            h = np.diff(GW)
            for i, hi in enumerate(h):
                w[i] += 0.5 * hi
                w[i + 1] += 0.5 * hi
                dw[i, i + 1] += 0.5
                dw[i, i] -= 0.5
                dw[i + 1, i + 1] += 0.5
                dw[i + 1, i] -= 0.5
        else:
            span = GW[-1] - GW[0]
            h = span / (nn - 1)
            coef = np.ones(nn)
            coef[1:-1:2] = 4.0
            coef[2:-1:2] = 2.0
            w = coef * h / 3.0
            dw[:, -1] = coef / (3.0 * (nn - 1))
            dw[:, 0] = -coef / (3.0 * (nn - 1))
        return w, dw

    def compute(self, inputs, outputs):
        y_name, out_name = self._names
        w = self._weights(inputs['GW'])[0]
        outputs[out_name] = np.dot(w, inputs[y_name])

    def compute_partials(self, inputs, partials):
        y_name, out_name = self._names
        y = inputs[y_name]
        w, dw = self._weights(inputs['GW'])
        partials[out_name, y_name] = w
        partials[out_name, 'GW'] = y @ dw
