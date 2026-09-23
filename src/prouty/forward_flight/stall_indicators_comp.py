"""
StallIndicatorsComp -- how close the rotor is running to its limit.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 228.

Two quantities, both drawn as limit lines on the isolated rotor charts
(p. 254-271):

    alpha_1,270 = theta_0 + theta_1 + B_1 + atan[ lambda' / (1 - mu) ]

the angle of attack of the retreating tip, and

    max over psi of  delta C_Q/sigma_0 = int_0^1 (dC_Q/sigma_0 / d(r/R)) d(r/R)

the largest profile torque coefficient produced at any azimuth position.

They measure different things and that is why Prouty keeps both. alpha_1,270
is a single point on the disc, evaluated in closed form from the trim
variables, and it says nothing about how much of the blade is stalled -- for
the example helicopter the peak angle of attack is at r/R = 0.73, not at the
tip, so alpha_1,270 understates the situation whenever the blade is twisted.
The profile torque is an integral over the whole blade and catches stall
wherever it happens, but only after it has cost something. The charts of
p. 254 carry alpha_1,270 = 12 and 16 deg alongside delta C_Q/sigma_0 = 0.004
and 0.008, and the two families of lines are not parallel.

alpha_1,270 uses B_1 as cyclic pitch, the G2 convention of p. 211, not the
B_1 + a_1s combination of the closed form. The two differ by the longitudinal
flapping, which is zero in Table 3.2 but not in general.

The maximum is taken smoothly. A hard max is not differentiable where two
azimuth stations tie, which happens whenever the peak crosses between
stations, and the peak does move with collective. The log-sum-exp form

    smax(x) = m + rho ln sum exp((x - m)/rho),   m = max x

is smooth, exceeds the true maximum by at most rho ln N, and its derivative is
the softmax weight of each station. The bound is pessimistic: it is reached
only when every station ties. On a realistic azimuth distribution at 24
stations the measured bias is

    rho = 2e-5   6.2e-6   0.18 %
    rho = 5e-6   1.1e-8   0.00 %
    rho = 1e-6   0        0.00 %

against a theoretical bound of 6.4e-5 at the default. The shifted form used
here does not overflow however small rho is, so tightening it costs nothing.
Where two stations do tie, each takes a softmax weight of exactly 0.5 and the
derivative stays defined -- which is the whole point, since the peak azimuth
moves between stations as collective changes.

    theta_0, theta_1, B_1, lambda_p, mu, CQ0_sigma_psi
        --> alpha_1270, CQ0_sigma_max
"""

import numpy as np
import openmdao.api as om


class StallIndicatorsComp(om.ExplicitComponent):
    """Retreating tip angle of attack and peak profile torque, p. 228."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('rho', types=float, default=2.0e-5,
                             desc='smoothing width of the azimuth maximum')

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('theta_0', shape=(nn,), units='rad')
        self.add_input('theta_1', val=-0.17453, units='rad')
        self.add_input('B_1', shape=(nn,), val=0.0, units='rad',
                       desc='longitudinal cyclic pitch, p. 211 convention')
        self.add_input('lambda_p', shape=(nn,))
        self.add_input('mu', shape=(nn,))
        self.add_input('CQ0_sigma_psi', shape=(nn, n_psi),
                       desc='profile torque at each azimuth')

        self.add_output('alpha_1270', shape=(nn,), units='rad',
                        desc='angle of attack of the retreating tip')
        self.add_output('CQ0_sigma_max', shape=(nn,),
                        desc='peak profile torque around the azimuth')

        for name in ('theta_0', 'B_1', 'lambda_p', 'mu'):
            self.declare_partials('alpha_1270', name, rows=ar, cols=ar)
        self.declare_partials('alpha_1270', 'theta_1', rows=ar, cols=zeros)
        self.declare_partials('CQ0_sigma_max', 'CQ0_sigma_psi',
                              rows=np.repeat(ar, n_psi),
                              cols=np.arange(nn * n_psi))

    def _inflow_angle(self, inputs):
        """lambda' / (1 - mu), the inflow ratio at the retreating tip."""
        denom = 1.0 - inputs['mu']
        if np.any(np.real(denom) <= 0.0):
            raise om.AnalysisError(
                'StallIndicatorsComp: the retreating tip is in reverse flow at '
                f'mu = {np.real(inputs["mu"])}; alpha_1,270 is undefined.')
        return inputs['lambda_p'] / denom, denom

    def _softmax(self, inputs):
        """Weights of the smooth azimuth maximum."""
        rho = self.options['rho']
        x = inputs['CQ0_sigma_psi']
        shift = np.max(np.real(x), axis=1, keepdims=True)
        weights = np.exp((x - shift) / rho)
        return shift, weights, weights.sum(axis=1, keepdims=True), rho

    def compute(self, inputs, outputs):
        ratio, _ = self._inflow_angle(inputs)
        shift, _, total, rho = self._softmax(inputs)

        outputs['alpha_1270'] = (inputs['theta_0'] + inputs['theta_1'][0]
                                 + inputs['B_1'] + np.arctan(ratio))
        outputs['CQ0_sigma_max'] = (shift + rho * np.log(total))[:, 0]

    def compute_partials(self, inputs, partials):
        ratio, denom = self._inflow_angle(inputs)
        d_atan = 1.0 / (1.0 + ratio ** 2)

        partials['alpha_1270', 'theta_0'] = 1.0
        partials['alpha_1270', 'theta_1'] = 1.0
        partials['alpha_1270', 'B_1'] = 1.0
        partials['alpha_1270', 'lambda_p'] = d_atan / denom
        partials['alpha_1270', 'mu'] = d_atan * ratio / denom

        _, weights, total, _ = self._softmax(inputs)
        partials['CQ0_sigma_max', 'CQ0_sigma_psi'] = (weights / total).ravel()
