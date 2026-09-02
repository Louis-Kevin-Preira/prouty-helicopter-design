"""
TailRotorTrimComp -- collective and flapping of a tail rotor with delta-three.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 187-191, and Table 3.1 p. 191.

A tail rotor is a flapping rotor with no cyclic pitch, so its disc tilts freely
and the shaft angle -- not the tip path plane angle -- is what is known. Most
tail rotors also carry a mechanical pitch-flap coupling, delta_three, through a
slanted flapping hinge (Figure 3.45): flapping up feathers the blade nose down,

    delta_theta = delta_beta tan(delta_3)

so the pitch distribution becomes (p. 187)

    theta = (theta_0 + a_0 tan d3) + theta_1 r/R
            - a_1s tan(d3) cos psi - b_1s tan(d3) sin psi

The last two terms have the form of cyclic pitch, which is why the coupling
folds into the ordinary rotor equations as A_1eff = a_1s tan(d3) and
B_1eff = b_1s tan(d3), with no new integration.

The three governing equations (p. 187-188) are

  thrust   C_T/sigma = (a/4)[ (2/3 + mu^2)(theta_0 + a_0 tan d3)
                            + (1/2 + mu^2/2) theta_1
                            - mu b_1s tan d3 + lambda ]
  long.    a_1s + b_1s tan d3 = k3 [ (8/3)(theta_0 + a_0 tan d3)
                                   + 2 theta_1 + 2(mu a_1s + lambda) ]
  lat.     b_1s - a_1s tan d3 = [ (4/3) mu a_0 + v1/OmegaR ] / (1 + mu^2/2)

with k3 = mu / (1 + 1.5 mu^2). They are LINEAR in the three unknowns theta_0,
a_1s and b_1s -- a_1s appearing on both sides of the second one is the only
coupling -- so this component assembles the 3x3 system and solves it directly.
That is exact and avoids transcribing the explicit solution printed on p. 189,
which spans a third of the page and is easy to mistype. Derivatives follow from
dx = A^-1 (db - dA x).

For a shaft perpendicular to the flight path, lambda = -v1/OmegaR (p. 189);
alpha_s is kept as an input so a deliberately tilted shaft, or sideslip, can be
handled through lambda = mu alpha_s - v1/OmegaR.

Two outputs exist purely for reuse downstream. Prouty's tail rotor H-force
equation (p. 189) is the main rotor equation of p. 176 with theta_0 replaced by
theta_0 + a_0 tan d3 and lambda' replaced by lambda + mu a_1s. Feeding
theta_0_eff and lambda_eff into HForceCoefComp(form='direct') therefore
reproduces it exactly, and C_Q/sigma_T follows from the same power identity as
the main rotor. No separate tail rotor force components are needed.

Prouty's own caution, p. 189: flapping from these closed-form equations can be
up to 50 % below what the chart methods give, though the trends hold. And
p. 190: the effect of delta_three on trim is small enough to neglect for
performance -- Table 3.1 moves theta_0 by 0.05 deg between d3 = 0 and 30 deg.
Its real value is in reducing flapping during manoeuvres.

    mu, CT_sigma, theta_1, a, a0, vi_OR, delta_3, alpha_s
        --> theta_0, a1s, b1s, theta_0_eff, lambda_eff
"""

import numpy as np
import openmdao.api as om

OUTPUTS = ('theta_0', 'a1s', 'b1s')
INPUTS = ('mu', 'CT_sigma', 'a0', 'vi_OR', 'alpha_s',
          'theta_1', 'a', 'delta_3')


class TailRotorTrimComp(om.ExplicitComponent):
    """Tail rotor collective and flapping with delta-three coupling."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('mu', shape=(nn,), desc='tail rotor tip speed ratio')
        self.add_input('CT_sigma', shape=(nn,), desc='C_T/sigma of the tail rotor')
        self.add_input('a0', shape=(nn,), units='rad', desc='coning angle')
        self.add_input('vi_OR', shape=(nn,), desc='v1 / (Omega R)')
        self.add_input('alpha_s', shape=(nn,), val=0.0, units='rad',
                       desc='shaft angle of attack, zero for an upright shaft')
        self.add_input('theta_1', val=-0.08727, units='rad', desc='blade twist')
        self.add_input('a', val=6.0, units='1/rad', desc='lift curve slope')
        self.add_input('delta_3', val=0.0, units='rad',
                       desc='pitch-flap coupling angle')

        self.add_output('theta_0', shape=(nn,), units='rad', desc='collective')
        self.add_output('a1s', shape=(nn,), units='rad',
                        desc='longitudinal flapping')
        self.add_output('b1s', shape=(nn,), units='rad', desc='lateral flapping')
        self.add_output('theta_0_eff', shape=(nn,), units='rad',
                        desc='theta_0 + a_0 tan(delta_3), for the force equations')
        self.add_output('lambda_eff', shape=(nn,), units=None,
                        desc="lambda + mu a_1s, the tail rotor's lambda'")

        for out in OUTPUTS + ('theta_0_eff', 'lambda_eff'):
            for name in ('mu', 'CT_sigma', 'a0', 'vi_OR', 'alpha_s'):
                self.declare_partials(out, name, rows=ar, cols=ar)
            for name in ('theta_1', 'a', 'delta_3'):
                self.declare_partials(out, name, rows=ar, cols=zeros)

    def _system(self, v):
        """A and b of A [theta_0, a_1s, b_1s]^T = b, p. 187-188.

        `v` is a plain dict of arrays, not the OpenMDAO input vector, so that
        compute_partials can perturb it.
        """
        nn = self.options['num_nodes']
        mu, mu2 = v['mu'], v['mu'] ** 2
        a, theta_1 = v['a'][0], v['theta_1'][0]
        t = np.tan(v['delta_3'][0])
        a0, vi = v['a0'], v['vi_OR']
        lam = mu * v['alpha_s'] - vi

        k1 = 2.0 / 3.0 + mu2
        k2 = 0.5 * (1.0 + mu2)
        k3 = mu / (1.0 + 1.5 * mu2)

        dtype = (complex if any(np.iscomplexobj(v[k]) for k in INPUTS)
                 else float)
        A = np.zeros((nn, 3, 3), dtype=dtype)
        b = np.zeros((nn, 3), dtype=dtype)

        A[:, 0, 0] = k1
        A[:, 0, 2] = -mu * t
        b[:, 0] = 4.0 / a * v['CT_sigma'] - k1 * a0 * t - k2 * theta_1 - lam

        A[:, 1, 0] = -(8.0 / 3.0) * k3
        A[:, 1, 1] = 1.0 - 2.0 * mu * k3
        A[:, 1, 2] = t
        b[:, 1] = k3 * ((8.0 / 3.0) * a0 * t + 2.0 * theta_1 + 2.0 * lam)

        A[:, 2, 1] = -t
        A[:, 2, 2] = 1.0
        b[:, 2] = ((4.0 / 3.0) * mu * a0 + vi) / (1.0 + 0.5 * mu2)

        return A, b, t, lam

    def _values(self, inputs):
        return {k: np.array(inputs[k]) for k in INPUTS}

    def compute(self, inputs, outputs):
        A, b, t, _ = self._system(self._values(inputs))
        x = np.linalg.solve(A, b[:, :, None])[:, :, 0]

        for k, name in enumerate(OUTPUTS):
            outputs[name] = x[:, k]
        outputs['theta_0_eff'] = x[:, 0] + inputs['a0'] * t
        outputs['lambda_eff'] = (inputs['mu'] * inputs['alpha_s']
                                 - inputs['vi_OR'] + inputs['mu'] * x[:, 1])

    def compute_partials(self, inputs, partials):
        """dx = A^-1 (db - dA x). The assembly is differentiated by complex
        step, which is exact; the linear solve itself stays analytic."""
        v = self._values(inputs)
        A, b, t, _ = self._system(v)
        x = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        nn = x.shape[0]
        mu, a0, a1s = v['mu'], v['a0'], x[:, 1]

        step = 1e-30
        for name in INPUTS:
            base = v[name]
            for k in range(base.size):
                pert = dict(v)
                col = np.array(base, dtype=complex)
                col[k] += 1j * step
                pert[name] = col

                dA, db, _, _ = self._system(pert)
                rhs = np.imag(db) / step - np.einsum(
                    'nij,nj->ni', np.imag(dA) / step, x)
                dx = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]

                # a scalar input feeds every node; a vectorised one only
                # feeds node k, so write that single entry
                sel = np.arange(nn) if base.size == 1 else np.array([k])

                d_t = (1.0 + t ** 2) if name == 'delta_3' else 0.0
                d_a0 = 1.0 if name == 'a0' else 0.0
                d_lam = np.zeros(sel.size)
                if name == 'mu':
                    d_lam = v['alpha_s'][sel] + a1s[sel]
                elif name == 'alpha_s':
                    d_lam = mu[sel]
                elif name == 'vi_OR':
                    d_lam = -np.ones(sel.size)

                block = {out: dx[sel, j] for j, out in enumerate(OUTPUTS)}
                block['theta_0_eff'] = dx[sel, 0] + d_a0 * t + a0[sel] * d_t
                block['lambda_eff'] = d_lam + mu[sel] * dx[sel, 1]
                for out, value in block.items():
                    partials[out, name][sel] = value
