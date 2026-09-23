"""
RotorDiscGridComp -- the (r/R, psi) grid and its quadrature weights.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 209-212.

The numerical method replaces the closed-form integrals by a double
quadrature: a radial integral along the blade, then a simple average around
the azimuth (p. 209, 211):

    delta C_T/sigma = int_{x_0}^{B} (d C_T/sigma / d(r/R)) d(r/R)
    C_T/sigma       = (1/N) sum_{n=1}^{N} delta C_T/sigma_n

Azimuth. N equally spaced stations, averaged with equal weight. For a periodic
integrand that simple average is spectrally accurate -- it is exact for every
harmonic below N -- which is why Prouty can say eight stations suffice when
retreating blade stall is not a factor (p. 211). Stall is a factor here: the
dynamic stall delay of p. 220-221 depends on d(alpha)/d(psi), evaluated as a
difference between neighbouring stations, so the answer becomes sensitive to
the spacing itself. 24 is the default.

Radius. Two integration domains coexist, and they are not the same:

  * lift and induced drag act between the root cutout and the tip loss
    station, x_0 to B (p. 200);
  * pressure drag and skin friction act over the whole blade, because "no root
    cutout or tip loss is applied to the portion of the chordwise force
    produced by pressure drag" (p. 212).

Rather than carry two grids and evaluate the velocity field twice, this
component keeps ONE fixed grid on [0, 1] and emits TWO weight vectors:
w_r_full for the full blade, w_r_lift windowed to [x_0, B].

w_r_lift uses exact partial-cell trapezoidal weights rather than a smoothed
window. The first attempt here was a smoothstep of one cell width, following
the blending used elsewhere in this package for discontinuities; it cost 2.5 %
on the zeroth moment, because at B = 1 half the ramp falls outside the grid
and the endpoint weight loses it. Integrating the linear interpolant over the
partial cells instead is exact for a piecewise-linear integrand AND
differentiable: as B crosses a node, the weight moves continuously from one
node to the next, and d(integral)/dB is just the integrand at B. Nothing is
given up.

That matters because B is not a constant: p. 199 obtains it from the hover
equation, so it moves with C_T, and an optimiser will differentiate through
it.

The weights can be checked against the closed-form coefficients of p. 200
directly, since the moments of the lift window are exactly the p_n groups:

    sum w_lift          = B   - x_0   = p1
    sum w_lift (r/R)    = (B^2 - x_0^2)/2 = p2/2
    sum w_lift (r/R)^2  = (B^3 - x_0^3)/3 = p3/3
    sum w_lift (r/R)^3  = (B^4 - x_0^4)/4 = p4/4

That is a genuine cross-check between G1 and G2 rather than an internal
consistency test, and it is what the tests assert.

    B, x_0 --> RotorDiscGridComp --> r_R, psi, w_r_full, w_r_lift
"""

import numpy as np
import openmdao.api as om


def _clip_real(z, lo, hi):
    """np.clip that decides on the real part and keeps the imaginary one.

    Plain np.clip compares complex numbers lexicographically, so a complex
    step perturbation of a limit sitting exactly on a cell boundary gets
    clipped away and check_partials reports a spurious zero derivative.
    """
    return np.where(np.real(z) < lo, lo, np.where(np.real(z) > hi, hi, z))


def _cell_weights(x_0, B, xi, h):
    """Exact trapezoidal weights for the integral from x_0 to B.

    Over each cell the integrand is taken as the linear interpolant between
    its end nodes and integrated over the part of the cell that lies inside
    [x_0, B]. With s and t the normalised ends of that overlap,

        int = h [ f_j ((t-s) - (t^2-s^2)/2) + f_{j+1} (t^2-s^2)/2 ]

    Returns the weights and, for the derivatives, the normalised positions of
    x_0 and B inside the cell that contains them.
    """
    left, right = xi[:-1], xi[1:]                     # cell boundaries
    lo = _clip_real(x_0, left, right)
    hi = _clip_real(B, left, right)
    hi = np.where(np.real(hi) < np.real(lo), lo, hi)

    s = (lo - left) / h
    t = (hi - left) / h
    half = 0.5 * (t ** 2 - s ** 2)

    w = np.zeros(B.shape[:-1] + (xi.size,), dtype=np.result_type(B, x_0))
    w[..., :-1] += h * ((t - s) - half)
    w[..., 1:] += h * half
    return w, s, t


class RotorDiscGridComp(om.ExplicitComponent):
    """Blade element grid and quadrature weights, p. 209-212."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24,
                             desc='azimuth stations, p. 211')
        self.options.declare('num_radial', types=int, default=21,
                             desc='radial stations over the full blade')

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']

        self._xi = np.linspace(0.0, 1.0, n_r)
        self._h = 1.0 / (n_r - 1)
        self._w = np.full(n_r, self._h)
        self._w[0] = self._w[-1] = 0.5 * self._h            # trapezoid on [0,1]

        self.add_input('B', shape=(nn,), val=1.0, desc='tip loss factor')
        self.add_input('x_0', val=0.0, desc='root cutout r/R')

        self.add_output('r_R', shape=(n_r,), val=self._xi,
                        desc='radial stations, fixed on [0, 1]')
        self.add_output('psi', shape=(n_psi,), units='rad',
                        val=np.arange(n_psi) * 2.0 * np.pi / n_psi,
                        desc='azimuth stations, equally spaced from 0')
        self.add_output('w_r_full', shape=(n_r,), val=self._w,
                        desc='radial weights over the whole blade, p. 212')
        self.add_output('w_r_lift', shape=(nn, n_r),
                        desc='radial weights from x_0 to B, p. 200')
        self.add_output('w_psi', val=1.0 / n_psi,
                        desc='azimuth weight, 1/N, p. 211')

        flat = np.arange(nn * n_r)
        self.declare_partials('w_r_lift', 'B', rows=flat,
                              cols=np.repeat(np.arange(nn), n_r))
        self.declare_partials('w_r_lift', 'x_0', rows=flat,
                              cols=np.zeros(nn * n_r, dtype=int))

    def compute(self, inputs, outputs):
        n_psi = self.options['num_azimuth']
        B = inputs['B'][:, np.newaxis]

        outputs['r_R'] = self._xi
        outputs['psi'] = np.arange(n_psi) * 2.0 * np.pi / n_psi
        outputs['w_r_full'] = self._w
        outputs['w_psi'] = 1.0 / n_psi
        outputs['w_r_lift'] = _cell_weights(inputs['x_0'][0], B,
                                            self._xi, self._h)[0]

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        n_r = self.options['num_radial']
        B, x_0 = inputs['B'][:, np.newaxis], inputs['x_0'][0]
        h, xi = self._h, self._xi
        left, right = xi[:-1], xi[1:]

        _, s, t = _cell_weights(x_0, B, xi, h)
        # Only the cell containing the limit moves the weights. The intervals
        # are half open so that B = 1 and x_0 = 0, which sit exactly on the
        # outer nodes, still land in a cell; a symmetric test would silently
        # return a zero derivative there.
        live_B = ((B > left) & (B <= right)).astype(float)
        live_0 = ((x_0 >= left) & (x_0 < right)).astype(float)

        d_B = np.zeros((nn, n_r))
        d_B[:, :-1] += live_B * (1.0 - t)
        d_B[:, 1:] += live_B * t

        d_0 = np.zeros((nn, n_r))
        d_0[:, :-1] -= live_0 * (1.0 - s)
        d_0[:, 1:] -= live_0 * s

        partials['w_r_lift', 'B'] = d_B.ravel()
        partials['w_r_lift', 'x_0'] = d_0.ravel()
