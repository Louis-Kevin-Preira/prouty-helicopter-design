"""
StallDelayComp -- dynamic stall delay from the blade's pitch rate.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 220-221.

An element whose angle of attack is rising fast does not stall at the static
stall angle; separation is delayed until some angle above it, the lift
overshoot of Chapter 6. Prouty uses the empirical form of references 3.34 and
3.41, the one that produced the isolated rotor charts of this chapter:

    delta_alpha_stall = gamma sqrt| (c/2) alpha_dot / V | sign(alpha_dot)  deg
    gamma = 1.76 ln(0.6 / M)

and then rewrites it in terms of the change of angle of attack between
neighbouring azimuth stations (p. 221):

    delta_alpha_stall = gamma sqrt[ (c/R)(1/4 pi)(1/U_B) |d_alpha| /
                                    (d_psi/360) ]      deg

with d_alpha and d_psi both in degrees. Writing the bracket as the azimuthal
rate d(alpha)/d(psi), which is the same number in any consistent angular unit,
collapses it to

    delta_alpha_stall = gamma sqrt[ (90/pi)(c/R) |rate| / U_B ] sign(rate)

Where it goes. p. 221: "this angle increment should be added to the static
stall angle in whatever type of airfoil data presentation is used. If the
equation form developed in Chapter 6 is used, delta_alpha_stall is simply
added to alpha_L after it has been modified for sweep effects." So the
combined stall angle is alpha_L0 sec(Lambda) + delta_alpha_stall, and this
component supplies the second term. For tabulated data the same information
enters as alpha_ref = alpha cos(Lambda) - delta_alpha_stall.

gamma changes sign at M = 0.6, and that is the physics, not a bug. Above
Mach 0.6 the logarithm turns negative and the stall angle is REDUCED: shocks
promote separation, so a fast-pitching element at the advancing tip stalls
earlier than a static one, not later. At the advancing tip of the example
helicopter, M = 0.757 gives gamma = -0.41; on the retreating side, M = 0.41
gives gamma = +0.67.

Magnitudes, measured on the example helicopter at mu = 0.3. Outboard of
r/R = 0.4 the delay stays between -8.5 and +4.5 deg, and it is a few tenths of
a degree at both tips -- +0.06 at the advancing tip, -0.10 at the retreating
one. Near the reverse flow boundary it reaches -200 to +60 deg. That is not a
coding error; p. 221 says so in as many words: "along the boundary of the
reverse-flow region, the calculated angle of attack, the sweep, and the rate
of change of angle of attack on the blade element, are very high. This
combination can result in the equations predicting unreasonably high -- or
low -- lift coefficients." His remedy is not to cap this term but to bound the
lift coefficient itself, and that is what the next component does. Without
those bounds the delay computed here is meaningless in that region.

Three numerical points, all of which bite.

The azimuthal difference must be taken modulo 360 deg. alpha jumps by exactly
that as an element crosses the reverse flow boundary, by construction of the
quadrant rule in AlphaComp. Read literally, the largest rate on the disc is
9.8 deg per deg of azimuth; unwrapped it is 5.4, and every point above the
unwrapped maximum is an artefact of the branch cut, not a physical pitch
rate.

sqrt(|rate|) has an infinite derivative where the rate vanishes, which happens
on every azimuthal extremum of alpha -- a curve crossing the whole disc, not
an isolated point. The signed root is therefore evaluated as
rate / sqrt(|rate| + epsilon), which is smooth, odd, and indistinguishable
from the exact form once |rate| exceeds epsilon.

gamma diverges logarithmically as M goes to zero, near the root and along the
reverse flow boundary. A floor on M keeps it finite; at the default 0.02 it
caps gamma at 6.0, and no station above r/R = 0.1 is affected at mu = 0.3.

    alpha, M, UB_bar, psi, c_R --> StallDelayComp --> d_alpha_stall, alpha_rate
"""

import numpy as np
import openmdao.api as om

GAMMA_COEF = 1.76           # p. 220, from reference 3.44
GAMMA_MACH = 0.6            # Mach at which the delay changes sign


class StallDelayComp(om.ExplicitComponent):
    """Dynamic stall delay, p. 220-221."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('epsilon', types=float, default=1.0e-6,
                             desc='softens the signed square root of the rate')
        self.options.declare('mach_floor', types=float, default=0.02,
                             desc='floor under M in the logarithm')
        self.options.declare('max_delay', types=float, default=10.0,
                             desc='saturation of the delay in degrees; 0 '
                                  'leaves it unbounded, as printed')

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)
        size = nn * n_psi * n_r

        self.add_input('alpha', shape=field, units='rad')
        self.add_input('M', shape=field)
        self.add_input('UB_bar', shape=field)
        self.add_input('psi', shape=(n_psi,), units='rad')
        self.add_input('c_R', val=2.0 / 30.0, desc='chord over radius')

        self.add_output('d_alpha_stall', shape=field, units='rad',
                        desc='increment to the stall angle, p. 221')
        self.add_output('alpha_rate', shape=field,
                        desc='d(alpha)/d(psi), wrapped, dimensionless')

        rows = np.arange(size)
        zeros = np.zeros(size, dtype=int)
        index = np.arange(size).reshape(field)
        self._next = np.roll(index, -1, axis=1).ravel()
        self._prev = np.roll(index, 1, axis=1).ravel()

        for out in ('d_alpha_stall', 'alpha_rate'):
            self.declare_partials(out, 'alpha', rows=np.tile(rows, 2),
                                  cols=np.concatenate([self._next, self._prev]))
            # the rate divides by the azimuth step, so it depends on the first
            # two stations; easy to forget because psi is a fixed grid, and
            # check_partials on the assembled chain is what catches it
            self.declare_partials(out, 'psi', rows=np.tile(rows, 2),
                                  cols=np.concatenate([np.zeros(size, int),
                                                       np.ones(size, int)]))
        for name in ('M', 'UB_bar'):
            self.declare_partials('d_alpha_stall', name, rows=rows, cols=rows)
        self.declare_partials('d_alpha_stall', 'c_R', rows=rows, cols=zeros)

    def _rate(self, inputs):
        """Centred d(alpha)/d(psi), differenced modulo one turn."""
        alpha = inputs['alpha']
        d_psi = inputs['psi'][1] - inputs['psi'][0]
        raw = np.roll(alpha, -1, axis=1) - np.roll(alpha, 1, axis=1)

        # unwrap on the real part so a complex step cannot change the branch
        turns = np.round(np.real(raw) / (2.0 * np.pi))
        return (raw - 2.0 * np.pi * turns) / (2.0 * d_psi)

    def _pieces(self, inputs):
        rate = self._rate(inputs)
        eps = self.options['epsilon']
        M = np.where(np.real(inputs['M']) > self.options['mach_floor'],
                     inputs['M'], self.options['mach_floor'])

        gamma = GAMMA_COEF * np.log(GAMMA_MACH / M)
        # U_B is positive by construction, but a solver exploring outside the
        # physical set can hand this a negative value; abs keeps the square
        # root real without changing anything at the solution
        scale = np.sqrt(90.0 / np.pi * inputs['c_R'][0]
                        / np.where(np.real(inputs['UB_bar']) < 0.0,
                                   -inputs['UB_bar'], inputs['UB_bar']))
        signed = rate / np.sqrt(np.abs(np.real(rate)) + eps)
        return rate, gamma, scale, signed, M

    def _saturate(self, raw):
        """Smooth limit on the delay, in degrees.

        The printed formula is unbounded and reaches -200 deg on the reverse
        flow boundary. On the lift side that is harmless, because the p. 221
        bounds clip c_l afterwards. On the DRAG side it is not: the delay
        shifts the drag divergence angle alpha_D, and an alpha_D of -116 deg
        turns the (alpha - alpha_D)^2.54 rise into a drag coefficient of 110.
        Prouty bounds c_l and says nothing about c_d, which works with
        tabulated data because the table saturates, and does not with the
        Chapter 6 equations.

        L tanh(x/L) is smooth, odd, and indistinguishable from x while the
        delay stays small -- which it does over 89 % of the disc.
        """
        limit = self.options['max_delay']
        if limit <= 0.0:
            return raw, np.ones_like(raw)
        ratio = np.tanh(raw / limit)
        return limit * ratio, 1.0 - ratio ** 2

    def compute(self, inputs, outputs):
        rate, gamma, scale, signed, _ = self._pieces(inputs)

        outputs['alpha_rate'] = rate
        # written out rather than np.deg2rad, which rejects complex input
        limited, _ = self._saturate(gamma * scale * signed)
        outputs['d_alpha_stall'] = (np.pi / 180.0) * limited

    def compute_partials(self, inputs, partials):
        rate, gamma, scale, signed, M = self._pieces(inputs)
        eps = self.options['epsilon']
        d_psi = inputs['psi'][1] - inputs['psi'][0]
        to_rad = np.pi / 180.0

        root = np.sqrt(np.abs(np.real(rate)) + eps)
        d_signed = 1.0 / root                       # d(signed)/d(rate)
        d_rate = 1.0 / (2.0 * d_psi)
        _, d_sat = self._saturate(gamma * scale * signed)

        chain = (to_rad * d_sat * gamma * scale * d_signed * d_rate).ravel()
        partials['d_alpha_stall', 'alpha'] = np.concatenate([chain, -chain])
        partials['alpha_rate', 'alpha'] = np.concatenate(
            [np.full(rate.size, d_rate), np.full(rate.size, -d_rate)])

        # d(rate)/d(psi_0) = +rate/d_psi, d(rate)/d(psi_1) = -rate/d_psi
        by_step = (rate / d_psi).ravel()
        d_delay = chain / d_rate                    # d(delay)/d(rate)
        partials['d_alpha_stall', 'psi'] = np.concatenate(
            [d_delay * by_step, -d_delay * by_step])
        partials['alpha_rate', 'psi'] = np.concatenate([by_step, -by_step])

        value = to_rad * d_sat * gamma * scale * signed
        active = np.real(inputs['M']) > self.options['mach_floor']
        partials['d_alpha_stall', 'M'] = np.where(
            active,
            -to_rad * d_sat * GAMMA_COEF * scale * signed / M, 0.0).ravel()
        UB = inputs['UB_bar']
        UB = np.where(np.real(UB) < 0.0, -UB, UB)
        partials['d_alpha_stall', 'UB_bar'] = (
            -0.5 * value / UB * np.sign(np.real(inputs['UB_bar']))).ravel()
        partials['d_alpha_stall', 'c_R'] = (
            0.5 * value / inputs['c_R'][0]).ravel()
