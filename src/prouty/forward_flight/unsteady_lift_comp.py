"""
UnsteadyLiftComp -- unsteady potential flow correction to the lift.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 222-225.

A blade element does not respond instantly to a change of angle of attack.
The shed vorticity trailing behind it induces velocities at the airfoil, so
the lift lags and is reduced. Prouty takes the classical oscillating-airfoil
result of reference 3.45, written for a section pitching about its quarter
chord (p. 222), and specialises it to a blade element oscillating about a mean
angle of attack (p. 223):

    c_l = a { alpha_mean + C [ delta_alpha + k theta_dot/Omega ]
              + (k/2)(alpha_dot/Omega) + (k/2)^2 (theta_ddot/Omega^2) }

with C = F + iG, F a lift deficiency and G a lag. Assuming both delta_alpha
and theta_dot are pure oscillations at rotor frequency turns i delta_alpha
into alpha_dot/Omega and i theta_dot into theta_ddot/Omega (p. 224), which
removes the imaginary parts and leaves the increment this component returns:

    delta_c_l = a { (F - 1) delta_alpha + F k theta_dot/Omega
                    + G [ alpha_dot/Omega + k theta_ddot/Omega^2 ]
                    + (k/2)(alpha_dot/Omega) + (k/2)^2 (theta_ddot/Omega^2) }

added to the steady coefficient, which is why it is an increment rather than
a replacement.

Reduced frequency and the functions, p. 225:

    k = (c/2R) / sqrt(U_T^2 + U_R^2)          note U_TR, not U_T
    1/k < 15 :  F = 0.9 - 0.00178 (15 - 1/k)^2
    1/k > 15 :  F = 0.864 + 0.0024 (1/k)
                G = -0.2 + 0.0025 (1/k)

F is continuous at 1/k = 15 -- both branches give exactly 0.9 -- but its slope
is not, jumping from 0 to 0.0024. The kink is small and sits at a reduced
frequency of 0.067, well outboard on the disc.

The mean angle of attack, p. 225:

    alpha_mean = theta_0 + theta_1 (r/R)
                 + (1/2) atan[lambda'/(r/R)]
                 + (1/4) atan[lambda'/(r/R + mu)]
                 + (1/4) atan[lambda'/(r/R - mu)]

Three effective radii, weighted a half and two quarters, standing in for the
azimuthal mean of the induced angle.

That formula is available under mean_form='book' but it is NOT the default,
and the measurement is why. Its last denominator vanishes exactly on the
reverse flow radius, so alpha_mean swings 35 deg between two neighbouring
stations at r/R = mu -- +21.9 deg at 0.275, -13.5 deg at 0.300 -- and its
first term diverges at the root. The result is |delta_c_l| above 1 over 12.5 %
of the disc, reaching -9.5 inboard of r/R = 0.125, and a NET LOSS of 1.5 % in
thrust where p. 225 says the effect should "increase the rotor thrust
slightly". The correction ends up dominated by the singularities of its own
mean, not by the shed vorticity it is meant to represent.

mean_form='azimuth_average', the default, takes alpha_mean to be what the
p. 225 formula is approximating in the first place: the actual mean of alpha
around the azimuth at each radial station. The model already has alpha
everywhere, so the approximation buys nothing, and the average is smooth and
free of singularities. The one subtlety is that alpha jumps 360 deg across the
reverse flow boundary by construction of the quadrant rule, so the azimuthal
sequence is unwrapped before averaging -- the same modulo-360 treatment
StallDelayComp needs for the same reason. Unwrapping shifts each station by a
locally constant multiple of 2 pi, so the derivative of the mean with respect
to every station is simply 1/N.

theta_dot/Omega and theta_ddot/Omega^2 come from PitchDistComp and
alpha_dot/Omega is the same azimuthal rate the dynamic stall delay uses, so
nothing is recomputed here.

The azimuthal rate is saturated, and that is the third guard this correction
needs. p. 224 derives the whole thing by assuming delta_alpha and theta_dot
are PURE OSCILLATIONS AT ROTOR FREQUENCY -- that is what turns i delta_alpha
into alpha_dot/Omega and removes the imaginary parts. Near the reverse flow
boundary alpha does not oscillate once per revolution, it swings between
neighbouring stations: the median |d(alpha)/d(psi)| there is 0.34 against
0.046 over the rest of the disc, and the term a (G + k/2) d(alpha)/d(psi)
alone reaches +/- 3 in lift coefficient, dominating the increment. The
founding assumption simply does not hold in that region.

max_rate limits the rate with the same smooth L tanh(x/L) used for the
dynamic stall delay, for the same reason and at the same place on the disc.
Outboard, where the correction is meant to act, the rate is far below the
limit and nothing changes: at r/R = 0.75 delta_c_l stays between -0.053 and
+0.039, which is the few per cent Figure 3.61 shows.

The correction is faded out where its own founding assumption fails, and that
is the only guard here that is not a patch. p. 224 assumes delta_alpha and
theta_dot are PURE OSCILLATIONS AT ROTOR FREQUENCY -- that is what turns
i delta_alpha into alpha_dot/Omega and removes the imaginary parts. A pure
1/rev oscillation satisfies |alpha_dot/Omega| = |delta_alpha| exactly, so the
ratio

    coherence = |alpha_dot/Omega| / (|delta_alpha| + eps)

is a free pointwise test of the assumption: both quantities are already
computed, and nothing has to be assumed about where the assumption holds. Near
the reverse flow boundary alpha does not oscillate, it swings between
neighbouring stations, and the ratio runs far from 1.

The window is exp(-((c - 1)/w)^2) with w the coherence option, smooth and equal
to 1 where the element behaves as the derivation requires. Set coherence = 0,
the default, to disable it and keep the printed behaviour.

Prouty's own assessment (p. 225): the measured effect is somewhat larger than
predicted, and "the overall result is to increase the rotor thrust slightly
for a given set of rotor conditions". Figure 3.61 shows it moving the lift
coefficient at the 75 % station by a few per cent around the azimuth.

    alpha, alpha_rate, theta_dot_Om, theta_ddot_Om2, UTR_bar, r_R,
    lambda_p, mu, theta_0, theta_1, c_R, a  -->  d_cl_unsteady
"""

import numpy as np
import openmdao.api as om

F_BREAK = 15.0          # value of 1/k where the two branches of F meet


class UnsteadyLiftComp(om.ExplicitComponent):
    """Unsteady lift increment, p. 224-225."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('mean_form',
                             values=('azimuth_average', 'book'),
                             default='azimuth_average',
                             desc="'book' uses the three-arctangent estimate "
                                  'of p. 225, which is singular at r/R = mu '
                                  "and at the root; 'azimuth_average' takes "
                                  'the true mean of alpha around the azimuth')
        self.options.declare('epsilon', types=float, default=1.0e-6,
                             desc='guard on the three effective radii')
        self.options.declare('coherence', types=float, default=0.0,
                             desc='width of the window that fades the '
                                  'correction out where the element does not '
                                  'oscillate once per revolution; 0 disables '
                                  'it. See the module header -- this is the '
                                  'p. 224 assumption tested pointwise, not a '
                                  'mask.')
        self.options.declare('max_rate', types=float, default=0.15,
                             desc='saturation of d(alpha)/d(psi); 0 leaves it '
                                  'unbounded, as printed')
        self.options.declare('max_k', types=float, default=0.5,
                             desc='saturation of the reduced frequency; 0 '
                                  'leaves it unbounded, as printed')

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)
        size = nn * n_psi * n_r

        for name in ('alpha', 'alpha_rate', 'UTR_bar'):
            units = 'rad' if name == 'alpha' else None
            self.add_input(name, shape=field, units=units)
        self.add_input('theta_dot_Om', shape=(nn, n_psi), units='rad')
        self.add_input('theta_ddot_Om2', shape=(nn, n_psi), units='rad')
        self.add_input('r_R', shape=(n_r,))
        self.add_input('lambda_p', shape=(nn,))
        self.add_input('mu', shape=(nn,))
        self.add_input('theta_0', shape=(nn,), units='rad')
        self.add_input('theta_1', val=-0.17453, units='rad')
        self.add_input('c_R', val=2.0 / 30.0)
        self.add_input('a', val=6.0, units='1/rad')

        self.add_output('d_cl_unsteady', shape=field)
        self.add_output('alpha_mean', shape=field, units='rad')
        self.add_output('k_red', shape=field, desc='reduced frequency')
        self.add_output('coherence', shape=field,
                        desc='how well the element satisfies the 1/rev '
                             'assumption of p. 224; 1 is perfect')

        rows = np.arange(size)
        node = np.repeat(np.arange(nn), n_psi * n_r)
        sweep = np.repeat(np.arange(nn * n_psi), n_r)
        radial = np.tile(np.arange(n_r), nn * n_psi)
        zeros = np.zeros(size, dtype=int)

        for name in ('alpha_rate', 'UTR_bar'):
            self.declare_partials('d_cl_unsteady', name, rows=rows, cols=rows)
        if self.options['mean_form'] == 'book':
            self.declare_partials('d_cl_unsteady', 'alpha', rows=rows,
                                  cols=rows)
        for name in ('theta_dot_Om', 'theta_ddot_Om2'):
            self.declare_partials('d_cl_unsteady', name, rows=rows, cols=sweep)
        if self.options['mean_form'] == 'book':
            for name in ('lambda_p', 'mu', 'theta_0'):
                self.declare_partials(['d_cl_unsteady', 'alpha_mean'], name,
                                      rows=rows, cols=node)
            self.declare_partials(['d_cl_unsteady', 'alpha_mean'], 'r_R',
                                  rows=rows, cols=radial)
            self.declare_partials(['d_cl_unsteady', 'alpha_mean'], 'theta_1',
                                  rows=rows, cols=zeros)
        else:
            # the mean at (n, psi, r) draws on every azimuth station of that
            # radius, so alpha enters through a dense block along axis 1
            index = np.arange(size).reshape(field)
            block = (nn, n_psi, n_psi, n_r)
            self._mean_rows = np.repeat(index.ravel(), n_psi)
            self._mean_cols = np.transpose(
                np.broadcast_to(index[:, np.newaxis, :, :], block),
                (0, 1, 3, 2)).ravel()
            self._identity = np.broadcast_to(
                np.eye(n_psi)[np.newaxis, :, np.newaxis, :],
                (nn, n_psi, n_r, n_psi))
            for out in ('d_cl_unsteady', 'alpha_mean'):
                self.declare_partials(out, 'alpha', rows=self._mean_rows,
                                      cols=self._mean_cols)
        self.declare_partials('d_cl_unsteady', ['c_R', 'a'], rows=rows,
                              cols=zeros)
        self.declare_partials('k_red', 'UTR_bar', rows=rows, cols=rows)
        self.declare_partials('k_red', 'c_R', rows=rows, cols=zeros)

    def _radii(self, inputs):
        """The three effective radii of p. 225, kept away from zero."""
        eps = self.options['epsilon']
        r = inputs['r_R'][np.newaxis, np.newaxis, :]
        mu = inputs['mu'][:, np.newaxis, np.newaxis]

        raw = (r, r + mu, r - mu)
        return tuple(np.where(np.real(d) >= 0.0, d + eps, d - eps)
                     for d in raw)

    def _oscillation(self, inputs):
        """delta_alpha and the mean it is measured from.

        In azimuth_average mode delta_alpha is built from the UNWRAPPED alpha,
        not the raw one. Mixing them was the first attempt and it is wrong by
        a whole turn: inside the reverse flow circle the quadrant rule puts
        alpha near 250 deg while the mean sits near zero, so delta_alpha came
        out at 4 rad and (F - 1) delta_alpha reached -10 in lift coefficient.
        The oscillation the theory wants is the departure from the mean of the
        same continuous signal.
        """
        mean, weights = self._mean_angle(inputs)
        if weights is None:
            unwrapped, _ = self._azimuth_mean(inputs)
            return unwrapped - mean, mean, weights
        return inputs['alpha'] - mean, mean, weights

    def _mean_angle(self, inputs):
        if self.options['mean_form'] == 'azimuth_average':
            return self._azimuth_mean(inputs)[1], None

        r = inputs['r_R'][np.newaxis, np.newaxis, :]
        lam = inputs['lambda_p'][:, np.newaxis, np.newaxis]
        weights = (0.5, 0.25, 0.25)

        angle = (inputs['theta_0'][:, np.newaxis, np.newaxis]
                 + inputs['theta_1'][0] * r)
        for weight, d in zip(weights, self._radii(inputs)):
            angle = angle + weight * np.arctan(lam / d)
        return angle, weights

    def _azimuth_mean(self, inputs):
        """Mean of alpha around the azimuth, unwrapped first.

        alpha jumps 360 deg across the reverse flow boundary, so the raw mean
        would be shifted by 2 pi times whatever fraction of the revolution the
        element spends reversed. Differencing, wrapping each step onto
        (-pi, pi] and re-accumulating removes the jump; the branch decision
        uses the real part so a complex step cannot move it.
        """
        alpha = inputs['alpha']
        steps = alpha[:, 1:, :] - alpha[:, :-1, :]
        turns = np.round(np.real(steps) / (2.0 * np.pi))
        unwrapped = alpha[:, :1, :] + np.concatenate(
            [np.zeros_like(alpha[:, :1, :]),
             np.cumsum(steps - 2.0 * np.pi * turns, axis=1)], axis=1)
        return unwrapped, np.broadcast_to(
            unwrapped.mean(axis=1, keepdims=True), alpha.shape)

    def _coherence(self, delta, rate):
        """Window on how well the element satisfies the 1/rev assumption."""
        width = self.options['coherence']
        if width <= 0.0:
            return np.ones_like(delta), np.zeros_like(delta), \
                np.zeros_like(delta)

        eps = 1.0e-6
        magnitude = np.sqrt(delta ** 2 + eps ** 2)
        ratio = np.sqrt(rate ** 2 + eps ** 2) / magnitude
        offset = (ratio - 1.0) / width
        window = np.exp(-offset ** 2)

        # d(window)/d(ratio), then the chain onto rate and delta
        d_window = -2.0 * offset * window / width
        d_rate = d_window * rate / (np.sqrt(rate ** 2 + eps ** 2) * magnitude)
        d_delta = -d_window * ratio * delta / magnitude ** 2
        return window, d_rate, d_delta

    def _rate(self, inputs):
        """Azimuthal rate, saturated. Returns the value and d(value)/d(raw)."""
        raw = inputs['alpha_rate']
        limit = self.options['max_rate']
        if limit <= 0.0:
            return raw, np.ones_like(raw)
        ratio = np.tanh(raw / limit)
        return limit * ratio, 1.0 - ratio ** 2

    def _deficiency(self, inputs):
        """k, 1/k, F, G and the slopes dF/du, dG/du.

        The reduced frequency is capped. k = (c/2R)/U_TR diverges wherever the
        local velocity vanishes, and it does: at psi = 270 deg and r/R = mu,
        U_T and U_R vanish together, so U_TR falls to the regularisation
        epsilon and k reaches 3e6. The k^2 term then returns a lift increment
        of 1.6e12. That is not Prouty's problem to have foreseen -- the
        oscillating airfoil theory of reference 3.45 is an expansion in k and
        stops meaning anything much above 0.5, which is the default here.
        """
        raw = 0.5 * inputs['c_R'][0] / inputs['UTR_bar']
        limit = self.options['max_k']
        k = limit * np.tanh(raw / limit) if limit > 0.0 else raw
        u = 1.0 / k
        low = np.real(u) < F_BREAK

        F = np.where(low, 0.9 - 0.00178 * (F_BREAK - u) ** 2,
                     0.864 + 0.0024 * u)
        dF = np.where(low, 0.00356 * (F_BREAK - u), 0.0024)
        G = -0.2 + 0.0025 * u
        d_raw = (1.0 - (k / limit) ** 2) if limit > 0.0 else np.ones_like(k)
        return k, u, F, G, dF, np.full_like(np.real(u), 0.0025), d_raw

    def compute(self, inputs, outputs):
        delta, mean, _ = self._oscillation(inputs)
        k, _, F, G, _, _, _ = self._deficiency(inputs)
        rate, _ = self._rate(inputs)
        theta_d = inputs['theta_dot_Om'][:, :, np.newaxis]
        theta_dd = inputs['theta_ddot_Om2'][:, :, np.newaxis]

        window, _, _ = self._coherence(delta, rate)

        outputs['alpha_mean'] = mean
        outputs['k_red'] = k
        outputs['coherence'] = window
        outputs['d_cl_unsteady'] = inputs['a'][0] * window * (
            (F - 1.0) * delta
            + F * k * theta_d
            + G * (rate + k * theta_dd)
            + 0.5 * k * rate
            + 0.25 * k ** 2 * theta_dd)

    def compute_partials(self, inputs, partials):
        shape = inputs['alpha'].shape
        a = inputs['a'][0]
        delta, mean, weights = self._oscillation(inputs)
        k, u, F, G, dF, dG, d_raw = self._deficiency(inputs)
        rate, d_rate_raw = self._rate(inputs)
        UTR, c_R = inputs['UTR_bar'], inputs['c_R'][0]
        theta_d = inputs['theta_dot_Om'][:, :, np.newaxis]
        theta_dd = inputs['theta_ddot_Om2'][:, :, np.newaxis]
        lam = inputs['lambda_p'][:, np.newaxis, np.newaxis]
        radii = self._radii(inputs)
        full = lambda x: np.broadcast_to(x, shape).ravel()

        value = a * ((F - 1.0) * delta + F * k * theta_d
                     + G * (rate + k * theta_dd) + 0.5 * k * rate
                     + 0.25 * k ** 2 * theta_dd)

        d_alpha = a * (F - 1.0)
        d_rate = a * (G + 0.5 * k) * d_rate_raw
        d_k = a * (F * theta_d + G * theta_dd + 0.5 * rate
                   + 0.5 * k * theta_dd)
        d_F = a * (delta + k * theta_d)
        d_G = a * (rate + k * theta_dd)

        if weights is None:
            # delta_alpha = alpha - mean(alpha), so each station sees itself
            # with weight 1 and every station of its radius with -1/N
            n_psi = self.options['num_azimuth']
            spread = self._identity - 1.0 / n_psi
            partials['alpha_mean', 'alpha'] = np.full(
                self._mean_rows.size, 1.0 / n_psi)
            partials['d_cl_unsteady', 'alpha'] = (
                d_alpha[:, :, :, np.newaxis] * spread).ravel()
        else:
            partials['d_cl_unsteady', 'alpha'] = full(d_alpha)
        partials['d_cl_unsteady', 'alpha_rate'] = full(d_rate)
        partials['d_cl_unsteady', 'theta_dot_Om'] = full(a * F * k)
        partials['d_cl_unsteady', 'theta_ddot_Om2'] = full(
            a * (G * k + 0.25 * k ** 2))
        partials['d_cl_unsteady', 'a'] = (value / a).ravel()

        # k = (c_R/2)/U_TR and u = 1/k, so both move with U_TR and c_R
        # d(k)/d(raw) = d_raw from the saturation, and raw = (c_R/2)/U_TR
        raw = 0.5 * c_R / UTR
        dk_UTR = -d_raw * raw / UTR
        dk_cR = d_raw * raw / c_R
        du = -1.0 / k ** 2
        chain = (d_F * dF + d_G * dG) * du

        partials['d_cl_unsteady', 'UTR_bar'] = full((d_k + chain) * dk_UTR)
        partials['d_cl_unsteady', 'c_R'] = full((d_k + chain) * dk_cR)
        partials['k_red', 'UTR_bar'] = full(dk_UTR)
        partials['k_red', 'c_R'] = full(dk_cR)

        if weights is None:
            return

        # the mean angle enters only through -(F - 1) alpha_mean
        r = inputs['r_R'][np.newaxis, np.newaxis, :]
        d_lam = np.zeros_like(mean)
        d_r = np.full_like(mean, inputs['theta_1'][0])
        d_mu = np.zeros_like(mean)
        for index, (weight, d) in enumerate(zip(weights, radii)):
            denom = d ** 2 + lam ** 2
            d_lam = d_lam + weight * d / denom
            d_r = d_r - weight * lam / denom
            if index == 1:
                d_mu = d_mu - weight * lam / denom
            elif index == 2:
                d_mu = d_mu + weight * lam / denom

        for name, derivative in (('lambda_p', d_lam), ('mu', d_mu),
                                 ('r_R', d_r),
                                 ('theta_0', np.ones_like(mean)),
                                 ('theta_1', np.broadcast_to(r, shape))):
            partials['alpha_mean', name] = full(derivative)
            partials['d_cl_unsteady', name] = full(-d_alpha * derivative)
