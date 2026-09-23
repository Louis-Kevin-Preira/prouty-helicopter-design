"""Routh's discriminant of a cubic, quartic or quintic.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 556-557. Used on p. 602
(Table 9.17), p. 618 and pp. 633-634.
"""

import numpy as np
import openmdao.api as om


class RouthDiscriminantComp(om.ExplicitComponent):
    """Routh's discriminant for a cubic, quartic or quintic (pp. 556-557).

    With ``A s**deg + B s**(deg-1) + ... = 0``:

        R.D.(3) = BC - AD
        R.D.(4) = BCD - A D**2 - B**2 E
        R.D.(5) = D(BC - AD)(BE - AF) - B(BE - AF)**2 - F(BC - AD)**2

    The quintic form differs from the printed one, which reads ``-B(BF-AF)**2``
    -- a misprint; see docs/validation_notes_ch9.md, item C9-1.
    """

    def initialize(self):
        self.options.declare('degree', values=(3, 4, 5))
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ncoef = self.options['degree'] + 1
        self.add_input('char_coeffs', shape=(nn, ncoef))
        self.add_output('routh_discriminant', shape=(nn,))
        self.declare_partials('routh_discriminant', 'char_coeffs',
                              rows=np.repeat(np.arange(nn), ncoef),
                              cols=np.arange(nn * ncoef))

    def compute(self, inputs, outputs):
        A, B, C, D, E, F = self._unpack(inputs['char_coeffs'])
        deg = self.options['degree']

        if deg == 3:
            rd = B * C - A * D
        elif deg == 4:
            rd = B * C * D - A * D ** 2 - B ** 2 * E
        else:
            P, Q = B * C - A * D, B * E - A * F
            rd = D * P * Q - B * Q ** 2 - F * P ** 2
        outputs['routh_discriminant'] = rd

    def compute_partials(self, inputs, partials):
        A, B, C, D, E, F = self._unpack(inputs['char_coeffs'])
        deg = self.options['degree']

        if deg == 3:
            d = [-D, C, B, -A]
        elif deg == 4:
            d = [-D ** 2, C * D - 2 * B * E, B * D, B * C - 2 * A * D, -B ** 2]
        else:
            P, Q = B * C - A * D, B * E - A * F
            d = [-D ** 2 * Q + D * P * F + 2 * B * Q * F,
                 D * (C * Q + P * E) - Q ** 2 - 2 * B * Q * E - 2 * F * P * C,
                 D * B * Q - 2 * F * P * B,
                 P * Q - A * D * Q + 2 * A * F * P,
                 D * P * B - 2 * B ** 2 * Q,
                 -A * D * P + 2 * A * B * Q - P ** 2]

        # d[] is ordered A, B, C, ...; char_coeffs is ascending in s.
        ncoef = self.options['degree'] + 1
        partials['routh_discriminant', 'char_coeffs'] = np.stack(
            d[:ncoef][::-1], axis=-1).reshape(-1)

    def _unpack(self, coeffs):
        """Return A, B, C, D, E, F (descending), padded with zeros."""
        zero = np.zeros(coeffs.shape[0], dtype=coeffs.dtype)
        vals = [coeffs[:, -1 - k] for k in range(coeffs.shape[1])]
        return (vals + [zero] * 6)[:6]
