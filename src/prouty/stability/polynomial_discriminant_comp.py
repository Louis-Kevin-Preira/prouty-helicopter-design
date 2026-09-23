"""The discriminant of a characteristic polynomial.

Not Routh's discriminant. The two answer different questions and Chapter 9
uses both, naming only one.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", p. 618. The right-hand boundary of
Figure 9.15 "was determined by finding combinations of the two derivatives
that made the roots of the characteristic equation switch from complex to
real"; that is this discriminant vanishing, and it has a closed form.
"""

import numpy as np
import openmdao.api as om


def _sylvester(descending):
    """The Sylvester matrix of ``p`` and ``p'``, batched over leading axes.

    ``descending`` is ``(..., n + 1)``. The result is
    ``(..., 2n - 1, 2n - 1)``: ``n - 1`` shifted copies of ``p`` above ``n``
    shifted copies of ``p'``.
    """
    batch = descending.shape[:-1]
    order = descending.shape[-1] - 1
    rows = order - 1
    size = order + rows

    derivative = descending[..., :-1] * np.arange(order, 0, -1)
    matrix = np.zeros(batch + (size, size), dtype=descending.dtype)
    for index in range(rows):
        matrix[..., index, index:index + order + 1] = descending
    for index in range(order):
        matrix[..., rows + index, index:index + order] = derivative
    return matrix


def _cofactors(matrix):
    """Cofactor matrix, by explicit minors.

    ``det(S) inv(S).T`` would be quicker and fails exactly where it matters:
    the discriminant vanishing makes ``S`` singular, and that locus is the
    boundary the component exists to find.
    """
    size = matrix.shape[-1]
    cofactors = np.zeros_like(matrix)
    index = np.arange(size)
    for i in range(size):
        without_row = matrix[..., index[index != i], :]
        for j in range(size):
            minor = without_row[..., :, index[index != j]]
            cofactors[..., i, j] = (-1.0) ** (i + j) * np.linalg.det(minor)
    return cofactors


def discriminant(char_coeffs):
    """Discriminant of a polynomial given in ascending-power form."""
    descending = np.asarray(char_coeffs)[..., ::-1]
    order = descending.shape[-1] - 1
    sign = (-1.0) ** (order * (order - 1) // 2)
    resultant = np.linalg.det(_sylvester(descending))
    return sign * resultant / descending[..., 0]


def quartic_root_character(char_coeffs, tol=1e-9):
    """Name the root pattern of a quartic without solving for the roots.

    The standard classification, in closed form. With ``a s^4 + b s^3 +
    c s^2 + d s + e`` and

        P = 8 a c - 3 b^2
        D = 64 a^3 e - 16 a^2 c^2 + 16 a b^2 c - 16 a^2 b d - 3 b^4

    a negative discriminant means two real roots and one complex pair; a
    positive one means four real roots when ``P`` and ``D`` are both negative
    and two complex pairs otherwise; zero means a repeated root.

    This is what fills the gap Table 9.17 leaves. Routh's discriminant says
    *whether* the aircraft is unstable and says nothing about the kind; the
    example helicopter in forward flight has ``R.D. = -.32`` and four real
    roots, a pure divergence, which no Routh test distinguishes from an
    unstable oscillation. Here it is algebra rather than a root search.
    """
    a, b, c, d, e = np.asarray(char_coeffs)[..., ::-1]
    delta = discriminant(char_coeffs)
    P = 8.0 * a * c - 3.0 * b ** 2
    D = (64.0 * a ** 3 * e - 16.0 * a ** 2 * c ** 2 + 16.0 * a * b ** 2 * c
         - 16.0 * a ** 2 * b * d - 3.0 * b ** 4)

    if abs(delta) <= tol * max(1.0, abs(a) ** 6):
        return 'repeated root'
    if delta < 0.0:
        return 'two real roots and one complex pair'
    if P < 0.0 and D < 0.0:
        return 'four real roots'
    return 'two complex pairs'


class PolynomialDiscriminantComp(om.ExplicitComponent):
    """The discriminant of the characteristic polynomial.

    Sits beside ``RouthDiscriminantComp`` and answers a different question::

        Routh's discriminant = 0   a root pair crosses the imaginary axis
                                   -> stable oscillation / unstable oscillation
        this discriminant   = 0    a root becomes double
                                   -> complex pair / two real roots

    The first is about a real part changing sign, the second about an imaginary
    part vanishing. Chapter 9's Figure 9.15 is bounded by both and labels only
    the first; p. 618 says the other "was determined by finding combinations of
    the two derivatives that made the roots of the characteristic equation
    switch from complex to real", which is a root search where an algebraic
    condition will do.

    How it is computed
    ------------------
    ``disc(p) = (-1)^(n(n-1)/2) Res(p, p') / a_n``, with the resultant as a
    Sylvester determinant built from the coefficients alone. No roots, exact,
    and differentiable: the partials are cofactors of that determinant, and
    since each coefficient appears in known positions of the Sylvester matrix
    the chain rule is a sum over those positions.

    The cofactors are taken by explicit minors rather than through
    ``det(S) inv(S).T``, which is quicker and singular precisely on the locus
    the component exists to find.

    What the sign means
    -------------------
    For a quartic, **negative** is unambiguous: two real roots and one complex
    pair. **Positive** is not -- four real roots, or two complex pairs, and
    those are opposite ends of the map. The example helicopter's longitudinal
    quartic has ``disc = +67.9`` with four real roots at 18 ft^2 of stabilizer
    and ``disc = +4.85`` with two complex pairs at 90 ft^2.
    :func:`quartic_root_character` resolves it with the two companion
    quantities ``P`` and ``D``, still without root-finding.

    Options
    -------
    degree : int
    num_nodes : int

    Example helicopter at 115 knots, longitudinal
    ---------------------------------------------
    ======== ========== =========== ==================
    area     R.D.(4)    disc        roots
    ======== ========== =========== ==================
    18 ft^2  -.3191     **+67.92**  4 real
    36 ft^2  -.5345     **-4.244**  2 real + pair
    54 ft^2  -.4735     -6.464      2 real + pair
    72 ft^2  -.0553     -.5302      2 real + pair
    90 ft^2  +.8025     **+4.853**  2 pairs
    ======== ========== =========== ==================

    ``disc`` changes sign between 18 and 36 square feet, which is exactly
    where p. 619 says doubling the stabilizer moves the aircraft "from a region
    of pure divergences to one of unstable oscillations". Routh's discriminant
    changes sign between 72 and 90, the other boundary. The two are
    independent and the map needs both.
    """

    def initialize(self):
        self.options.declare('degree', types=int, lower=2, upper=6)
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ncoef = self.options['degree'] + 1

        self.add_input('char_coeffs', shape=(nn, ncoef))
        self.add_output('discriminant', shape=(nn,))
        self.declare_partials('discriminant', 'char_coeffs',
                              rows=np.repeat(np.arange(nn), ncoef),
                              cols=np.arange(nn * ncoef))

    def compute(self, inputs, outputs):
        outputs['discriminant'] = discriminant(inputs['char_coeffs'])

    def compute_partials(self, inputs, J):
        order = self.options['degree']
        rows = order - 1
        sign = (-1.0) ** (order * (order - 1) // 2)

        descending = inputs['char_coeffs'][:, ::-1]
        matrix = _sylvester(descending)
        cofactors = _cofactors(matrix)
        resultant = np.linalg.det(matrix)
        leading = descending[:, 0]

        # each coefficient sits in known positions of the Sylvester matrix:
        # descending[k] in rows 0..rows-1 at column i+k, and its contribution
        # to p' , (order - k) descending[k], in rows rows..rows+order-1
        gradient = np.zeros_like(descending)
        for k in range(order + 1):
            total = sum(cofactors[:, i, i + k] for i in range(rows))
            if k < order:
                total = total + (order - k) * sum(
                    cofactors[:, rows + i, i + k] for i in range(order))
            gradient[:, k] = total

        derivative = sign * gradient / leading[:, None]
        derivative[:, 0] -= sign * resultant / leading ** 2

        # back to ascending order, matching the input layout
        J['discriminant', 'char_coeffs'] = derivative[:, ::-1].reshape(-1)
