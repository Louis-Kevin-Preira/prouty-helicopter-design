"""The 2x2 flapping system shared by pp. 468-469 and p. 473.

Prouty writes four nested-fraction expressions in this chapter: ``a_1s`` and
``b_1s`` in steady forward flight (pp. 468-469), and their increments due to
pitch and roll rates (p. 473). All four are the solution of the same system::

    [ 1 - mu^2/2      -kappa   ] [a_1s]   [N_1]
    [   kappa      1 + mu^2/2  ] [b_1s] = [N_2]

    kappa = 12 (e/R) / [gamma (1 - e/R)^3] = 4 h / G

only the right-hand side changes. This has been checked symbolically against
each printed expression.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 468-469 and 473.
"""


def kappa(gamma, e_over_R):
    """Off-diagonal coupling term, ``12 (e/R) / [gamma (1 - e/R)^3]``."""
    return 12.0 * e_over_R / (gamma * (1.0 - e_over_R) ** 3)


def d_kappa_d_e_over_R(gamma, e_over_R):
    return 12.0 * (1.0 + 2.0 * e_over_R) / (gamma * (1.0 - e_over_R) ** 4)


def determinant(mu, kap, exact=True):
    """``1 - mu^4/4 + kappa^2``, or without ``kappa^2`` for the p. 469 form."""
    det = 1.0 - mu ** 4 / 4.0
    return det + kap ** 2 if exact else det


def solve(N1, N2, mu, kap, det):
    """Return ``(a_1s, b_1s)``."""
    q = mu ** 2 / 2.0
    return ((N1 * (1.0 + q) + kap * N2) / det,
            (N2 * (1.0 - q) - kap * N1) / det)


def differentiate(N1, N2, a_1s, b_1s, mu, kap, det,
                  dN1=0.0, dN2=0.0, dkap=0.0, ddet=0.0, dq=0.0):
    """Chain one input through the solve.

    ``dq`` is the derivative of ``mu^2/2``, nonzero only for ``mu`` itself.
    """
    q = mu ** 2 / 2.0
    dP = (1.0 + q) * dN1 + N1 * dq + kap * dN2 + N2 * dkap
    dQ = (1.0 - q) * dN2 - N2 * dq - kap * dN1 - N1 * dkap
    return (dP - a_1s * ddet) / det, (dQ - b_1s * ddet) / det
