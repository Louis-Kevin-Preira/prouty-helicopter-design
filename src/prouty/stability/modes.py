"""Off-model post-processing of a characteristic equation (Prouty Ch. 9).

Root extraction is kept out of the OpenMDAO graph on purpose: root ordering
changes discontinuously as coefficients vary, which would poison the analytic
derivatives.  The model exposes ``char_coeffs`` and ``routh_discriminant``;
everything below is for reading and validating results.

Modal metrics follow pp. 546-549 (period, damping, time to half/double
amplitude) and p. 554 (time constant).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Mode:
    """One root of the characteristic equation and its physical reading."""

    root: complex
    period: float = np.nan            # seconds, oscillatory roots only
    frequency: float = np.nan         # rad/sec, imaginary part
    damping_ratio: float = np.nan     # zeta
    time_to_half: float = np.nan      # seconds, stable roots only
    time_to_double: float = np.nan    # seconds, unstable roots only
    time_constant: float = np.nan     # seconds, non-oscillatory roots

    @property
    def stable(self):
        return self.root.real < 0.0

    @property
    def oscillatory(self):
        return abs(self.root.imag) > 1e-12


def polynomial_roots(char_coeffs):
    """Roots of an ascending-power polynomial, sorted by decreasing damping."""
    roots = np.roots(np.asarray(char_coeffs)[::-1])
    return roots[np.lexsort((-np.abs(roots.imag), roots.real))]


def describe_mode(root):
    """Period, damping and doubling/halving times of a single root (pp. 546-549)."""
    mode = Mode(root=complex(root))
    sigma, omega = root.real, abs(root.imag)

    if omega > 1e-12:
        mode.frequency = omega
        mode.period = 2.0 * np.pi / omega
        mode.damping_ratio = -sigma / np.hypot(sigma, omega)
    elif sigma != 0.0:
        mode.time_constant = -1.0 / sigma

    if sigma < 0.0:
        mode.time_to_half = np.log(2.0) / -sigma
    elif sigma > 0.0:
        mode.time_to_double = np.log(2.0) / sigma
    return mode


def describe_modes(char_coeffs):
    """All modes of a characteristic equation, most damped first."""
    return [describe_mode(r) for r in polynomial_roots(char_coeffs)]


def routh_tests(char_coeffs, routh_discriminant, tol=1e-12):
    """The six tests of p. 557 / Table 9.17 (p. 602), as a dict of booleans."""
    coeffs = np.asarray(char_coeffs, dtype=float)
    rd = float(routh_discriminant)
    return {
        'all_coefficients_positive': bool(np.all(coeffs > 0.0)),
        'no_pure_divergence': bool(np.all(coeffs > 0.0)),
        'no_unstable_oscillation': rd > tol,
        'neutrally_stable': abs(rd) <= tol,
        'unstable': rd < -tol,
        'non_oscillatory_neutral': abs(coeffs[0]) <= tol,
    }


def format_modes(char_coeffs):
    """Readable one-line-per-mode summary, for validation scripts."""
    lines = []
    for mode in describe_modes(char_coeffs):
        text = f"s = {mode.root.real:+.4f}"
        if mode.oscillatory:
            text += f" {mode.root.imag:+.4f}i   P = {mode.period:6.2f} s"
        if not np.isnan(mode.time_to_double):
            text += f"   t_double = {mode.time_to_double:6.2f} s"
        elif not np.isnan(mode.time_to_half):
            text += f"   t_half   = {mode.time_to_half:6.2f} s"
        lines.append(text)
    return "\n".join(lines)
