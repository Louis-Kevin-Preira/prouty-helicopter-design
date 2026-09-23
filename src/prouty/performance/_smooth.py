"""
Smooth minimum shared by the Chapter 4 components.

min(a, b) is rounded by a quadratic fillet over |a - b| < w:

    d = a - b
    smoothmin = b + d                  d <= -w
              = b - (d - w)^2 / (4 w)  |d| < w
              = b                      d >= w

It is C1, monotone in a and b, never above min(a, b), and w/4 below it at
a = b. A smoothstep blend of the two branches overshoots min(a, b) and was
rejected (C4-5): a limited power would rise with altitude near the corner.
"""

import numpy as np


def smoothmin(a, b, w):
    """Return smoothmin(a, b), d/da and d/db. Complex-step safe."""
    d = a - b
    dr = np.real(d)
    x = np.where(dr <= -w, -w, np.where(dr >= w, w, d))
    lower = dr <= -w
    value = np.where(lower, a, b - (x - w) ** 2 / (4.0 * w))
    da = np.where(lower, 1.0, (w - x) / (2.0 * w))
    return value, da, 1.0 - da
