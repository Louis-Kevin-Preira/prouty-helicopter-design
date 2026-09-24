"""
Cubic smoothstep shared by the Chapter 2 components.

    s(t) = 3 t^2 - 2 t^3,   t = (x - x0) / (x1 - x0) clipped to [0, 1]

C1 at both ends, used to blend two branches of a model without a kink.
"""

import numpy as np


def smoothstep(x, x0, x1):
    """Return s and ds/dx. Complex-step safe."""
    t = (x - x0) / (x1 - x0)
    tr = np.real(t)
    inside = (tr > 0.0) & (tr < 1.0)
    s = np.where(inside, t * t * (3.0 - 2.0 * t), np.where(tr >= 1.0, 1.0, 0.0))
    ds = np.where(inside, 6.0 * t * (1.0 - t) / (x1 - x0), 0.0)
    return s, ds
