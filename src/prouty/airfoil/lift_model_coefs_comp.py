"""
LiftModelCoefsComp -- Mach-dependent coefficients of the NACA 0012 lift model.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations":
  - p. 427-428 : lift-curve slope a(M), stall onset angle alpha_L(M)
  - p. 429     : exponent K2(M)
  - p. 430     : coefficient K1(M), equations above M = 0.725

Lift model assembled downstream:
    cl = a * alpha                                      |alpha| <= alpha_L
    cl = a * alpha - K1 * (|alpha| - alpha_L)**K2       |alpha| >  alpha_L
with alpha in degrees.
"""

import numpy as np
import openmdao.api as om

M_BREAK = 0.725      # compressibility break, p. 426
M_BLEND = 0.020      # blending band width [M_BREAK, M_BREAK + M_BLEND]
X_TINY = 1.0e-12     # guard for the (M - M_BREAK)**0.44 singularity
M_FLOOR = 0.0        # see _clip
M_CEIL = 0.99        # see _clip


def _clip(M):
    """Keep the Mach number inside the domain where the model is a number.

    The Prandtl-Glauert factor 1/sqrt(1 - M^2) is undefined at and above M = 1
    and M**7.15 is undefined below zero, so a caller handing this component an
    out-of-range Mach number gets NaN back and, if it is a Newton solver, loses
    the whole residual vector to it. That is not hypothetical: the numerical
    rotor method of Chapter 3 carries M as a solver state, so an overshooting
    step passes values this model was never fitted for. Clipping returns a
    finite -- if meaningless -- coefficient instead, with zero derivative
    outside the range so the solver is not pulled further out.

    The range is never reached in a converged solution: the advancing tip of
    the example helicopter sits at M = 0.757.
    """
    # the floor is inclusive: M = 0 is a legitimate operating point at the
    # root of the disc and the model is perfectly well defined there, it is
    # only M < 0 that breaks M**7.15
    inside = (np.real(M) >= M_FLOOR) & (np.real(M) < M_CEIL)
    return np.where(inside, M, np.where(np.real(M) <= M_FLOOR,
                                        M_FLOOR, M_CEIL)), inside


class LiftModelCoefsComp(om.ExplicitComponent):
    """Lift-model coefficients a, alpha_L, K1, K2 as functions of Mach number."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('M', shape=(nn,), desc='local Mach number')

        self.add_output('a', shape=(nn,), units='1/deg', desc='lift-curve slope')
        self.add_output('alpha_L', shape=(nn,), units='deg', desc='stall onset angle')
        self.add_output('K1', shape=(nn,), desc='post-stall lift decrement factor')
        self.add_output('K2', shape=(nn,), desc='post-stall lift decrement exponent')

        ar = np.arange(nn)
        for out in ('a', 'alpha_L', 'K1', 'K2'):
            self.declare_partials(out, 'M', rows=ar, cols=ar)

    # ------------------------------------------------------------------
    # Blending weight: 0 below M_BREAK, 1 above M_BREAK + M_BLEND.
    # Cubic smoothstep -> C1 continuity, and w ~ t**2 near t = 0, which
    # cancels the infinite slope of (M - M_BREAK)**0.44 at the break.
    # ------------------------------------------------------------------
    @staticmethod
    def _weight(M):
        t = np.clip((M - M_BREAK) / M_BLEND, 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
        dw = np.where((t > 0.0) & (t < 1.0), 6.0 * t * (1.0 - t) / M_BLEND, 0.0)
        return w, dw

    def compute(self, inputs, outputs):
        M, _ = _clip(inputs['M'])
        w, _ = self._weight(M)

        # --- low-Mach branch, M <= 0.725 (p. 427-428, 430) -------------
        s = np.sqrt(1.0 - M ** 2)
        a_lo = 0.1 / s - 0.01 * M
        aL_lo = 15.0 - 16.0 * M
        K1_lo = 0.0233 + 0.342 * M ** 7.15

        # --- high-Mach branch, M > 0.725 (p. 430) ----------------------
        x = np.maximum(M - M_BREAK, X_TINY)
        a_hi = 0.677 - 0.744 * M
        aL_hi = np.full_like(M, 3.4)
        K1_hi = 0.0575 - 0.144 * x ** 0.44

        outputs['a'] = (1.0 - w) * a_lo + w * a_hi
        outputs['alpha_L'] = (1.0 - w) * aL_lo + w * aL_hi
        outputs['K1'] = (1.0 - w) * K1_lo + w * K1_hi
        outputs['K2'] = 2.05 - 0.95 * M          # both regimes, p. 429

    def compute_partials(self, inputs, partials):
        M, inside = _clip(inputs['M'])
        w, dw = self._weight(M)

        s = np.sqrt(1.0 - M ** 2)
        a_lo = 0.1 / s - 0.01 * M
        aL_lo = 15.0 - 16.0 * M
        K1_lo = 0.0233 + 0.342 * M ** 7.15

        da_lo = 0.1 * M / s ** 3 - 0.01
        daL_lo = -16.0
        dK1_lo = 0.342 * 7.15 * M ** 6.15

        x = np.maximum(M - M_BREAK, X_TINY)
        a_hi = 0.677 - 0.744 * M
        aL_hi = np.full_like(M, 3.4)
        K1_hi = 0.0575 - 0.144 * x ** 0.44

        da_hi = -0.744
        daL_hi = 0.0
        # zeroed where the branch is inactive, so w * inf never appears
        dK1_hi = np.where(M > M_BREAK, -0.144 * 0.44 * x ** (-0.56), 0.0)

        partials['a', 'M'] = inside * (
            (1 - w) * da_lo + w * da_hi + (a_hi - a_lo) * dw)
        partials['alpha_L', 'M'] = inside * (
            (1 - w) * daL_lo + w * daL_hi + (aL_hi - aL_lo) * dw)
        partials['K1', 'M'] = inside * (
            (1 - w) * dK1_lo + w * dK1_hi + (K1_hi - K1_lo) * dw)
        partials['K2', 'M'] = -0.95 * inside


if __name__ == '__main__':
    M = np.array([0.2, 0.5, 0.7, 0.725, 0.745, 0.85, 0.95])

    p = om.Problem()
    p.model.add_subsystem('comp', LiftModelCoefsComp(num_nodes=M.size),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('M', M)
    p.run_model()

    print(f"{'M':>7}{'a':>10}{'alpha_L':>10}{'K1':>10}{'K2':>8}")
    for i, m in enumerate(M):
        print(f"{m:7.3f}{p.get_val('a')[i]:10.5f}{p.get_val('alpha_L')[i]:10.3f}"
              f"{p.get_val('K1')[i]:10.5f}{p.get_val('K2')[i]:8.4f}")

    p.check_partials(method='fd', compact_print=True)
