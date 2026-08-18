"""
LiftCoefComp -- NACA 0012 lift coefficient assembled from the Mach-dependent
coefficients produced by LiftModelCoefsComp.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations":
  - p. 428 : linear region and post-stall form below M = 0.725
  - p. 430 : same functional form above M = 0.725
  - p. 435 : sign convention for angles below zero lift

    cl = a * alpha                                           |alpha| <= alpha_L
    cl = a * alpha - K1 * (|alpha| - alpha_L)**K2 * sgn(a)   |alpha| >  alpha_L

alpha in degrees, a in 1/deg. The NACA 0012 is symmetric, so alpha_L0 = 0 and
the post-stall decrement is applied oddly about alpha = 0.
"""

import numpy as np
import openmdao.api as om

X_TINY = 1.0e-12     # guard for x**(K2-1) and log(x) at x = 0


class LiftCoefComp(om.ExplicitComponent):
    """Assemble cl from alpha and the Mach-dependent model coefficients."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare(
            'clip_K1', types=bool, default=True,
            desc='Clip K1 to non-negative values. Above M ~ 0.83 the straight-line '
                 'fit of p. 430 extrapolates to K1 < 0, which would turn the '
                 'post-stall decrement into a lift increase.')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha', shape=(nn,), units='deg', desc='angle of attack')
        self.add_input('a', shape=(nn,), units='1/deg', desc='lift-curve slope')
        self.add_input('alpha_L', shape=(nn,), units='deg', desc='stall onset angle')
        self.add_input('K1', shape=(nn,), desc='post-stall lift decrement factor')
        self.add_input('K2', shape=(nn,), desc='post-stall lift decrement exponent')

        self.add_output('cl', shape=(nn,), desc='section lift coefficient')

        ar = np.arange(nn)
        for var in ('alpha', 'a', 'alpha_L', 'K1', 'K2'):
            self.declare_partials('cl', var, rows=ar, cols=ar)

    def _terms(self, inputs):
        """Shared intermediates: excess angle, sign, effective K1."""
        alpha = inputs['alpha']
        x = np.maximum(np.abs(alpha) - inputs['alpha_L'], 0.0)   # stall excess
        sgn = np.sign(alpha)
        K1 = inputs['K1']
        if self.options['clip_K1']:
            K1 = np.maximum(K1, 0.0)
        return x, sgn, K1

    def compute(self, inputs, outputs):
        x, sgn, K1 = self._terms(inputs)
        K2 = inputs['K2']

        decrement = np.where(x > 0.0, K1 * np.maximum(x, X_TINY) ** K2, 0.0)
        outputs['cl'] = inputs['a'] * inputs['alpha'] - sgn * decrement

    def compute_partials(self, inputs, partials):
        x, sgn, K1 = self._terms(inputs)
        K2 = inputs['K2']
        xs = np.maximum(x, X_TINY)
        active = x > 0.0

        xK2 = np.where(active, xs ** K2, 0.0)              # (|a| - aL)**K2
        dxK2 = np.where(active, K2 * xs ** (K2 - 1.0), 0.0)  # d/dx of the above

        partials['cl', 'alpha'] = inputs['a'] - K1 * dxK2
        partials['cl', 'a'] = inputs['alpha']
        partials['cl', 'alpha_L'] = sgn * K1 * dxK2
        partials['cl', 'K2'] = -sgn * K1 * xK2 * np.where(active, np.log(xs), 0.0)

        dK1 = -sgn * xK2
        if self.options['clip_K1']:
            dK1 = np.where(inputs['K1'] > 0.0, dK1, 0.0)
        partials['cl', 'K1'] = dK1


if __name__ == '__main__':
    from prouty.airfoil.lift_model_coefs_comp import LiftModelCoefsComp

    M = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 0.7, 0.8, 0.5])
    alpha = np.array([4.0, 14.0, 4.0, 8.0, 12.0, 12.0, 8.0, -12.0])

    p = om.Problem()
    p.model.add_subsystem('coefs', LiftModelCoefsComp(num_nodes=M.size),
                          promotes=['*'])
    p.model.add_subsystem('lift', LiftCoefComp(num_nodes=M.size), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('M', M)
    p.set_val('alpha', alpha)
    p.run_model()

    print(f"{'M':>6}{'alpha':>8}{'alpha_L':>9}{'cl':>9}")
    for i in range(M.size):
        print(f"{M[i]:6.2f}{alpha[i]:8.1f}{p.get_val('alpha_L')[i]:9.2f}"
              f"{p.get_val('cl')[i]:9.4f}")

    # FD, not CS: np.abs / np.sign are not complex-step safe.
    p.check_partials(method='fd', compact_print=True, includes=['lift'])
