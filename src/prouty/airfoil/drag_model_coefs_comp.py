"""
DragModelCoefsComp -- Mach-dependent coefficients of the NACA 0012 drag model.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations":
  - p. 432 : drag divergence angle alpha_D(M), coefficients K3, K4 below M = 0.725
  - p. 433 : coefficients above M = 0.725, zero-angle compressibility term K5, K6

Drag model assembled downstream:
    cd = cd_incomp + K3 * (alpha - alpha_D)**K4 + delta_cd_M     alpha > alpha_D
with alpha in degrees and

    delta_cd_M = K5 * (M - 0.725)**K6      (zero above M = 0.725 only)

Below M = 0.725 : alpha_D = 17 - 23.4 M, K3 = 0.00066, delta_cd_M = 0
Above M = 0.725 : alpha_D = 0,           K3 = 0.00035, K5 = 21, K6 = 3.2
"""

import numpy as np
import openmdao.api as om

M_BREAK = 0.725      # drag divergence Mach number, p. 432
M_BLEND = 0.010      # blending half-width, band [M_BREAK - h, M_BREAK + h]

K3_LOW = 0.00066     # p. 432, average of the M = 0.3 / 0.5 / 0.7 fits
K3_HIGH = 0.00035    # p. 433
K4 = 2.54            # same exponent in both regimes, p. 432-433
K5 = 21.0            # p. 433
K6 = 3.2             # p. 433


class DragModelCoefsComp(om.ExplicitComponent):
    """Drag-model coefficients alpha_D, K3, K4 and the compressibility increment."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('M', shape=(nn,), desc='local Mach number')

        self.add_output('alpha_D', shape=(nn,), units='deg',
                        desc='drag divergence angle of attack')
        self.add_output('K3', shape=(nn,), desc='drag rise factor')
        self.add_output('K4', shape=(nn,), desc='drag rise exponent')
        self.add_output('delta_cd_M', shape=(nn,),
                        desc='zero-angle compressibility drag increment')

        ar = np.arange(nn)
        for out in ('alpha_D', 'K3', 'delta_cd_M'):
            self.declare_partials(out, 'M', rows=ar, cols=ar)
        self.declare_partials('K4', 'M', dependent=False)

    # ------------------------------------------------------------------
    # Blending weight: 0 below M_BREAK - h, 1 above M_BREAK + h.
    # Centred on the break here, unlike the lift model: K3 has a genuine
    # value jump (0.00066 -> 0.00035) and no derivative singularity.
    # ------------------------------------------------------------------
    @staticmethod
    def _weight(M):
        t = np.clip((M - (M_BREAK - M_BLEND)) / (2.0 * M_BLEND), 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
        dw = np.where((t > 0.0) & (t < 1.0),
                      6.0 * t * (1.0 - t) / (2.0 * M_BLEND), 0.0)
        return w, dw

    def compute(self, inputs, outputs):
        M = inputs['M']
        w, _ = self._weight(M)

        aD_lo = 17.0 - 23.4 * M          # p. 432
        aD_hi = 0.0                      # p. 433

        outputs['alpha_D'] = (1.0 - w) * aD_lo + w * aD_hi
        outputs['K3'] = (1.0 - w) * K3_LOW + w * K3_HIGH
        outputs['K4'] = K4

        # K6 = 3.2 > 1, so the term is already C1 at the break: clip, no blend.
        x = np.maximum(M - M_BREAK, 0.0)
        outputs['delta_cd_M'] = K5 * x ** K6

    def compute_partials(self, inputs, partials):
        M = inputs['M']
        w, dw = self._weight(M)

        aD_lo = 17.0 - 23.4 * M
        aD_hi = 0.0

        partials['alpha_D', 'M'] = (1.0 - w) * (-23.4) + (aD_hi - aD_lo) * dw
        partials['K3', 'M'] = (K3_HIGH - K3_LOW) * dw

        x = np.maximum(M - M_BREAK, 0.0)
        partials['delta_cd_M', 'M'] = np.where(x > 0.0, K5 * K6 * x ** (K6 - 1.0), 0.0)


if __name__ == '__main__':
    M = np.array([0.1, 0.3, 0.5, 0.7, 0.725, 0.75, 0.80, 0.85])

    p = om.Problem()
    p.model.add_subsystem('comp', DragModelCoefsComp(num_nodes=M.size),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('M', M)
    p.run_model()

    print(f"{'M':>7}{'alpha_D':>10}{'K3':>11}{'K4':>7}{'delta_cd_M':>12}")
    for i, m in enumerate(M):
        print(f"{m:7.3f}{p.get_val('alpha_D')[i]:10.3f}{p.get_val('K3')[i]:11.6f}"
              f"{p.get_val('K4')[i]:7.2f}{p.get_val('delta_cd_M')[i]:12.6f}")

    p.check_partials(method='fd', compact_print=True)
