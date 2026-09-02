"""
DragCoefHoverComp -- NACA 0012 drag coefficient, hover form.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations":
  - p. 432 : cd below drag divergence,
             cd = cd_incomp + 0.00066 (alpha - 17 + 23.4 M)**2.54
  - p. 433 : cd above drag divergence,
             cd = cd_incomp + 0.00035 alpha**2.54 + 21 (M - 0.725)**3.2

Both regimes collapse into a single expression once alpha_D, K3, K4 and the
zero-angle increment delta_cd_M are supplied by DragModelCoefsComp
(alpha_D = 0 above the break, so (alpha - alpha_D)**K4 reduces to alpha**K4):

    cd = cd_incomp + K3 * max(alpha - alpha_D, 0)**K4 + delta_cd_M

alpha in degrees.
"""

import numpy as np
import openmdao.api as om

X_TINY = 1.0e-12     # guard for x**(K4-1) and log(x) at x = 0
X_MAX = 200.0        # ceiling on the drag rise excess; see lift_coef_comp


class DragCoefHoverComp(om.ExplicitComponent):
    """Assemble cd from cd_incomp and the Mach-dependent model coefficients."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare(
            'symmetric_stall', types=bool, default=False,
            desc='Use |alpha| in the drag rise term so negative angles also '
                 'trigger it. False reproduces p. 432 literally, which is valid '
                 'in hover where alpha stays positive over most of the disc.')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha', shape=(nn,), units='deg', desc='angle of attack')
        self.add_input('cd_incomp', shape=(nn,), desc='incompressible drag')
        self.add_input('alpha_D', shape=(nn,), units='deg',
                       desc='drag divergence angle')
        self.add_input('K3', shape=(nn,), desc='drag rise factor')
        self.add_input('K4', shape=(nn,), desc='drag rise exponent')
        self.add_input('delta_cd_M', shape=(nn,),
                       desc='zero-angle compressibility increment')

        self.add_output('cd', shape=(nn,), desc='section drag coefficient')

        ar = np.arange(nn)
        for var in ('alpha', 'cd_incomp', 'alpha_D', 'K3', 'K4', 'delta_cd_M'):
            self.declare_partials('cd', var, rows=ar, cols=ar)

    def _excess(self, inputs):
        """Angle in excess of drag divergence, and d(reference angle)/d(alpha)."""
        alpha = inputs['alpha']
        if self.options['symmetric_stall']:
            ref, dref = np.abs(alpha), np.sign(alpha)
        else:
            ref, dref = alpha, np.ones_like(alpha)
        raw = ref - inputs['alpha_D']
        x = np.clip(np.real(raw), 0.0, X_MAX)
        x = np.where((np.real(raw) > 0.0) & (np.real(raw) < X_MAX), raw, x)
        return x, np.where(np.real(raw) < X_MAX, dref, 0.0)

    def compute(self, inputs, outputs):
        x, _ = self._excess(inputs)
        K4 = inputs['K4']

        rise = np.where(x > 0.0, inputs['K3'] * np.maximum(x, X_TINY) ** K4, 0.0)
        outputs['cd'] = inputs['cd_incomp'] + rise + inputs['delta_cd_M']

    def compute_partials(self, inputs, partials):
        x, dref = self._excess(inputs)
        K3, K4 = inputs['K3'], inputs['K4']
        xs = np.maximum(x, X_TINY)
        # see LiftCoefComp: existence and mobility of the drag rise are two
        # different conditions once the excess is clipped
        active = np.real(x) > 0.0
        moving = active & (np.real(x) < X_MAX)

        xK4 = np.where(active, xs ** K4, 0.0)                 # (a - aD)**K4
        dxK4 = np.where(moving, K4 * xs ** (K4 - 1.0), 0.0)   # d/dx of the above

        partials['cd', 'alpha'] = K3 * dxK4 * dref
        partials['cd', 'alpha_D'] = -K3 * dxK4
        partials['cd', 'K3'] = xK4
        partials['cd', 'K4'] = K3 * xK4 * np.where(active, np.log(xs), 0.0)
        partials['cd', 'cd_incomp'] = 1.0
        partials['cd', 'delta_cd_M'] = 1.0


if __name__ == '__main__':
    from prouty.airfoil.drag_model_coefs_comp import DragModelCoefsComp
    from prouty.airfoil.incomp_drag_hover_comp import IncompDragHoverComp

    M = np.array([0.2, 0.5, 0.5, 0.5, 0.7, 0.725, 0.80, 0.80])
    alpha = np.array([4.0, 4.0, 8.0, 10.0, 8.0, 8.0, 0.0, 4.0])

    p = om.Problem()
    p.model.add_subsystem('coefs', DragModelCoefsComp(num_nodes=M.size),
                          promotes=['*'])
    p.model.add_subsystem('incomp', IncompDragHoverComp(num_nodes=M.size),
                          promotes=['*'])
    p.model.add_subsystem('drag', DragCoefHoverComp(num_nodes=M.size),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('M', M)
    p.set_val('alpha', alpha)
    p.run_model()

    print(f"{'M':>6}{'alpha':>7}{'alpha_D':>9}{'cd_incomp':>11}{'cd':>10}")
    for i in range(M.size):
        print(f"{M[i]:6.2f}{alpha[i]:7.1f}{p.get_val('alpha_D')[i]:9.2f}"
              f"{p.get_val('cd_incomp')[i]:11.6f}{p.get_val('cd')[i]:10.6f}")

    # FD, not CS: np.maximum / np.where are not complex-step safe.
    p.check_partials(method='fd', compact_print=True, includes=['drag'])
