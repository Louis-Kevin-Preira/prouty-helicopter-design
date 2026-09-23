"""
DragCoefFwdComp -- NACA 0012 drag coefficient over the full 0-360 deg range.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations", p. 434 (table of
segments), Figure 6.47 p. 434.

    alpha <  20 deg          cd = generated coefficient   (DragCoefHoverComp)
     20 < alpha < 340        cd = 1.03 - 1.02 cos 2 alpha
    340 < alpha < 360        cd = generated coefficient

Same cumulative blend as LiftCoefFwdComp, with two breakpoints only:

    cd = f0 + sum_k w_k(alpha) * (f_{k+1} - f_k)

Both boundaries sit against the generated branch and carry a real jump, so the
blend is doing genuine work here, unlike the interior lift boundaries.

Inputs : alpha_360 [deg] from AlphaWrapComp, cd_gen from DragCoefHoverComp
         (fed with alpha_sig and symmetric_stall=True)
Output : cd
"""

import numpy as np
import openmdao.api as om

DEG = np.pi / 180.0
BREAKS = np.array([20.0, 340.0])       # p. 434
BLEND = 2.0                            # half-width, deg


class DragCoefFwdComp(om.ExplicitComponent):
    """Segment assembly of the drag coefficient over 0-360 deg."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha_360', shape=(nn,), units='deg',
                       desc='angle of attack wrapped onto [0, 360)')
        self.add_input('cd_gen', shape=(nn,),
                       desc='generated drag coefficient, low incidence')

        self.add_output('cd', shape=(nn,), desc='section drag coefficient')

        ar = np.arange(nn)
        self.declare_partials('cd', 'alpha_360', rows=ar, cols=ar)
        self.declare_partials('cd', 'cd_gen', rows=ar, cols=ar)

    @staticmethod
    def _weights(a):
        """Cubic smoothstep weight and slope for each breakpoint."""
        t = np.clip((a[None, :] - (BREAKS[:, None] - BLEND)) / (2.0 * BLEND),
                    0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
        dw = np.where((t > 0.0) & (t < 1.0),
                      6.0 * t * (1.0 - t) / (2.0 * BLEND), 0.0)
        return w, dw

    @staticmethod
    def _branches(a, cd_gen):
        """Branch values and their d/d(alpha), ordered f0 .. f2."""
        zero = np.zeros_like(a)
        flat = 1.03 - 1.02 * np.cos(2.0 * a * DEG)
        dflat = 2.04 * np.sin(2.0 * a * DEG) * DEG

        f = np.array([cd_gen, flat, cd_gen])
        df = np.array([zero, dflat, zero])
        return f, df

    def compute(self, inputs, outputs):
        a, cd_gen = inputs['alpha_360'], inputs['cd_gen']
        w, _ = self._weights(a)
        f, _ = self._branches(a, cd_gen)

        outputs['cd'] = f[0] + np.sum(w * (f[1:] - f[:-1]), axis=0)

    def compute_partials(self, inputs, partials):
        a, cd_gen = inputs['alpha_360'], inputs['cd_gen']
        w, dw = self._weights(a)
        f, df = self._branches(a, cd_gen)

        partials['cd', 'alpha_360'] = df[0] + np.sum(
            dw * (f[1:] - f[:-1]) + w * (df[1:] - df[:-1]), axis=0)

        partials['cd', 'cd_gen'] = 1.0 - w[0] + w[-1]


if __name__ == '__main__':
    from prouty.airfoil.alpha_wrap_comp import AlphaWrapComp
    from prouty.airfoil.drag_model_coefs_comp import DragModelCoefsComp
    from prouty.airfoil.incomp_drag_fwd_comp import IncompDragFwdComp
    from prouty.airfoil.drag_coef_hover_comp import DragCoefHoverComp

    alpha = np.array([0., 10., 19., 20., 21., 45., 90., 135., 180., 270.,
                      339., 340., 341., 350.])
    nn = alpha.size

    p = om.Problem()
    p.model.add_subsystem('wrap', AlphaWrapComp(num_nodes=nn), promotes=['*'])
    p.model.add_subsystem('coefs', DragModelCoefsComp(num_nodes=nn), promotes=['*'])
    p.model.add_subsystem('incomp', IncompDragFwdComp(num_nodes=nn),
                          promotes_outputs=['cd_incomp'])
    p.model.add_subsystem(
        'gen', DragCoefHoverComp(num_nodes=nn, symmetric_stall=True),
        promotes_inputs=['cd_incomp', 'alpha_D', 'K3', 'K4', 'delta_cd_M'],
        promotes_outputs=[('cd', 'cd_gen')])
    p.model.add_subsystem('seg', DragCoefFwdComp(num_nodes=nn), promotes=['*'])
    p.model.connect('alpha_sig', ['incomp.alpha', 'gen.alpha'])

    p.setup(force_alloc_complex=True)
    p.set_val('alpha_raw', alpha)
    p.set_val('M', np.full(nn, 0.3))
    p.run_model()

    print('M = 0.30')
    print(f"{'alpha':>8}{'cd_gen':>10}{'cd':>10}")
    for i, a in enumerate(alpha):
        print(f"{a:8.1f}{p.get_val('cd_gen')[i]:10.4f}{p.get_val('cd')[i]:10.4f}")

    p.check_partials(method='fd', compact_print=True, includes=['seg'])
