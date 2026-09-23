"""
LiftCoefFwdComp -- NACA 0012 lift coefficient over the full 0-360 deg range.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations", p. 433 (table of
segments), Figure 6.47 p. 434.

Inboard elements on the retreating side see angles of attack up to 360 deg, so
the low-incidence equations are extended by segments:

    alpha <  20 deg          cl = generated coefficient   (LiftCoefComp)
     20 < alpha < 161        cl = 1.15 sin 2 alpha
    161 < alpha < 173        cl = -0.7
    173 < alpha < 187        cl = 0.1 (alpha - 180)
    187 < alpha < 199        cl =  0.7
    199 < alpha < 340        cl = 1.15 sin 2 alpha
    340 < alpha < 360        cl = generated coefficient

The segments are assembled as a cumulative blend

    cl = f0 + sum_k w_k(alpha) * (f_{k+1} - f_k)

with w_k a cubic smoothstep over [b_k - d, b_k + d]. The four interior
boundaries are already continuous in value; the blend only makes them C1. The
two boundaries against the generated branch (20 and 340 deg) carry a genuine
jump of about 0.1 in cl, which the blend removes.

Inputs : alpha_360 [deg] from AlphaWrapComp, cl_gen from LiftCoefComp
Output : cl
"""

import numpy as np
import openmdao.api as om

DEG = np.pi / 180.0
BREAKS = np.array([20.0, 161.0, 173.0, 187.0, 199.0, 340.0])   # p. 433
BLEND = 2.0                                                    # half-width, deg


class LiftCoefFwdComp(om.ExplicitComponent):
    """Segment assembly of the lift coefficient over 0-360 deg."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha_360', shape=(nn,), units='deg',
                       desc='angle of attack wrapped onto [0, 360)')
        self.add_input('cl_gen', shape=(nn,),
                       desc='generated lift coefficient, low incidence')

        self.add_output('cl', shape=(nn,), desc='section lift coefficient')

        ar = np.arange(nn)
        self.declare_partials('cl', 'alpha_360', rows=ar, cols=ar)
        self.declare_partials('cl', 'cl_gen', rows=ar, cols=ar)

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
    def _branches(a, cl_gen):
        """Branch values and their d/d(alpha), ordered f0 .. f6."""
        sin2, cos2 = np.sin(2.0 * a * DEG), np.cos(2.0 * a * DEG)
        zero = np.zeros_like(a)

        f = np.array([cl_gen,                # alpha < 20
                      1.15 * sin2,           # 20  - 161
                      -0.7 * np.ones_like(a),  # 161 - 173
                      0.1 * (a - 180.0),     # 173 - 187
                      0.7 * np.ones_like(a),  # 187 - 199
                      1.15 * sin2,           # 199 - 340
                      cl_gen])               # 340 - 360

        dsin = 2.3 * cos2 * DEG
        df = np.array([zero, dsin, zero, 0.1 * np.ones_like(a), zero, dsin, zero])
        return f, df

    def compute(self, inputs, outputs):
        a, cl_gen = inputs['alpha_360'], inputs['cl_gen']
        w, _ = self._weights(a)
        f, _ = self._branches(a, cl_gen)

        outputs['cl'] = f[0] + np.sum(w * (f[1:] - f[:-1]), axis=0)

    def compute_partials(self, inputs, partials):
        a, cl_gen = inputs['alpha_360'], inputs['cl_gen']
        w, dw = self._weights(a)
        f, df = self._branches(a, cl_gen)

        partials['cl', 'alpha_360'] = df[0] + np.sum(
            dw * (f[1:] - f[:-1]) + w * (df[1:] - df[:-1]), axis=0)

        # only f0 and f6 carry cl_gen: weight 1 - w[0] + w[-1]
        partials['cl', 'cl_gen'] = 1.0 - w[0] + w[-1]


if __name__ == '__main__':
    from prouty.airfoil.alpha_wrap_comp import AlphaWrapComp
    from prouty.airfoil.lift_model_coefs_comp import LiftModelCoefsComp
    from prouty.airfoil.lift_coef_comp import LiftCoefComp

    alpha = np.array([0., 10., 19., 20., 21., 45., 90., 161., 167., 173.,
                      180., 187., 193., 199., 270., 339., 340., 341., 350.])
    nn = alpha.size

    p = om.Problem()
    p.model.add_subsystem('wrap', AlphaWrapComp(num_nodes=nn), promotes=['*'])
    p.model.add_subsystem('coefs', LiftModelCoefsComp(num_nodes=nn), promotes=['*'])
    p.model.add_subsystem('gen', LiftCoefComp(num_nodes=nn),
                          promotes_inputs=['a', 'alpha_L', 'K1', 'K2'],
                          promotes_outputs=[('cl', 'cl_gen')])
    p.model.add_subsystem('seg', LiftCoefFwdComp(num_nodes=nn), promotes=['*'])
    p.model.connect('alpha_sig', 'gen.alpha')

    p.setup(force_alloc_complex=True)
    p.set_val('alpha_raw', alpha)
    p.set_val('M', np.full(nn, 0.3))
    p.run_model()

    print('M = 0.30')
    print(f"{'alpha':>8}{'cl_gen':>10}{'cl':>10}")
    for i, a in enumerate(alpha):
        print(f"{a:8.1f}{p.get_val('cl_gen')[i]:10.4f}{p.get_val('cl')[i]:10.4f}")

    p.check_partials(method='fd', compact_print=True, includes=['seg'])
