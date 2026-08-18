"""
IncompDragFwdComp -- incompressible profile drag of the NACA 0012, forward
flight form.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations", p. 433,
"Equations Suited for Forward Flight Analysis".

Negative angles of attack occur on the advancing tip, so the incompressible
drag must be symmetric about alpha = 0. Prouty replaces the asymmetric hover
series of p. 432 by a four-term series in even powers only:

    cd_incomp = 0.0081 + (65.8 a^2 - 0.226 a^4 + 0.0046 a^6) * 1e-6

with alpha in degrees. Written in u = alpha^2 the expression is a cubic, hence
even by construction: no absolute value is needed and the result is C-infinity
at alpha = 0.

Validity: |alpha| < 20 deg. Beyond that the segment equations of p. 434 take
over (cd = 1.03 - 1.02 cos 2a), which is exactly where this series stops being
used, so its steep alpha^6 growth is never exercised.
"""

import numpy as np
import openmdao.api as om

CD0 = 0.0081                                    # zero-angle drag, p. 433
CSER = np.array([65.8, -0.226, 0.0046])         # coefficients of a^2, a^4, a^6
SCALE = 1.0e-6                                  # p. 433


class IncompDragFwdComp(om.ExplicitComponent):
    """Symmetric incompressible drag coefficient, forward flight form."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha', shape=(nn,), units='deg',
                       desc='signed angle of attack, expects alpha_sig')
        self.add_output('cd_incomp', shape=(nn,),
                        desc='incompressible profile drag coefficient')

        ar = np.arange(nn)
        self.declare_partials('cd_incomp', 'alpha', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        u = inputs['alpha'] ** 2
        c1, c2, c3 = CSER
        outputs['cd_incomp'] = CD0 + SCALE * u * (c1 + u * (c2 + u * c3))

    def compute_partials(self, inputs, partials):
        a = inputs['alpha']
        u = a ** 2
        c1, c2, c3 = CSER
        dfdu = c1 + u * (2.0 * c2 + u * 3.0 * c3)
        partials['cd_incomp', 'alpha'] = SCALE * 2.0 * a * dfdu


if __name__ == '__main__':
    from prouty.airfoil.incomp_drag_hover_comp import IncompDragHoverComp

    alpha = np.array([0.0, 2.0, 6.0, 10.0, 14.7, -2.0, -6.0, -14.7, 19.0])

    p = om.Problem()
    p.model.add_subsystem('fwd', IncompDragFwdComp(num_nodes=alpha.size),
                          promotes=['alpha'])
    p.model.add_subsystem('hov', IncompDragHoverComp(num_nodes=alpha.size),
                          promotes=['alpha'])
    p.setup(force_alloc_complex=True)
    p.set_val('alpha', alpha)
    p.run_model()

    print(f"{'alpha':>8}{'cd fwd':>11}{'cd hover':>11}{'diff':>11}")
    for i, a in enumerate(alpha):
        f = p.get_val('fwd.cd_incomp')[i]
        h = p.get_val('hov.cd_incomp')[i]
        print(f"{a:8.1f}{f:11.6f}{h:11.6f}{f - h:11.6f}")

    p.check_partials(method='cs', compact_print=True, includes=['fwd'])
