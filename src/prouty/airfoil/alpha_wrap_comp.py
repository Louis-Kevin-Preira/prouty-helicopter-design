"""
AlphaWrapComp -- angle of attack normalisation for forward flight analysis.

Reference: Prouty, "Helicopter Performance, Stability and Control":
  - p. 214 : quadrant convention. Where U_P and U_T are both negative the angle
             lies in the third quadrant and is returned negative by atan2, so
             360 deg must be added before the segment tests are applied.
  - p. 433 : segment bounds (20, 161, 173, 187, 199, 340 deg) are defined on
             the [0, 360) interval.

Two complementary normalisations are produced:

    alpha_360 in [0, 360)     -> segment selection (p. 433-434)
    alpha_sig in [-180, 180)  -> low-incidence equations reused from the hover
                                 components, which expect a signed angle

Both are pure modulo operations, so d/d(alpha_raw) = 1 almost everywhere.
"""

import numpy as np
import openmdao.api as om


class AlphaWrapComp(om.ExplicitComponent):
    """Wrap a raw angle of attack onto [0, 360) and [-180, 180)."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('alpha_raw', shape=(nn,), units='deg',
                       desc='angle of attack as returned by atan2')

        self.add_output('alpha_360', shape=(nn,), units='deg',
                        desc='angle of attack wrapped onto [0, 360)')
        self.add_output('alpha_sig', shape=(nn,), units='deg',
                        desc='angle of attack wrapped onto [-180, 180)')

        # Unit slope everywhere except on the fold points (measure zero).
        for out in ('alpha_360', 'alpha_sig'):
            self.declare_partials(out, 'alpha_raw', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        a360 = np.mod(inputs['alpha_raw'], 360.0)
        outputs['alpha_360'] = a360
        outputs['alpha_sig'] = np.mod(a360 + 180.0, 360.0) - 180.0


if __name__ == '__main__':
    alpha_raw = np.array([0.0, 8.0, -10.0, -135.0, 225.0, 350.0, 400.0, 180.0])

    p = om.Problem()
    p.model.add_subsystem('comp', AlphaWrapComp(num_nodes=alpha_raw.size),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('alpha_raw', alpha_raw)
    p.run_model()

    print(f"{'alpha_raw':>11}{'alpha_360':>11}{'alpha_sig':>11}")
    for i, a in enumerate(alpha_raw):
        print(f"{a:11.1f}{p.get_val('alpha_360')[i]:11.1f}"
              f"{p.get_val('alpha_sig')[i]:11.1f}")

    p.check_partials(method='fd', compact_print=True)
