"""
IncompDragHoverComp -- incompressible profile drag of the NACA 0012, hover form.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations", p. 432.

Five-term power series fitted to the M = 0.1 test data at the control points
alpha = 0, 2, 6, 10 and 14.7 degrees:

    cd_incomp = 0.0081 + (-350 a + 396 a^2 - 63.3 a^3 + 3.66 a^4) * 1e-6

with alpha in degrees. The series is asymmetric in alpha and is only fitted
over the positive-alpha range: see IncompDragFwdComp (p. 433) for the
even-power form required in forward flight.
"""

import numpy as np
import openmdao.api as om

CD0 = 0.0081                                     # zero-angle drag, p. 432
CSER = np.array([-350.0, 396.0, -63.3, 3.66])    # coefficients of a^1..a^4
SCALE = 1.0e-6                                   # p. 432


class IncompDragHoverComp(om.ExplicitComponent):
    """Incompressible drag coefficient as a function of angle of attack."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha', shape=(nn,), units='deg', desc='angle of attack')
        self.add_output('cd_incomp', shape=(nn,),
                        desc='incompressible profile drag coefficient')

        ar = np.arange(nn)
        self.declare_partials('cd_incomp', 'alpha', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        a = inputs['alpha']
        c1, c2, c3, c4 = CSER
        series = a * (c1 + a * (c2 + a * (c3 + a * c4)))     # Horner
        outputs['cd_incomp'] = CD0 + SCALE * series

    def compute_partials(self, inputs, partials):
        a = inputs['alpha']
        c1, c2, c3, c4 = CSER
        dseries = c1 + a * (2.0 * c2 + a * (3.0 * c3 + a * 4.0 * c4))
        partials['cd_incomp', 'alpha'] = SCALE * dseries


if __name__ == '__main__':
    alpha = np.array([0.0, 2.0, 6.0, 10.0, 14.7, 4.0, -4.0, 18.0])

    p = om.Problem()
    p.model.add_subsystem('comp', IncompDragHoverComp(num_nodes=alpha.size),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('alpha', alpha)
    p.run_model()

    print(f"{'alpha':>8}{'cd_incomp':>12}")
    for i, al in enumerate(alpha):
        print(f"{al:8.1f}{p.get_val('cd_incomp')[i]:12.6f}")

    p.check_partials(method='cs', compact_print=True)
