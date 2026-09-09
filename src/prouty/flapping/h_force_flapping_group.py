"""H-force and Y-force due to flapping.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 478-479.
"""

import openmdao.api as om

from prouty.flapping.flapping_moment_buildup_comp import \
    FlappingMomentBuildupComp
from prouty.flapping.h_force_flapping_deriv_comp import HForceFlappingDerivComp


class HForceFlappingGroup(om.Group):
    """Effect of inflow on the in-plane force produced by flapping.

    ========================== ========= ====================================
    Component                  Pages     Produces
    ========================== ========= ====================================
    HForceFlappingDerivComp    478-479   lambda_prime, dCHsigma_da1s,
                                         dCYsigma_db1s
    FlappingMomentBuildupComp  479       dMCG_da1s_force, dMCG_da1s,
                                         dLCG_db1s_force, dLCG_db1s
    ========================== ========= ====================================

    Options
    -------
    num_nodes : int
    form : {'lambda', 'theta75'}
        Which of the two p. 479 expressions to evaluate. They are
        algebraically identical; ``'lambda'`` is recommended for forward
        flight, ``'theta75'`` is convenient in hover.
    include_buildup : bool
        Add the p. 479 moment table. Default True. It expects ``dMM_da1s``
        from ``FlappingMomentsGroup``; left unconnected it defaults to zero
        and the totals reduce to the rotor force contribution alone.

    What this section is for
    ------------------------
    Assuming the rotor force stays perpendicular to the tip path plane makes
    the in-plane force ``T a_1s``. p. 478 shows the inflow angle spoils that,
    and p. 479 quantifies it: the derivative becomes ``C_T/sigma +
    (a/8) lambda'``, and ``lambda'`` is negative in forward flight. For the
    example helicopter in hover this cuts the in-plane force derivative from
    20,000 lb to about 9,300, a factor of 2.16, and with it the flapping
    stiffness of the whole aircraft.

    Feeding ``dCHsigma_da1s`` back into ``CGMomentComp(inplane_force=
    'external')`` is how that correction reaches the moment equation of
    p. 476.

    Notes
    -----
    Feed-forward, no solver. The simplification Prouty makes on p. 479 drops
    a term worth about a third of the answer at 115 kt; see entry C7-11 of
    ``docs/validation_flapping.md``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('form', values=('lambda', 'theta75'),
                             default='lambda')
        self.options.declare('include_buildup', types=bool, default=True)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('deriv',
                           HForceFlappingDerivComp(num_nodes=nn,
                                                   form=self.options['form']),
                           promotes=['*'])

        if self.options['include_buildup']:
            self.add_subsystem('buildup',
                               FlappingMomentBuildupComp(num_nodes=nn),
                               promotes=['*'])
