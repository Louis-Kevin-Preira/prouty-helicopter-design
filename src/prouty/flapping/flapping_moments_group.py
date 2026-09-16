"""Moments produced by rotor flapping.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 476-477 and 479.
"""

import openmdao.api as om

from prouty.flapping.blade_inertia_comp import BladeInertiaComp
from prouty.flapping.cg_moment_comp import CGMomentComp
from prouty.flapping.hub_moment_comp import HubMomentComp
from prouty.flapping.rotor_stiffness_comp import RotorStiffnessComp


class FlappingMomentsGroup(om.Group):
    """Rotor stiffness, hub couple and the resulting moment about the c.g.

    ==================== ========= ==========================================
    Component            Pages     Produces
    ==================== ========= ==========================================
    BladeInertiaComp     456-457   I_b, M_b/g, M_b, e
    RotorStiffnessComp   477       dMM_da1s
    HubMomentComp        476-477   M_M, L_M
    CGMomentComp         476, 479  M_CG, L_CG
    ==================== ========= ==========================================

    Options
    -------
    num_nodes : int
    blade_input : {'m', 'I_b', 'external'}
        ``'external'`` expects ``I_b``, ``M_b_over_g`` and ``e`` to be
        supplied directly; the other two build them with
        ``BladeInertiaComp`` from ``R`` and ``e_over_R``.
    convention : {'hinge', 'center'}
        Which first static moment the rotor stiffness uses. ``'hinge'``
        follows the derivation of p. 477, ``'center'`` reproduces the printed
        shortcuts, which are 1/(1 - e/R) lower. See entry C7-6 of
        ``docs/validation_flapping.md``.
    inplane_force : {'simple', 'external'}
        ``'simple'`` builds the in-plane rotor force as ``T a_1s + H_0``
        (p. 476). ``'external'`` takes ``H`` and ``Y`` directly, which is how
        the inflow correction of p. 479 gets applied. That correction matters:
        it more than halves the rotor force contribution in hover.

    Notes
    -----
    Feed-forward, no solver. The flapping angles ``a_1s`` and ``b_1s`` are
    group inputs, meant to come from ``ForwardFlightFlappingGroup`` or from
    the totals of ``RateFlappingGroup``.

    Sizing a hingeless rotor works the other way round, from a known hub
    stiffness to an equivalent offset. That is
    ``RotorStiffnessComp(mode='e_over_R')`` used on its own, not through this
    group.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('blade_input', values=('m', 'I_b', 'external'),
                             default='I_b')
        self.options.declare('convention', values=('hinge', 'center'),
                             default='hinge')
        self.options.declare('inplane_force', values=('simple', 'external'),
                             default='simple')

    def setup(self):
        nn = self.options['num_nodes']
        blade = self.options['blade_input']

        if blade != 'external':
            self.add_subsystem('inertia', BladeInertiaComp(input_mode=blade),
                               promotes=['*'])

        self.add_subsystem(
            'stiffness',
            RotorStiffnessComp(num_nodes=nn,
                               convention=self.options['convention']),
            promotes=['*'])
        self.add_subsystem('hub', HubMomentComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem(
            'cg',
            CGMomentComp(num_nodes=nn,
                         inplane_force=self.options['inplane_force']),
            promotes=['*'])
