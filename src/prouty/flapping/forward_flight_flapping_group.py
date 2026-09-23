"""Flapping in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 463-469.
"""

import openmdao.api as om

from prouty.flapping.blade_inertia_comp import BladeInertiaComp
from prouty.flapping.closed_form_flapping_comp import ClosedFormFlappingComp
from prouty.flapping.flapping_matrix_comp import FlappingMatrixComp
from prouty.flapping.flapping_solve_comp import FlappingSolveComp
from prouty.flapping.thrust_inflow_comp import ThrustInflowComp


class ForwardFlightFlappingGroup(om.Group):
    """Coning and first-harmonic flapping of a rotor with hinge offset.

    Two interchangeable routes to the same three angles:

    ``method='numerical'`` (default)
        Assemble the constant, sine and cosine hinge-moment equations with
        their full ``e/R`` brackets (pp. 463-466) and solve the resulting 3x3
        system. Nothing is dropped.

    ``method='closed_form'``
        Evaluate the solved expressions of pp. 467-469. Faster to read
        against the book, but it inherits the p. 466 simplification that
        strips ``e/R`` from inside the brackets, which costs up to 20 % on
        coning and 0.9 degrees on ``a_1s`` at ``mu = 0.45``. Intended for
        auditing, not production.

    ==================== ========= ==========================================
    Component            Pages     Produces
    ==================== ========= ==========================================
    BladeInertiaComp     456-457   I_b, M_b/g, M_b, e
    ThrustInflowComp     467-468   C_T/sigma, v1/(Omega R)
    FlappingMatrixComp   463-466   flap_A, flap_b
    FlappingSolveComp    466-468   a_0, a_1s, b_1s
    ClosedFormFlappingComp 467-469 a_0, a_1s, b_1s  (alternative route)
    ==================== ========= ==========================================

    Options
    -------
    num_nodes : int
    method : {'numerical', 'closed_form'}
    inflow : {'internal', 'external'}
        ``'internal'`` closes the ``C_T/sigma`` and ``v_1`` loop with the
        high-speed momentum relation of p. 468, which is not valid near
        hover. ``'external'`` takes ``v1_over_OmegaR`` as a group input, for
        the Chapter 1 or Chapter 3 inflow.
    include_weight : bool
        Keep the blade weight moment. Default True. p. 468 drops it.
    exact_denominator : bool
        Closed-form route only: keep the ``kappa^2`` term of p. 468.
    blade_input : {'m', 'I_b', 'external'}
        Numerical route only. ``'external'`` expects ``I_b``,
        ``M_b_over_g``, ``M_b`` and ``e`` to be supplied directly and
        consistently; the other two build them with ``BladeInertiaComp``.
        The closed forms never touch the blade inertia, so this option is
        ignored there.

    Notes
    -----
    Feed-forward throughout, no solver. The induced-velocity loop is linear
    and closed algebraically inside ``ThrustInflowComp``; the flapping system
    is linear and solved explicitly.

    Two corrections to the printed text are applied on both routes, C7-3 and
    C7-4 of ``docs/validation_flapping.md``. A third, C7-9, affects the closed
    forms only.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('method', values=('numerical', 'closed_form'),
                             default='numerical')
        self.options.declare('inflow', values=('internal', 'external'),
                             default='internal')
        self.options.declare('include_weight', types=bool, default=True)
        self.options.declare('exact_denominator', types=bool, default=True)
        self.options.declare('blade_input', values=('m', 'I_b', 'external'),
                             default='I_b')

    def setup(self):
        nn = self.options['num_nodes']
        weight = self.options['include_weight']
        numerical = self.options['method'] == 'numerical'
        blade = self.options['blade_input']

        if numerical and blade != 'external':
            self.add_subsystem('inertia', BladeInertiaComp(input_mode=blade),
                               promotes=['*'])

        self.add_subsystem('thrust',
                           ThrustInflowComp(num_nodes=nn,
                                            inflow=self.options['inflow']),
                           promotes=['*'])

        if numerical:
            self.add_subsystem('matrix',
                               FlappingMatrixComp(num_nodes=nn,
                                                  include_weight=weight),
                               promotes=['*'])
            self.add_subsystem('solve', FlappingSolveComp(num_nodes=nn),
                               promotes=['*'])
        else:
            self.add_subsystem(
                'closed',
                ClosedFormFlappingComp(
                    num_nodes=nn, include_weight=weight,
                    exact_denominator=self.options['exact_denominator']),
                promotes=['*'])
