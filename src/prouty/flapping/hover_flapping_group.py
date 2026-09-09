"""Flapping in hover, including the effect of hinge offset.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 455-462.
"""

import openmdao.api as om

from prouty.flapping.accel_coupling_comp import AccelCouplingComp
from prouty.flapping.blade_inertia_comp import BladeInertiaComp
from prouty.flapping.blade_time_constant_comp import BladeTimeConstantComp
from prouty.flapping.flap_damping_comp import FlapDampingComp
from prouty.flapping.flap_frequency_comp import FlapFrequencyComp
from prouty.flapping.lock_number_comp import LockNumberComp
from prouty.flapping.phase_angle_comp import PhaseAngleComp


class HoverFlappingGroup(om.Group):
    """Frequency ratio, damping, phase lag and cross-coupling of a flapping
    blade with hinge offset.

    Chain of components, all feed-forward, no solver required:

    ==================== ========= ==========================================
    Component            Pages     Produces
    ==================== ========= ==========================================
    BladeInertiaComp     456-457   I_b, M_b/g, M_b, e
    LockNumberComp       458       gamma
    FlapFrequencyComp    456-457   omega_n/Omega   (or e/R, hingeless case)
    FlapDampingComp      458-459   c_damp, c_crit, zeta, omega_n
    PhaseAngleComp       459       phi
    AccelCouplingComp    459-461   b1s/a1s, Delta_alpha/a1s
    BladeTimeConstantComp 462      t_63, psi_63
    ==================== ========= ==========================================

    Options
    -------
    blade_input : {'m', 'I_b'}
        Whether the blade is described by its linear mass density or by its
        flapping inertia. ``gamma`` is always derived from ``rho``, ``a``,
        ``c``, ``R`` and ``I_b`` rather than imposed, because it varies with
        density and so cannot be a single design constant across nodes. To
        impose a Lock number instead, run ``LockNumberComp(mode='I_b')`` at a
        reference condition ahead of this group and feed the resulting
        ``I_b`` in.
    hinge_input : {'e_over_R', 'omega_n_ratio'}
        ``'e_over_R'`` articulated rotor: the offset is given and the
        frequency ratio follows.
        ``'omega_n_ratio'`` hingeless rotor: the frequency ratio comes from a
        blade dynamics analysis and the equivalent offset is recovered from
        p. 457. ``e_over_R`` then becomes a group *output*, feeding the
        inertia, damping and coupling components.

    Notes
    -----
    ``gamma``, and therefore everything downstream of it, varies with air
    density. Only ``I_b``, ``M_b/g``, ``e`` and ``omega_n/Omega`` are
    node-independent.

    Two results printed in the book are not reproduced, on purpose:
    ``b1s/a1s = -0.07`` and ``Delta_alpha/a1s = 0.07`` (pp. 460-461) follow a
    reduced form of the coupling equation that drops ``(1 - e/R)^4``. See
    ``docs/validation_flapping.md``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('blade_input', values=('m', 'I_b'), default='I_b',
                             desc='How the blade mass properties are given.')
        self.options.declare('hinge_input',
                             values=('e_over_R', 'omega_n_ratio'),
                             default='e_over_R',
                             desc='Articulated or hingeless rotor.')

    def setup(self):
        nn = self.options['num_nodes']
        hingeless = self.options['hinge_input'] == 'omega_n_ratio'

        if hingeless:
            self.add_subsystem(
                'frequency', FlapFrequencyComp(mode='e_over_R'),
                promotes_inputs=['omega_n_ratio'],
                promotes_outputs=[('e_over_R_eff', 'e_over_R')])

        self.add_subsystem(
            'inertia', BladeInertiaComp(input_mode=self.options['blade_input']),
            promotes=['*'])
        self.add_subsystem('lock', LockNumberComp(num_nodes=nn), promotes=['*'])

        if not hingeless:
            self.add_subsystem('frequency', FlapFrequencyComp(), promotes=['*'])

        self.add_subsystem('damping', FlapDampingComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('phase', PhaseAngleComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('coupling', AccelCouplingComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('time_constant', BladeTimeConstantComp(num_nodes=nn),
                           promotes=['*'])
