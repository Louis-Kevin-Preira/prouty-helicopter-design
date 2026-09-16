"""Rotor hub stiffness, and the equivalent hinge offset of a hingeless rotor.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 476-477.
"""

import numpy as np
import openmdao.api as om


class RotorStiffnessComp(om.ExplicitComponent):
    """Hub moment per radian of longitudinal flapping.

    Rotor stiffness comes from the vertical component of blade centrifugal
    force acting at the hinge offset (p. 476)::

        C.F._vert = Omega^2 (beta - a_0) M_b/g

    For pure longitudinal flapping ``beta - a_0 = -a_1s cos(psi)``, and
    averaging over azimuth for ``b`` blades (p. 477)::

        M_M = (1/2) e b Omega^2 a_1s M_b/g

    so that the stiffness is::

        dM_M/da_1s = (1/2) e b Omega^2 (M_b/g)

    Options
    -------
    num_nodes : int
    mode : {'stiffness', 'e_over_R'}
        ``'stiffness'`` computes the derivative above. ``'e_over_R'`` inverts
        it: a hingeless rotor's stiffness is known from a mode shape
        analysis, and p. 477 turns it into an equivalent articulated rotor.
    convention : {'hinge', 'center'}
        Which first static moment the uniform-mass shortcut uses. See below.

    The two conventions
    -------------------
    The expression above is exact whatever the mass distribution, provided
    ``M_b/g`` is the moment about the *flapping hinge*, as defined on p. 456::

        M_b/g = int_0^(R-e) m r' dr' = (m R^2/2)(1 - e/R)^2

    p. 477 then prints two uniform-mass shortcuts and calls them equivalent::

        dM_M/da_1s = (1/4)(e/R) b m R (Omega R)^2
                   = (3/4)(e/R) A_b rho R (Omega R)^2 a / gamma

    They are equivalent, but only if ``M_b/g = m R^2/2`` and
    ``I_b = m R^3/3``, that is with both moments taken from the *centre of
    rotation* rather than from the hinge. The ``(1 - e/R)^2`` and
    ``(1 - e/R)^3`` factors of p. 457 are dropped. For the example
    helicopter that makes the printed stiffness 5.9 % low:

    ============================= =====================
    dM_M/da_1s                    ft-lb/rad
    ============================= =====================
    hinge convention, exact       212,732
    centre convention             202,096
    as printed on p. 477          200,940
    ============================= =====================

    ``convention='hinge'`` is the default and follows the derivation.
    ``'center'`` reproduces p. 477, and is written as
    ``(3/4)(e/R) b Omega^2 I_b``, which is algebraically the printed ``A_b``
    form since ``c rho a R^4 = gamma I_b``. See entry C7-6 of
    ``docs/validation_flapping.md``.

    Inversion
    ---------
    With ``K' = (4/3) (dM_M/da_1s) / (b Omega^2 I_b)``::

        centre convention : (e/R)_eff = K'
        hinge convention  : (e/R)_eff = K' / (1 + K')

    The centre form is the one printed on p. 477. The hinge form follows from
    the uniform-mass relation ``M_b/g = (3/2) I_b / [R (1 - e/R)]``, so it
    assumes a uniform blade even though the forward direction does not.

    Notes
    -----
    ``b`` is a continuous input rather than an option, so blade count can be
    swept during preliminary sizing.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', values=('stiffness', 'e_over_R'),
                             default='stiffness')
        self.options.declare('convention', values=('hinge', 'center'),
                             default='hinge')

    def setup(self):
        nn = self.options['num_nodes']
        inverse = self.options['mode'] == 'e_over_R'

        self.add_input('b', val=4.0, desc='Number of blades.')
        self.add_input('Omega', val=np.ones(nn), units='rad/s',
                       desc='Rotor angular speed.')
        self.add_input('I_b', val=1.0, units='slug*ft**2',
                       desc='Blade flapping inertia.')

        if inverse:
            self.add_input('dMM_da1s', val=np.zeros(nn), units='lbf*ft/rad',
                           desc='Rotor stiffness from a blade dynamics run.')
            self.add_output('e_over_R_eff', val=np.zeros(nn),
                            desc='Equivalent hinge offset ratio.')
            out, sources = 'e_over_R_eff', ['dMM_da1s', 'Omega']
            scalars = ['b', 'I_b']
        else:
            self.add_input('e', val=0.0, units='ft', desc='Hinge offset.')
            self.add_input('e_over_R', val=0.0,
                           desc='Hinge offset ratio, e/R.')
            self.add_input('M_b_over_g', val=1.0, units='slug*ft',
                           desc='Blade first mass moment about the hinge.')
            self.add_output('dMM_da1s', val=np.zeros(nn), units='lbf*ft/rad',
                            desc='Rotor stiffness.')
            out, sources = 'dMM_da1s', ['Omega']
            scalars = (['b', 'e', 'M_b_over_g'] if not self._center()
                       else ['b', 'e_over_R', 'I_b'])

        ar = np.arange(nn)
        self.declare_partials(out, sources, rows=ar, cols=ar)
        self.declare_partials(out, scalars)

    def _center(self):
        return self.options['convention'] == 'center'

    def compute(self, inputs, outputs):
        b, Om = inputs['b'], inputs['Omega']

        if self.options['mode'] == 'stiffness':
            if self._center():
                outputs['dMM_da1s'] = (0.75 * inputs['e_over_R'] * b
                                       * Om ** 2 * inputs['I_b'])
            else:
                outputs['dMM_da1s'] = (0.5 * inputs['e'] * b * Om ** 2
                                       * inputs['M_b_over_g'])
        else:
            k = (4.0 / 3.0) * inputs['dMM_da1s'] / (b * Om ** 2 * inputs['I_b'])
            outputs['e_over_R_eff'] = k if self._center() else k / (1.0 + k)

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        b, Om, I_b = inputs['b'], inputs['Omega'], inputs['I_b']

        def col(v):
            return np.broadcast_to(v, (nn,)).reshape(-1, 1).copy()

        if self.options['mode'] == 'stiffness':
            if self._center():
                x = inputs['e_over_R']
                J['dMM_da1s', 'Omega'] = 1.5 * x * b * Om * I_b
                J['dMM_da1s', 'e_over_R'] = col(0.75 * b * Om ** 2 * I_b)
                J['dMM_da1s', 'I_b'] = col(0.75 * x * b * Om ** 2)
                J['dMM_da1s', 'b'] = col(0.75 * x * Om ** 2 * I_b)
            else:
                e, Mg = inputs['e'], inputs['M_b_over_g']
                J['dMM_da1s', 'Omega'] = e * b * Om * Mg
                J['dMM_da1s', 'e'] = col(0.5 * b * Om ** 2 * Mg)
                J['dMM_da1s', 'M_b_over_g'] = col(0.5 * e * b * Om ** 2)
                J['dMM_da1s', 'b'] = col(0.5 * e * Om ** 2 * Mg)
            return

        K = inputs['dMM_da1s']
        den = b * Om ** 2 * I_b
        k = (4.0 / 3.0) * K / den

        dk = {'dMM_da1s': (4.0 / 3.0) / den,
              'Omega': -2.0 * k / Om,
              'b': -k / b,
              'I_b': -k / I_b}

        chain = 1.0 if self._center() else 1.0 / (1.0 + k) ** 2
        for name, val in dk.items():
            out = chain * val
            J['e_over_R_eff', name] = col(out) if name in ('b', 'I_b') else out
