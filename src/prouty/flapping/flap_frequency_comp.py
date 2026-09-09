"""First flapping frequency ratio, and its inversion for hingeless rotors.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 456-457.
"""

import numpy as np
import openmdao.api as om


class FlapFrequencyComp(om.ExplicitComponent):
    """Ratio of the undamped flapping natural frequency to the rotor speed.

    The hinge offset turns the flapping blade from a system in resonance into
    one whose natural frequency exceeds the rotational frequency. The
    centrifugal restoring moment gives a spring rate (p. 456)::

        k = M_C.F. / beta = Omega^2 (I_b + e M_b/g)
        omega_n = sqrt(k / I_b)

    hence::

        omega_n / Omega = sqrt(1 + e (M_b/g) / I_b)

    The ratio is independent of ``Omega`` and of the flight condition, so it
    is a scalar.

    Options
    -------
    mode : {'frequency', 'e_over_R'}
        ``'frequency'``  compute the ratio from the blade mass properties.
        ``'e_over_R'``  invert it (p. 457). For a hingeless rotor the
        frequency ratio is computed separately by a blade dynamics analysis;
        the equivalent articulated rotor is then defined by::

            (e/R)_eff = 2 [(omega_n/Omega)^2 - 1] / [1 + 2 (omega_n/Omega)^2]

    Notes
    -----
    Only the general form is implemented. Prouty's uniform-mass expression
    ``sqrt(1 + (3/2)(e/R)/(1 - e/R))`` (p. 457) is recovered exactly by feeding
    this component from ``BladeInertiaComp``, whose ``M_b/g`` and ``I_b``
    satisfy ``e (M_b/g) / I_b = (3/2)(e/R)/(1 - e/R)``. Keeping a single path
    also makes the component valid for a non-uniform blade, where the two
    expressions differ.
    """

    def initialize(self):
        self.options.declare('mode', values=('frequency', 'e_over_R'),
                             default='frequency',
                             desc='Which side of the relation is solved for.')

    def setup(self):
        if self.options['mode'] == 'e_over_R':
            self.add_input('omega_n_ratio', val=1.0,
                           desc='Undamped flapping frequency ratio.')
            self.add_output('e_over_R_eff', val=0.0,
                            desc='Equivalent hinge offset ratio.')
            self.declare_partials('e_over_R_eff', 'omega_n_ratio')
        else:
            self.add_input('e', val=0.0, units='ft', desc='Hinge offset.')
            self.add_input('M_b_over_g', val=1.0, units='slug*ft',
                           desc='Blade first mass moment about the hinge.')
            self.add_input('I_b', val=1.0, units='slug*ft**2',
                           desc='Blade flapping inertia about the hinge.')
            self.add_output('omega_n_ratio', val=1.0,
                            desc='Undamped flapping frequency ratio.')
            self.declare_partials('omega_n_ratio', ['e', 'M_b_over_g', 'I_b'])

    def compute(self, inputs, outputs):
        if self.options['mode'] == 'e_over_R':
            nu2 = inputs['omega_n_ratio'] ** 2
            outputs['e_over_R_eff'] = 2.0 * (nu2 - 1.0) / (1.0 + 2.0 * nu2)
        else:
            s = inputs['e'] * inputs['M_b_over_g'] / inputs['I_b']
            outputs['omega_n_ratio'] = np.sqrt(1.0 + s)

    def compute_partials(self, inputs, J):
        if self.options['mode'] == 'e_over_R':
            nu = inputs['omega_n_ratio']
            J['e_over_R_eff', 'omega_n_ratio'] = 12.0 * nu / (1.0 + 2.0 * nu ** 2) ** 2
        else:
            e, M, I = inputs['e'], inputs['M_b_over_g'], inputs['I_b']
            nu = np.sqrt(1.0 + e * M / I)
            J['omega_n_ratio', 'e'] = M / I / (2.0 * nu)
            J['omega_n_ratio', 'M_b_over_g'] = e / I / (2.0 * nu)
            J['omega_n_ratio', 'I_b'] = -e * M / I ** 2 / (2.0 * nu)
