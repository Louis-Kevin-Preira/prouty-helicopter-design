"""Blade inertia properties about the flapping hinge.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 456-457.
"""

import openmdao.api as om

G_STANDARD = 32.174  # ft/s**2


class BladeInertiaComp(om.ExplicitComponent):
    """Flapping inertia ``I_b`` and first mass moment ``M_b/g`` of the blade.

    Both quantities refer to the blade portion *outboard of the flapping
    hinge*, integrated over ``r' = r - e`` (p. 456)::

        I_b     = int_0^(R-e) m r'^2 dr'
        M_b / g = int_0^(R-e) m r'   dr'

    For a uniform spanwise mass distribution ``m`` these integrate to (p. 457)::

        I_b     = (m R^3 / 3) (1 - e/R)^3
        M_b / g = (m R^2 / 2) (1 - e/R)^2

    Options
    -------
    input_mode : {'m', 'I_b'}
        ``'m'``    build both quantities from the linear mass density.
        ``'I_b'``  ``I_b`` is supplied (blade dynamics analysis, data sheet)
        and ``M_b/g`` is recovered from the uniform-mass relation implied by
        the two expressions above::

            M_b / g = 3 I_b / (2 R (1 - e/R))

    Notes
    -----
    These are blade properties, independent of the flight condition, hence
    scalars rather than ``(num_nodes,)`` vectors.

    ``M_b`` is the *weight* moment used directly in the coning equation
    (p. 467); ``M_b/g`` is the *mass* moment used in the frequency ratio
    (p. 456) and in the centrifugal hinge moments (p. 464).
    """

    def initialize(self):
        self.options.declare('input_mode', values=('m', 'I_b'), default='m',
                             desc='Source of the blade mass properties.')
        self.options.declare('g', default=G_STANDARD,
                             desc='Gravitational acceleration, ft/s**2.')

    def setup(self):
        self.add_input('R', val=1.0, units='ft', desc='Rotor radius.')
        self.add_input('e_over_R', val=0.0, desc='Hinge offset ratio, e/R.')

        if self.options['input_mode'] == 'm':
            self.add_input('m', val=1.0, units='slug/ft',
                           desc='Blade mass per unit span (uniform).')
        else:
            self.add_input('I_b_ref', val=1.0, units='slug*ft**2',
                           desc='Given blade flapping inertia.')

        self.add_output('I_b', val=1.0, units='slug*ft**2',
                        desc='Blade flapping inertia about the hinge.')
        self.add_output('M_b_over_g', val=1.0, units='slug*ft',
                        desc='Blade first mass moment about the hinge.')
        self.add_output('M_b', val=1.0, units='lbf*ft',
                        desc='Blade first weight moment about the hinge.')
        self.add_output('e', val=0.0, units='ft', desc='Hinge offset.')

        self.declare_partials('*', '*', dependent=False)
        self.declare_partials('e', ['R', 'e_over_R'])
        self.declare_partials(['M_b_over_g', 'M_b'], ['R', 'e_over_R'])

        if self.options['input_mode'] == 'm':
            self.declare_partials(['I_b', 'M_b_over_g', 'M_b'], 'm')
            self.declare_partials('I_b', ['R', 'e_over_R'])
        else:
            self.declare_partials(['M_b_over_g', 'M_b'], 'I_b_ref')
            self.declare_partials('I_b', 'I_b_ref', val=1.0)

    def compute(self, inputs, outputs):
        g = self.options['g']
        R = inputs['R']
        k = 1.0 - inputs['e_over_R']

        if self.options['input_mode'] == 'm':
            m = inputs['m']
            I_b = m * R ** 3 * k ** 3 / 3.0
            M_over_g = m * R ** 2 * k ** 2 / 2.0
        else:
            I_b = inputs['I_b_ref']
            M_over_g = 1.5 * I_b / (R * k)

        outputs['I_b'] = I_b
        outputs['M_b_over_g'] = M_over_g
        outputs['M_b'] = g * M_over_g
        outputs['e'] = inputs['e_over_R'] * R

    def compute_partials(self, inputs, J):
        g = self.options['g']
        R = inputs['R']
        k = 1.0 - inputs['e_over_R']

        if self.options['input_mode'] == 'm':
            m = inputs['m']
            J['I_b', 'm'] = R ** 3 * k ** 3 / 3.0
            J['I_b', 'R'] = m * R ** 2 * k ** 3
            J['I_b', 'e_over_R'] = -m * R ** 3 * k ** 2

            J['M_b_over_g', 'm'] = R ** 2 * k ** 2 / 2.0
            J['M_b_over_g', 'R'] = m * R * k ** 2
            J['M_b_over_g', 'e_over_R'] = -m * R ** 2 * k
        else:
            I_b = inputs['I_b_ref']
            J['M_b_over_g', 'I_b_ref'] = 1.5 / (R * k)
            J['M_b_over_g', 'R'] = -1.5 * I_b / (R ** 2 * k)
            J['M_b_over_g', 'e_over_R'] = 1.5 * I_b / (R * k ** 2)

        src = 'm' if self.options['input_mode'] == 'm' else 'I_b_ref'
        for wrt in (src, 'R', 'e_over_R'):
            J['M_b', wrt] = g * J['M_b_over_g', wrt]

        J['e', 'R'] = inputs['e_over_R']
        J['e', 'e_over_R'] = R
