"""Linear system for coning and first-harmonic flapping in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 463-468.
"""

import numpy as np
import openmdao.api as om


class FlappingMatrixComp(om.ExplicitComponent):
    """Assemble the 3x3 system whose solution is ``[a_0, a_1s, b_1s]``.

    Setting the constant, sine and cosine components of the hinge moment to
    zero gives three equations (pp. 466-468)::

        M_CF,const  + M_A,const  + M_W,const = 0
        M_CF,sine   + M_A,sine                = 0
        M_CF,cosine + M_A,cosine              = 0

    with the centrifugal and weight contributions of pp. 464 and 466::

        M_CF,const  = -Omega^2 a_0 (I_b + e M_b/g)
        M_CF,sine   =  Omega^2 b_1s e M_b/g
        M_CF,cosine =  Omega^2 a_1s e M_b/g
        M_W,const   = -M_b

    and the aerodynamic coefficients of pp. 465-466, kept with their full
    ``e/R`` brackets.

    The three equations are coupled: ``M_A,const`` carries a ``mu (e/R)
    a_1s/4`` term and ``M_A,cosine`` a ``-mu a_0 [1/3 + e/6R]`` term, so
    coning cannot be solved separately. Hence a 3x3 rather than the 2x2 the
    book's closed forms suggest.

    Scaling
    -------
    Every equation is divided by ``Omega^2 I_b``, which makes the system
    dimensionless and O(1). Three groupings appear::

        G = gamma (1 - e/R)^2 / 2       aerodynamic weight
        h = e (M_b/g) / I_b             hinge stiffness, equal to
                                        (omega_n/Omega)^2 - 1 (p. 456)
        W = M_b / (Omega^2 I_b)         weight of the blade

    The matrix is then::

        [ -(1+h)          G mu (e/R)/4   0     ] [a_0 ]   [ W - G C_const ]
        [  0             -G S_a          h     ] [a_1s] = [   - G C_sine  ]
        [ -G mu p1        h              G C_b ] [b_1s]   [   - G C_cos   ]

    Options
    -------
    num_nodes : int
    include_weight : bool
        Keep ``M_W`` (default). Setting it False drops the ``W`` term, which
        is what the closed forms of p. 468 do, and makes the two paths
        directly comparable.

    Notes
    -----
    The ``v_1/(Omega R)`` coefficient of the cosine equation is taken as
    ``1/4`` from p. 466, not the ``1/3`` printed on p. 468. See
    ``docs/validation_flapping.md``, entry C7-3: only the 1/4 reproduces the
    Chapter 3 relation that Prouty himself quotes on p. 474.

    Blade pitch follows the p. 464 convention, measured from the flapping
    hinge: ``theta = theta_0 + (r'/R) theta_1 - A_1 cos(psi) - B_1 sin(psi)``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('include_weight', types=bool, default=True,
                             desc='Keep the blade weight moment M_W.')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('I_b', val=1.0, units='slug*ft**2',
                       desc='Blade flapping inertia about the hinge.')
        self.add_input('M_b_over_g', val=1.0, units='slug*ft',
                       desc='Blade first mass moment about the hinge.')
        self.add_input('M_b', val=0.0, units='lbf*ft',
                       desc='Blade first weight moment about the hinge.')
        self.add_input('e', val=0.0, units='ft', desc='Hinge offset.')
        self.add_input('e_over_R', val=0.0, desc='Hinge offset ratio, e/R.')

        self.add_input('Omega', val=np.ones(nn), units='rad/s',
                       desc='Rotor angular speed.')
        self.add_input('gamma', val=np.full(nn, 8.0), desc='Lock number.')
        self.add_input('mu', val=np.full(nn, 0.3), desc='Advance ratio.')
        self.add_input('theta_0', val=np.zeros(nn), units='rad',
                       desc='Collective pitch at the flapping hinge.')
        self.add_input('theta_1', val=np.zeros(nn), units='rad',
                       desc='Linear blade twist.')
        self.add_input('alpha_s', val=np.zeros(nn), units='rad',
                       desc='Shaft angle of attack.')
        self.add_input('A_1', val=np.zeros(nn), units='rad',
                       desc='Lateral cyclic pitch.')
        self.add_input('B_1', val=np.zeros(nn), units='rad',
                       desc='Longitudinal cyclic pitch.')
        self.add_input('v1_over_OmegaR', val=np.zeros(nn),
                       desc='Mean induced velocity ratio.')

        self.add_output('flap_A', val=np.zeros((nn, 3, 3)),
                        desc='System matrix, rows are const/sine/cosine.')
        self.add_output('flap_b', val=np.zeros((nn, 3)),
                        desc='System right hand side.')

        # flap_A involves only the blade properties and the aerodynamic weight
        self.declare_partials('flap_A', ['I_b', 'M_b_over_g', 'e', 'e_over_R'])
        self.declare_partials('flap_A', ['gamma', 'mu'],
                              rows=np.arange(nn * 9),
                              cols=np.repeat(np.arange(nn), 9))

        # flap_b carries the controls, the attitude and the weight moment
        b_scalars = ['e_over_R']
        b_vectors = ['gamma', 'mu', 'theta_0', 'theta_1', 'alpha_s', 'A_1',
                     'B_1', 'v1_over_OmegaR']
        if self.options['include_weight']:
            b_scalars += ['I_b', 'M_b']
            b_vectors += ['Omega']

        self.declare_partials('flap_b', b_scalars)
        self.declare_partials('flap_b', b_vectors,
                              rows=np.arange(nn * 3),
                              cols=np.repeat(np.arange(nn), 3))

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _polys(x, mu):
        """Bracket polynomials of pp. 465-466 and their e/R derivatives."""
        p = {
            'p1': 1.0 / 3.0 + x / 6.0,
            'p2': 0.25 - x / 6.0 - x ** 2 / 12.0,
            'p3': 0.25 + x / 6.0 + x ** 2 / 12.0,
            'p4': 1.0 + mu ** 2 + 2.0 * x / 3.0 + x ** 2 / 3.0,
            'p5': (0.2 + mu ** 2 / 6.0 * (1.0 - x) - x / 10.0
                   - x ** 2 / 15.0 - x ** 3 / 30.0),
        }
        dp = {
            'p1': 1.0 / 6.0,
            'p2': -1.0 / 6.0 - x / 6.0,
            'p3': 1.0 / 6.0 + x / 6.0,
            'p4': 2.0 / 3.0 + 2.0 * x / 3.0,
            'p5': -mu ** 2 / 6.0 - 0.1 - 2.0 * x / 15.0 - x ** 2 / 10.0,
        }
        return p, dp

    def _state(self, inputs):
        """Everything both compute and compute_partials need."""
        x, mu = inputs['e_over_R'], inputs['mu']
        p, dp = self._polys(x, mu)

        s = {'x': x, 'mu': mu, 'p': p, 'dp': dp}
        s['G'] = inputs['gamma'] * (1.0 - x) ** 2 / 2.0
        s['h'] = inputs['e'] * inputs['M_b_over_g'] / inputs['I_b']
        s['W'] = (inputs['M_b'] / (inputs['Omega'] ** 2 * inputs['I_b'])
                  if self.options['include_weight'] else 0.0 * mu)

        s['S_a'] = p['p2'] - mu ** 2 / 8.0
        s['C_b'] = p['p2'] + mu ** 2 / 8.0
        s['lam'] = mu * inputs['alpha_s'] - inputs['v1_over_OmegaR']

        th0, th1 = inputs['theta_0'], inputs['theta_1']
        a1, b1 = inputs['A_1'], inputs['B_1']

        s['C_const'] = (th0 * p['p4'] / 4.0 + th1 * p['p5']
                        + (s['lam'] - mu * b1) * p['p1'])
        s['C_sine'] = (2.0 * th0 * mu * p['p1'] + 2.0 * th1 * mu * p['p2']
                       - b1 * (p['p3'] + 3.0 * mu ** 2 / 8.0)
                       + mu / 2.0 * s['lam'])
        s['C_cos'] = (-a1 * (p['p3'] + mu ** 2 / 8.0)
                      - inputs['v1_over_OmegaR'] * p['p2'])
        return s

    # ------------------------------------------------------------------
    def compute(self, inputs, outputs):
        s = self._state(inputs)
        G, h, mu, x = s['G'], s['h'], s['mu'], s['x']

        A = outputs['flap_A']
        A[:] = 0.0
        A[:, 0, 0] = -(1.0 + h)
        A[:, 0, 1] = G * mu * x / 4.0
        A[:, 1, 1] = -G * s['S_a']
        A[:, 1, 2] = h
        A[:, 2, 0] = -G * mu * s['p']['p1']
        A[:, 2, 1] = h
        A[:, 2, 2] = G * s['C_b']

        b = outputs['flap_b']
        b[:, 0] = s['W'] - G * s['C_const']
        b[:, 1] = -G * s['C_sine']
        b[:, 2] = -G * s['C_cos']

    # ------------------------------------------------------------------
    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        s = self._state(inputs)
        G, h, mu, x = s['G'], s['h'], s['mu'], s['x']
        p, dp = s['p'], s['dp']
        th0, th1 = inputs['theta_0'], inputs['theta_1']
        a1, b1, als = inputs['A_1'], inputs['B_1'], inputs['alpha_s']
        I_b, Om = inputs['I_b'], inputs['Omega']
        weight = self.options['include_weight']

        def zeros_A():
            return np.zeros((nn, 3, 3))

        def zeros_b():
            return np.zeros((nn, 3))

        def put(name, arr, scalar=False):
            key = 'flap_A' if arr.ndim == 3 else 'flap_b'
            J[key, name] = arr.reshape(-1, 1) if scalar else arr.ravel()

        # --- gamma: enters through G, which scales every aerodynamic term --
        unit = (1.0 - x) ** 2 / 2.0
        dA = zeros_A()
        dA[:, 0, 1] = unit * mu * x / 4.0
        dA[:, 1, 1] = -unit * s['S_a']
        dA[:, 2, 0] = -unit * mu * p['p1']
        dA[:, 2, 2] = unit * s['C_b']
        put('gamma', dA)

        db = zeros_b()
        db[:, 0] = -unit * s['C_const']
        db[:, 1] = -unit * s['C_sine']
        db[:, 2] = -unit * s['C_cos']
        put('gamma', db)

        # --- blade properties: h in the matrix, W in the right hand side ---
        for name, dh in (('e', inputs['M_b_over_g'] / I_b),
                         ('M_b_over_g', inputs['e'] / I_b),
                         ('I_b', -h / I_b)):
            dA = zeros_A()
            dA[:, 0, 0] = -dh
            dA[:, 1, 2] = dh
            dA[:, 2, 1] = dh
            put(name, dA, scalar=True)

        if weight:
            for name, dW in (('M_b', 1.0 / (Om ** 2 * I_b)),
                             ('I_b', -s['W'] / I_b)):
                db = zeros_b()
                db[:, 0] = dW
                put(name, db, scalar=True)

            db = zeros_b()
            db[:, 0] = -2.0 * s['W'] / Om
            put('Omega', db)

        # --- e/R: through G and through every bracket ----------------------
        dG = -inputs['gamma'] * (1.0 - x)
        dA = zeros_A()
        dA[:, 0, 1] = (dG * x + G) * mu / 4.0
        dA[:, 1, 1] = -(dG * s['S_a'] + G * dp['p2'])
        dA[:, 2, 0] = -mu * (dG * p['p1'] + G * dp['p1'])
        dA[:, 2, 2] = dG * s['C_b'] + G * dp['p2']
        put('e_over_R', dA, scalar=True)

        db = zeros_b()
        db[:, 0] = -(dG * s['C_const'] + G * (
            th0 * dp['p4'] / 4.0 + th1 * dp['p5']
            + (s['lam'] - mu * b1) * dp['p1']))
        db[:, 1] = -(dG * s['C_sine'] + G * (
            2.0 * th0 * mu * dp['p1'] + 2.0 * th1 * mu * dp['p2']
            - b1 * dp['p3']))
        db[:, 2] = -(dG * s['C_cos'] + G * (
            -a1 * dp['p3'] - inputs['v1_over_OmegaR'] * dp['p2']))
        put('e_over_R', db)

        # --- mu -------------------------------------------------------------
        dA = zeros_A()
        dA[:, 0, 1] = G * x / 4.0
        dA[:, 1, 1] = G * mu / 4.0
        dA[:, 2, 0] = -G * p['p1']
        dA[:, 2, 2] = G * mu / 4.0
        put('mu', dA)

        db = zeros_b()
        db[:, 0] = -G * (th0 * mu / 2.0 + th1 * mu * (1.0 - x) / 3.0
                         + (als - b1) * p['p1'])
        db[:, 1] = -G * (2.0 * th0 * p['p1'] + 2.0 * th1 * p['p2']
                         - 0.75 * b1 * mu + s['lam'] / 2.0 + mu * als / 2.0)
        db[:, 2] = -G * (-a1 * mu / 4.0)
        put('mu', db)

        # --- linear in the controls and the attitude ------------------------
        for name, dC, dS, dK in (
                ('theta_0', p['p4'] / 4.0, 2.0 * mu * p['p1'], 0.0),
                ('theta_1', p['p5'], 2.0 * mu * p['p2'], 0.0),
                ('alpha_s', mu * p['p1'], mu ** 2 / 2.0, 0.0),
                ('B_1', -mu * p['p1'], -(p['p3'] + 3.0 * mu ** 2 / 8.0), 0.0),
                ('A_1', 0.0, 0.0, -(p['p3'] + mu ** 2 / 8.0)),
                ('v1_over_OmegaR', -p['p1'], -mu / 2.0, -p['p2'])):
            db = zeros_b()
            db[:, 0] = -G * dC
            db[:, 1] = -G * dS
            db[:, 2] = -G * dK
            put(name, db)
