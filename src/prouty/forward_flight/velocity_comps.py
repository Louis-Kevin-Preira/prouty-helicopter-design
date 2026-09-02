"""
Blade element velocity components, non-dimensionalised by tip speed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 209 (definitions), p. 211 (radial) and p. 213 (tangential and
perpendicular).

    U_T / Omega R = r/R + mu sin(psi)
    U_P / Omega R = lambda' - (v1/Omega R)(r/R) cos(psi) - mu a_0 cos(psi)
                    + (r/R)(Theta_dot/Omega) cos(psi)
                    + (r/R)(Phi_dot/Omega) sin(psi)
    U_R / Omega R = mu cos(psi)

U_T is the component in the plane of rotation perpendicular to the blade, U_P
the component perpendicular to that plane, U_R the spanwise one. Only the
first two set the angle of attack; U_R enters through the skin friction, which
responds to the magnitude and direction of the TOTAL local velocity rather
than to its chordwise part alone (p. 211).

The reverse flow region falls out of U_T alone: it is where r/R < -mu sin(psi),
a circle of diameter mu sitting on the retreating side with its far edge at
psi = 270 deg. Nothing special is done about it here -- the sign of U_T carries
it, and the quadrant handling of p. 214 turns it into the right angle of
attack downstream.

A misprint worth knowing about. p. 209 states the induced velocity
distribution as v_L = v_i(1 + (r/R) sin psi); p. 213 writes the same term with
cos(psi), and so do p. 165 and p. 225. cos is right, and not merely by weight
of numbers: with psi = 0 over the tail, cos(psi) puts the extra downwash at
the REAR of the disc, which is where the skewed wake actually leaves it. The
sin form would put maximum downwash on the advancing side, which no wake does.

Note the coning term carries no r/R: the velocity a coned blade sees from the
free stream is V sin(a_0) cos(psi), uniform along the span, whereas the linear
inflow term grows outboard.

Theta_dot and Phi_dot are prescribed pitching and rolling rates (p. 213), zero
in steady flight. When they are non-zero the rotor is trimmed against
gyroscopic moments rather than to zero moment -- see p. 214.

Shapes: U_T and U_P are (num_nodes, num_azimuth, num_radial); U_R does not
depend on radius and stays (num_nodes, num_azimuth).

Two identities the tests use:

    U_T at r/R = 1, psi = 90 deg   =  1 + mu, the advancing tip
    U_P at psi = 270 deg           =  lambda', every station

The second is why alpha at the retreating tip reduces to Prouty's
alpha_1,270 = theta_0 + theta_1 + B_1 + atan[lambda'/(1 - mu)] of p. 228.
"""

import numpy as np
import openmdao.api as om

FIELD_ARGS = ('num_nodes', 'num_azimuth', 'num_radial')


class _DiscComp(om.ExplicitComponent):
    """Shared grid handling for the velocity components."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('gradient_phase', values=('cos', 'sin'),
                             default='cos',
                             desc="phase of the linear inflow gradient. "
                                  "'cos' follows p. 165, 213 and 225; 'sin' "
                                  'follows p. 209 as printed. The two are not '
                                  'interchangeable: cos is fore-and-aft and '
                                  'trims through A_1, sin is lateral and '
                                  'trims through B_1.')

    @property
    def _shape(self):
        return tuple(self.options[name] for name in FIELD_ARGS)

    def _index(self):
        """Row and column index sets for the (nn, n_psi, n_r) sparsity."""
        nn, n_psi, n_r = self._shape
        size = nn * n_psi * n_r
        return dict(rows=np.arange(size),
                    node=np.repeat(np.arange(nn), n_psi * n_r),
                    psi=np.repeat(np.tile(np.arange(n_psi), nn), n_r),
                    radial=np.tile(np.arange(n_r), nn * n_psi))

    def _grid(self, inputs):
        r = inputs['r_R'][np.newaxis, np.newaxis, :]
        psi = inputs['psi'][np.newaxis, :, np.newaxis]
        return r, np.sin(psi), np.cos(psi)


class TangentialVelComp(_DiscComp):
    """U_T / Omega R = r/R + mu sin(psi), p. 213."""

    def setup(self):
        nn, n_psi, n_r = self._shape
        idx = self._index()

        self.add_input('r_R', shape=(n_r,))
        self.add_input('psi', shape=(n_psi,), units='rad')
        self.add_input('mu', shape=(nn,), desc='tip speed ratio')

        self.add_output('UT_bar', shape=(nn, n_psi, n_r),
                        desc='U_T / (Omega R)')

        self.declare_partials('UT_bar', 'mu', rows=idx['rows'], cols=idx['node'])
        self.declare_partials('UT_bar', 'psi', rows=idx['rows'], cols=idx['psi'])
        self.declare_partials('UT_bar', 'r_R', rows=idx['rows'],
                              cols=idx['radial'], val=1.0)

    def compute(self, inputs, outputs):
        r, sin_psi, _ = self._grid(inputs)
        outputs['UT_bar'] = r + inputs['mu'][:, np.newaxis, np.newaxis] * sin_psi

    def compute_partials(self, inputs, partials):
        nn, n_psi, n_r = self._shape
        r, sin_psi, cos_psi = self._grid(inputs)
        mu = inputs['mu'][:, np.newaxis, np.newaxis]

        partials['UT_bar', 'mu'] = np.broadcast_to(
            sin_psi, (nn, n_psi, n_r)).ravel()
        partials['UT_bar', 'psi'] = np.broadcast_to(
            mu * cos_psi, (nn, n_psi, n_r)).ravel()


class PerpVelComp(_DiscComp):
    """U_P / Omega R, p. 213, including the prescribed rate terms."""

    def setup(self):
        nn, n_psi, n_r = self._shape
        idx = self._index()

        self.add_input('r_R', shape=(n_r,))
        self.add_input('psi', shape=(n_psi,), units='rad')
        self.add_input('mu', shape=(nn,))
        self.add_input('lambda_p', shape=(nn,), desc="inflow ratio lambda'")
        self.add_input('vi_OR', shape=(nn,), desc='v1 / (Omega R)')
        self.add_input('kappa', val=1.0,
                       desc='multiplier on the linear inflow gradient of '
                            'p. 209; 1 is the printed value, 0 removes the '
                            'gradient entirely. A pure first harmonic, so it '
                            'leaves C_T and lambda alone and drives C_H and '
                            'the lateral cyclic. Diagnostic knob, see '
                            'validation_notes section 4.')
        self.add_input('a0', shape=(nn,), units='rad', desc='coning angle')
        self.add_input('Theta_dot_Om', shape=(nn,), val=0.0,
                       desc='prescribed pitching rate over Omega')
        self.add_input('Phi_dot_Om', shape=(nn,), val=0.0,
                       desc='prescribed rolling rate over Omega')

        self.add_output('UP_bar', shape=(nn, n_psi, n_r),
                        desc='U_P / (Omega R)')

        self.declare_partials('UP_bar', 'kappa',
                              rows=np.arange(nn*n_psi*n_r),
                              cols=np.zeros(nn*n_psi*n_r, dtype=int))
        for name in ('mu', 'lambda_p', 'vi_OR', 'a0', 'Theta_dot_Om',
                     'Phi_dot_Om'):
            self.declare_partials('UP_bar', name, rows=idx['rows'],
                                  cols=idx['node'])
        self.declare_partials('UP_bar', 'psi', rows=idx['rows'], cols=idx['psi'])
        self.declare_partials('UP_bar', 'r_R', rows=idx['rows'],
                              cols=idx['radial'])

    def _phase(self, sin_psi, cos_psi):
        """The gradient's azimuth shape, and the other one for derivatives."""
        if self.options['gradient_phase'] == 'sin':
            return sin_psi, cos_psi
        return cos_psi, -sin_psi

    def _terms(self, inputs):
        r, sin_psi, cos_psi = self._grid(inputs)
        col = lambda name: inputs[name][:, np.newaxis, np.newaxis]
        return r, sin_psi, cos_psi, col

    def compute(self, inputs, outputs):
        r, sin_psi, cos_psi, col = self._terms(inputs)
        phase, _ = self._phase(sin_psi, cos_psi)

        outputs['UP_bar'] = (col('lambda_p')
                             - inputs['kappa'][0] * col('vi_OR') * r * phase
                             - col('mu') * col('a0') * cos_psi
                             + r * col('Theta_dot_Om') * cos_psi
                             + r * col('Phi_dot_Om') * sin_psi)

    def compute_partials(self, inputs, partials):
        shape = (self.options['num_nodes'], self.options['num_azimuth'],
                 self.options['num_radial'])
        r, sin_psi, cos_psi, col = self._terms(inputs)
        phase, d_phase = self._phase(sin_psi, cos_psi)
        full = lambda x: np.broadcast_to(x, shape).ravel()

        partials['UP_bar', 'lambda_p'] = 1.0
        k = inputs['kappa'][0]
        partials['UP_bar', 'vi_OR'] = full(-k * r * phase)
        partials['UP_bar', 'kappa'] = full(-col('vi_OR') * r * phase)
        partials['UP_bar', 'mu'] = full(-col('a0') * cos_psi)
        partials['UP_bar', 'a0'] = full(-col('mu') * cos_psi)
        partials['UP_bar', 'Theta_dot_Om'] = full(r * cos_psi)
        partials['UP_bar', 'Phi_dot_Om'] = full(r * sin_psi)
        partials['UP_bar', 'r_R'] = full(
            -k * col('vi_OR') * phase + col('Theta_dot_Om') * cos_psi
            + col('Phi_dot_Om') * sin_psi)
        partials['UP_bar', 'psi'] = full(
            -k * col('vi_OR') * r * d_phase
            + (col('mu') * col('a0') - r * col('Theta_dot_Om')) * sin_psi
            + r * col('Phi_dot_Om') * cos_psi)


class RadialVelComp(_DiscComp):
    """U_R / Omega R = mu cos(psi), p. 211. No radial dependence."""

    def setup(self):
        nn, n_psi, _ = self._shape
        size = nn * n_psi

        self.add_input('psi', shape=(n_psi,), units='rad')
        self.add_input('mu', shape=(nn,))

        self.add_output('UR_bar', shape=(nn, n_psi), desc='U_R / (Omega R)')

        self.declare_partials('UR_bar', 'mu', rows=np.arange(size),
                              cols=np.repeat(np.arange(nn), n_psi))
        self.declare_partials('UR_bar', 'psi', rows=np.arange(size),
                              cols=np.tile(np.arange(n_psi), nn))

    def compute(self, inputs, outputs):
        outputs['UR_bar'] = (inputs['mu'][:, np.newaxis]
                             * np.cos(inputs['psi'])[np.newaxis, :])

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        psi = inputs['psi'][np.newaxis, :]
        mu = inputs['mu'][:, np.newaxis]

        partials['UR_bar', 'mu'] = np.broadcast_to(
            np.cos(psi), (nn, n_psi)).ravel()
        partials['UR_bar', 'psi'] = np.broadcast_to(
            -mu * np.sin(psi), (nn, n_psi)).ravel()
