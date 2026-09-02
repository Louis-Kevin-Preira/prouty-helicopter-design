"""
PitchDistComp -- blade pitch over the rotor disc, and its rates.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 213 (pitch) and p. 225 (rates).

    theta          = theta_0 + (r/R) theta_1 - A_1 cos(psi) - B_1 sin(psi)
    theta_dot/Om   = A_1 sin(psi) - B_1 cos(psi)
    theta_ddot/Om^2 = A_1 cos(psi) + B_1 sin(psi)

The rates are exact derivatives of the pitch with respect to azimuth, since
theta_dot = Omega d(theta)/d(psi) for a rotor at constant speed. They are not
needed for the forces themselves but for the unsteady aerodynamics of p. 224,
where the shed vorticity responds to how fast the blade is feathering, and for
the dynamic stall delay of p. 220 through the rate of change of angle of
attack.

A_1 and B_1 are CYCLIC PITCH here, not the flapping combinations of the
closed-form chapter. p. 211 makes the substitution explicit: "the equivalence
of flapping and feathering allows performance calculations to be based on a
rigid rotor whose tip path plane is perpendicular to the shaft and whose
pitching and rolling moments are trimmed out with cyclic pitch". So the disc
does not flap in G2; the cyclic does the work instead, and A_1 and B_1 are the
two states the pitching and rolling moment balance of p. 214 solves for. Do
not connect B1_a1s from G1 here -- that is a different quantity in a different
plane, and it would be wrong by a_1s.

Shape convention for G2: fields carried over the disc are (num_nodes,
num_azimuth, num_radial); quantities that vary with azimuth alone drop the
last axis.

Useful identities, which the tests assert:

    theta at r/R = 1, psi = 270 deg  =  theta_0 + theta_1 + B_1
    azimuth average of theta         =  theta_0 + (r/R) theta_1
    d(theta_dot/Om)/d(psi)           =  theta_ddot/Om^2

The first is the pitch that enters Prouty's retreating tip angle of attack,
alpha_1,270 = theta_0 + theta_1 + B_1 + atan[lambda'/(1 - mu)] (p. 228).

    theta_0, theta_1, A_1, B_1, r_R, psi --> theta, theta_dot_Om, theta_ddot_Om2
"""

import numpy as np
import openmdao.api as om


class PitchDistComp(om.ExplicitComponent):
    """Pitch distribution over the disc and its azimuthal rates, p. 213, 225."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field, sweep = (nn, n_psi, n_r), (nn, n_psi)

        self.add_input('r_R', shape=(n_r,), desc='radial stations')
        self.add_input('psi', shape=(n_psi,), units='rad',
                       desc='azimuth stations')
        self.add_input('theta_0', shape=(nn,), units='rad', desc='collective')
        self.add_input('A_1', shape=(nn,), val=0.0, units='rad',
                       desc='lateral cyclic pitch')
        self.add_input('B_1', shape=(nn,), val=0.0, units='rad',
                       desc='longitudinal cyclic pitch')
        self.add_input('theta_1', val=-0.17453, units='rad',
                       desc='linear blade twist')

        self.add_output('theta', shape=field, units='rad', desc='blade pitch')
        self.add_output('theta_dot_Om', shape=sweep, units='rad',
                        desc='theta_dot / Omega, p. 225')
        self.add_output('theta_ddot_Om2', shape=sweep, units='rad',
                        desc='theta_ddot / Omega^2, p. 225')

        n_field, n_sweep = nn * n_psi * n_r, nn * n_psi
        node_of_field = np.repeat(np.arange(nn), n_psi * n_r)
        node_of_sweep = np.repeat(np.arange(nn), n_psi)

        for name in ('theta_0', 'A_1', 'B_1'):
            self.declare_partials('theta', name, rows=np.arange(n_field),
                                  cols=node_of_field)
        self.declare_partials('theta', 'theta_1', rows=np.arange(n_field),
                              cols=np.zeros(n_field, dtype=int))
        self.declare_partials('theta', 'r_R', rows=np.arange(n_field),
                              cols=np.tile(np.arange(n_r), nn * n_psi))
        self.declare_partials('theta', 'psi', rows=np.arange(n_field),
                              cols=np.repeat(np.tile(np.arange(n_psi), nn), n_r))

        for out in ('theta_dot_Om', 'theta_ddot_Om2'):
            for name in ('A_1', 'B_1'):
                self.declare_partials(out, name, rows=np.arange(n_sweep),
                                      cols=node_of_sweep)
            self.declare_partials(out, 'psi', rows=np.arange(n_sweep),
                                  cols=np.tile(np.arange(n_psi), nn))

    def _trig(self, inputs):
        """sin and cos broadcast to (num_nodes, num_azimuth)."""
        nn = self.options['num_nodes']
        psi = inputs['psi'][np.newaxis, :]
        shape = (nn, psi.shape[1])
        return (np.broadcast_to(np.sin(psi), shape),
                np.broadcast_to(np.cos(psi), shape))

    def compute(self, inputs, outputs):
        sin_psi, cos_psi = self._trig(inputs)
        A_1 = inputs['A_1'][:, np.newaxis]
        B_1 = inputs['B_1'][:, np.newaxis]

        cyclic = -A_1 * cos_psi - B_1 * sin_psi          # (nn, n_psi)
        span = (inputs['theta_0'][:, np.newaxis]
                + inputs['theta_1'][0] * inputs['r_R'][np.newaxis, :])

        outputs['theta'] = span[:, np.newaxis, :] + cyclic[:, :, np.newaxis]
        outputs['theta_dot_Om'] = A_1 * sin_psi - B_1 * cos_psi
        outputs['theta_ddot_Om2'] = A_1 * cos_psi + B_1 * sin_psi

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        sin_psi, cos_psi = self._trig(inputs)
        A_1 = inputs['A_1'][:, np.newaxis]
        B_1 = inputs['B_1'][:, np.newaxis]

        partials['theta', 'theta_0'] = 1.0
        partials['theta', 'theta_1'] = np.tile(inputs['r_R'], nn * n_psi)
        partials['theta', 'r_R'] = inputs['theta_1'][0]
        partials['theta', 'A_1'] = np.repeat(-cos_psi, n_r, axis=1).ravel()
        partials['theta', 'B_1'] = np.repeat(-sin_psi, n_r, axis=1).ravel()
        partials['theta', 'psi'] = np.repeat(
            A_1 * sin_psi - B_1 * cos_psi, n_r, axis=1).ravel()

        partials['theta_dot_Om', 'A_1'] = sin_psi.ravel()
        partials['theta_dot_Om', 'B_1'] = -cos_psi.ravel()
        partials['theta_dot_Om', 'psi'] = (A_1 * cos_psi + B_1 * sin_psi).ravel()

        partials['theta_ddot_Om2', 'A_1'] = cos_psi.ravel()
        partials['theta_ddot_Om2', 'B_1'] = sin_psi.ravel()
        partials['theta_ddot_Om2', 'psi'] = (-A_1 * sin_psi
                                             + B_1 * cos_psi).ravel()
