"""
GyroMomentComp -- gyroscopic hub moments from prescribed pitch and roll rates.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 213-214.

A rotor turning at Omega and forced to rotate about a transverse axis reacts
with a gyroscopic moment about the perpendicular axis. For a flat disc of
polar inertia J rolling at Phi_dot, that moment is J Phi_dot Omega; for a
single blade it is 2 I_b Phi_dot Omega (p. 213). Non-dimensionalising and
substituting the Lock number through I_b = c rho a R^4 / gamma collapses it to

    C_M/sigma_gyro = 2 a (Phi_dot/Omega) / gamma
    C_R/sigma_gyro = 2 a (Theta_dot/Omega) / gamma

Note the crossing: the PITCHING moment comes from the ROLL rate and the
rolling moment from the pitch rate. That is precession, and it is easy to wire
backwards because the two equations look identical.

Everything about the rotor except the Lock number and the lift curve slope has
cancelled -- radius, chord, density and tip speed are all gone. A rotor with
gamma = 8 reacts twice as stiffly as one with gamma = 16 at the same rate.

Trim (p. 214). The rotor is in balance when

    C_M/sigma_aero + C_M/sigma_gyro = 0
    C_R/sigma_aero + C_R/sigma_gyro = 0

so in steady flight, where both rates vanish, the aerodynamic moments must
vanish on their own and the cyclic pitch is what makes them do so. These two
equations are the residuals of the A_1, B_1 trim. Prescribed rates matter for
manoeuvres and for a rotor in a wind tunnel with a moving support; they are
zero by default here.

    a, gamma, Theta_dot_Om, Phi_dot_Om --> CM_sigma_gyro, CR_sigma_gyro
"""

import numpy as np
import openmdao.api as om


class GyroMomentComp(om.ExplicitComponent):
    """Gyroscopic hub moment coefficients, p. 214."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('gamma', shape=(nn,), desc='Lock number')
        self.add_input('a', val=6.0, units='1/rad', desc='lift curve slope')
        self.add_input('Theta_dot_Om', shape=(nn,), val=0.0,
                       desc='prescribed pitch rate over Omega')
        self.add_input('Phi_dot_Om', shape=(nn,), val=0.0,
                       desc='prescribed roll rate over Omega')

        self.add_output('CM_sigma_gyro', shape=(nn,),
                        desc='gyroscopic pitching moment, from the roll rate')
        self.add_output('CR_sigma_gyro', shape=(nn,),
                        desc='gyroscopic rolling moment, from the pitch rate')

        pairs = (('CM_sigma_gyro', 'Phi_dot_Om'),
                 ('CR_sigma_gyro', 'Theta_dot_Om'))
        for out, rate in pairs:
            self.declare_partials(out, rate, rows=ar, cols=ar)
            self.declare_partials(out, 'gamma', rows=ar, cols=ar)
            self.declare_partials(out, 'a', rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        scale = 2.0 * inputs['a'][0] / inputs['gamma']

        outputs['CM_sigma_gyro'] = scale * inputs['Phi_dot_Om']
        outputs['CR_sigma_gyro'] = scale * inputs['Theta_dot_Om']

    def compute_partials(self, inputs, partials):
        a, gamma = inputs['a'][0], inputs['gamma']
        scale = 2.0 * a / gamma

        for out, rate in (('CM_sigma_gyro', 'Phi_dot_Om'),
                          ('CR_sigma_gyro', 'Theta_dot_Om')):
            partials[out, rate] = scale
            partials[out, 'gamma'] = -scale * inputs[rate] / gamma
            partials[out, 'a'] = scale * inputs[rate] / a
