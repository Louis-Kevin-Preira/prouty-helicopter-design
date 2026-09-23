"""
RotorSideForceComp -- the rotor's force along the wind axis.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 248 (Table 3.5, case 9, step n).

    C_XR/sigma = - (C_T/sigma) sin(alpha_TPP_uncorr) - C_H/sigma

The thrust is perpendicular to the tip path plane, so tilting the disc back by
alpha_TPP puts a component of it along the flight direction; the H-force lies
in the plane and adds to it. C_XR is what is left over -- negative when the
rotor is being dragged, positive when it is propelling. It is the quantity a
wind tunnel balance measures directly, which is why case 9 reports it and the
flight cases do not.

Two angles, deliberately mixed. Prouty evaluates the sine at
alpha_TPP_UNCORRECTED while C_T/sigma is the CORRECTED value: 
-(0.106)(sin 5.3 deg) - (-0.0033) = -0.0065 on p. 248. That is not a slip. The
tunnel balance resolves forces along the tunnel axes, which are fixed by how
the model was mounted -- the geometric 5.3 deg -- whereas the thrust magnitude
belongs to the corrected aerodynamic condition the rotor actually sees. Using
the corrected angle in the sine would report a force in axes that do not exist.
The input is named alpha_TPP_uncorr so the choice cannot be made by accident.

At the worked condition the two terms nearly cancel: the thrust component is
-0.0098 and the H-force contributes +0.0033, leaving -0.0065. A small
difference of larger numbers, which is worth knowing when comparing against a
measurement -- the test gives -0.0058, an 11 % difference on C_XR that is under
2 % on either term separately.

    CT_sigma, CH_sigma, alpha_TPP_uncorr --> CXR_sigma
"""

import numpy as np
import openmdao.api as om


class RotorSideForceComp(om.ExplicitComponent):
    """Rotor force along the wind axis, p. 248."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('CT_sigma', shape=(nn,),
                       desc='corrected thrust coefficient')
        self.add_input('CH_sigma', shape=(nn,), val=0.0)
        self.add_input('alpha_TPP_uncorr', shape=(nn,), units='rad',
                       desc='geometric disc angle, NOT the wall-corrected one')

        self.add_output('CXR_sigma', shape=(nn,),
                        desc='wind axis force, negative when the rotor drags')
        self.add_output('CXR_thrust_part', shape=(nn,),
                        desc='the thrust tilt term alone, for diagnostics')

        for name in ('CT_sigma', 'alpha_TPP_uncorr'):
            self.declare_partials(['CXR_sigma', 'CXR_thrust_part'], name,
                                  rows=ar, cols=ar)
        self.declare_partials('CXR_sigma', 'CH_sigma', rows=ar, cols=ar,
                              val=-1.0)

    def compute(self, inputs, outputs):
        tilt = -inputs['CT_sigma'] * np.sin(inputs['alpha_TPP_uncorr'])

        outputs['CXR_thrust_part'] = tilt
        outputs['CXR_sigma'] = tilt - inputs['CH_sigma']

    def compute_partials(self, inputs, partials):
        alpha = inputs['alpha_TPP_uncorr']
        CT = inputs['CT_sigma']

        for out in ('CXR_sigma', 'CXR_thrust_part'):
            partials[out, 'CT_sigma'] = -np.sin(alpha)
            partials[out, 'alpha_TPP_uncorr'] = -CT * np.cos(alpha)
