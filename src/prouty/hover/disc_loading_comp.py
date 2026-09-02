"""
DiscLoadingComp -- rotor disc loading.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 17, p. 72:

    D.L. = C_T rho (Omega R)^2        [lb/ft^2]

This follows from T = C_T rho A (Omega R)^2 divided by the disc area, so it is
the thrust per unit disc area without ever forming the thrust itself. Step 21
computes T separately from the same coefficient.

Disc loading is the physical scale of the wake: it fixes the induced velocity,
and with it the tip vortex strength. That is why step 19 enters Figure 1.34
with the product (D.L.)(C_T/sigma) rather than with a coefficient alone, and it
is why this dimensional quantity appears in the middle of an otherwise
non-dimensional calculation.

For the example helicopter, p. 76: C_T = 0.00719 at sea level with a tip speed
of 650 ft/s gives 7.2 lb/ft^2.

Note. The effective disc loading of p. 35, based on the annulus between the
cutout and B rather than on the whole disc, is a different quantity used for
hover ceiling work; step 17 wants the plain one.

    CT, rho, V_tip --> DiscLoadingComp --> DL
"""

import openmdao.api as om


class DiscLoadingComp(om.ExplicitComponent):
    """Thrust per unit disc area."""

    def setup(self):
        self.add_input('CT', val=0.00719, desc='thrust coefficient, step 11')
        self.add_input('rho', val=0.002377, units='slug/ft**3',
                       desc='air density')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed')

        self.add_output('DL', val=7.2, units='lbf/ft**2', desc='disc loading')

        self.declare_partials('DL', ['CT', 'rho', 'V_tip'])

    def compute(self, inputs, outputs):
        outputs['DL'] = inputs['CT'] * inputs['rho'] * inputs['V_tip'] ** 2

    def compute_partials(self, inputs, partials):
        ct, rho, v = inputs['CT'], inputs['rho'], inputs['V_tip']
        partials['DL', 'CT'] = rho * v ** 2
        partials['DL', 'rho'] = ct * v ** 2
        partials['DL', 'V_tip'] = 2.0 * ct * rho * v
