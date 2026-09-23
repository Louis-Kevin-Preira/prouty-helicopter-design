"""
RotorGeometryComp -- dimensional rotor geometry to the non-dimensional ratios
used by the combined momentum and blade element method.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 1 ("Given: rotor geometry"), p. 69.

This is the only place where lengths appear, so R and the chords can be used
directly as design variables. Tip speed is carried as V_tip rather than rpm,
following the book's own practice (650 ft/s for the example helicopter, p. 76).

    R, c_root, c_tip, r_1, r_cutout, V_tip
        --> c_root_R, c_tip_R, x_t, x0, A, Omega

Defaults are the example helicopter main rotor, p. 669: R = 30 ft, c = 2 ft,
b = 4, V_tip = 650 ft/s, cutout = 0.15 R, giving sigma = 0.085.
"""

import numpy as np
import openmdao.api as om


class RotorGeometryComp(om.ExplicitComponent):
    """Non-dimensional geometry ratios, disc area and rotational speed."""

    def setup(self):
        self.add_input('R', val=30.0, units='ft', desc='rotor radius')
        self.add_input('c_root', val=2.0, units='ft', desc='inboard chord')
        self.add_input('c_tip', val=2.0, units='ft', desc='tip chord')
        self.add_input('r_1', val=30.0, units='ft', desc='taper start radius')
        self.add_input('r_cutout', val=4.5, units='ft', desc='root cutout radius')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed, Omega R')

        self.add_output('c_root_R', val=0.0667, desc='inboard chord ratio')
        self.add_output('c_tip_R', val=0.0667, desc='tip chord ratio')
        self.add_output('x_t', val=1.0, desc='taper start station, r/R')
        self.add_output('x0', val=0.15, desc='root cutout, r/R')
        self.add_output('A', val=2827.4, units='ft**2', desc='disc area')
        self.add_output('Omega', val=21.667, units='rad/s', desc='rotor speed')

        self.declare_partials('c_root_R', ['c_root', 'R'])
        self.declare_partials('c_tip_R', ['c_tip', 'R'])
        self.declare_partials('x_t', ['r_1', 'R'])
        self.declare_partials('x0', ['r_cutout', 'R'])
        self.declare_partials('A', 'R')
        self.declare_partials('Omega', ['V_tip', 'R'])

    def compute(self, inputs, outputs):
        R = inputs['R']
        outputs['c_root_R'] = inputs['c_root'] / R
        outputs['c_tip_R'] = inputs['c_tip'] / R
        outputs['x_t'] = inputs['r_1'] / R
        outputs['x0'] = inputs['r_cutout'] / R
        outputs['A'] = np.pi * R ** 2
        outputs['Omega'] = inputs['V_tip'] / R

    def compute_partials(self, inputs, partials):
        R = inputs['R']
        for out, num in (('c_root_R', 'c_root'), ('c_tip_R', 'c_tip'),
                         ('x_t', 'r_1'), ('x0', 'r_cutout'), ('Omega', 'V_tip')):
            partials[out, num] = 1.0 / R
            partials[out, 'R'] = -inputs[num] / R ** 2
        partials['A', 'R'] = 2.0 * np.pi * R
