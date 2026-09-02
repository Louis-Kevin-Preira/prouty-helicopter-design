"""
DiscAirfoilGroup -- the Chapter 6 airfoil model applied over the rotor disc.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, p. 426-434, driven from the numerical method of Chapter 3,
p. 214, 218 and 221.

AirfoilForwardFlightGroup is written for a flat list of num_nodes conditions,
while G2 carries fields of shape (num_nodes, num_azimuth, num_radial) -- 984
points for a single flight condition at the default grid. This group is the
adapter: it flattens, calls the airfoil model with num_nodes equal to the
total, and reshapes the coefficients back.

Nothing physical happens here. The only work is the reshaping and the unit
bookkeeping: alpha and d_alpha_stall arrive in radians from Chapter 3 and the
airfoil model wants degrees, which OpenMDAO converts across the connections
because both ends declare their units.

The airfoil model is instantiated with stall_angles='external', so it takes
sec_Lambda and d_alpha_stall and applies the sweep and dynamic overshoot
corrections of p. 218 and 221 internally. A caller that leaves them at their
neutral defaults gets the static angles, so this adapter is also the way to
run G2 with those two effects switched off -- feed sec_Lambda = 1 and
d_alpha_stall = 0 rather than removing the components.

cl comes out under the name cl_raw because it still has to pass through the
bounds of p. 221 before the force equations see it; those live in G2, not
here, since they are a property of the numerical method rather than of the
airfoil.

    alpha, M, sec_Lambda, d_alpha_stall  (field) --> cl_raw, cd  (field)
"""

import numpy as np
import openmdao.api as om

from prouty.airfoil.airfoil_forward_flight_group import AirfoilForwardFlightGroup

# name on the disc, name on the flat list, units
TO_FLAT = (('alpha', 'alpha_flat', 'rad'),
           ('M', 'M_flat', None),
           ('sec_Lambda', 'sec_Lambda_flat', None),
           ('d_alpha_stall', 'd_alpha_stall_flat', 'rad'))

FROM_FLAT = (('cl_flat', 'cl_raw', None),
             ('cd_flat', 'cd', None))


class _Reshape(om.ExplicitComponent):
    """Reshape a set of variables between the disc field and a flat list."""

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('names', types=tuple)
        self.options.declare('to_flat', types=bool, default=True)

    def setup(self):
        shape = self.options['shape']
        size = int(np.prod(shape))
        flat = (size,)
        rows = np.arange(size)

        for source, target, units in self.options['names']:
            in_shape = shape if self.options['to_flat'] else flat
            out_shape = flat if self.options['to_flat'] else shape

            self.add_input(source, shape=in_shape, units=units)
            self.add_output(target, shape=out_shape, units=units)
            self.declare_partials(target, source, rows=rows, cols=rows, val=1.0)

    def compute(self, inputs, outputs):
        out_shape = ((int(np.prod(self.options['shape'])),)
                     if self.options['to_flat'] else self.options['shape'])
        for source, target, _ in self.options['names']:
            outputs[target] = inputs[source].reshape(out_shape)


class DiscAirfoilGroup(om.Group):
    """NACA 0012 coefficients at every blade element of the disc."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('clip_K1', types=bool, default=True)

    def setup(self):
        shape = (self.options['num_nodes'], self.options['num_azimuth'],
                 self.options['num_radial'])
        size = int(np.prod(shape))

        self.add_subsystem('flatten',
                           _Reshape(shape=shape, names=TO_FLAT, to_flat=True),
                           promotes_inputs=['*'])
        self.add_subsystem('airfoil',
                           AirfoilForwardFlightGroup(
                               num_nodes=size, stall_angles='external',
                               clip_K1=self.options['clip_K1']))
        self.add_subsystem('unflatten',
                           _Reshape(shape=shape, names=FROM_FLAT,
                                    to_flat=False),
                           promotes_outputs=['*'])

        for source, target, _ in TO_FLAT:
            name = {'alpha': 'alpha_raw'}.get(source, source)
            self.connect(f'flatten.{target}', f'airfoil.{name}')
        self.connect('airfoil.cl', 'unflatten.cl_flat')
        self.connect('airfoil.cd', 'unflatten.cd_flat')
