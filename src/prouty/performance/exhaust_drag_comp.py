"""
ExhaustDragComp -- drag or residual thrust of the engine exhaust.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Miscellaneous Drag" p. 304; procedure and example p. 308.

A turboshaft exhaust designed for hover has low exhaust velocity, so above
some speed it produces residual drag instead of residual thrust, and a canted
stack loses the rearward momentum of the engine flow (p. 304):

    D_ex = m_dot (V - V_ex cos chi)          m_dot in slug/s, chi the cant angle
    f_ex = D_ex / q                          q = rho V^2 / 2

mode
    'thrust'     net residual thrust T_net of all engines, from the engine
                 manufacturer: f_ex = -T_net / q (positive drag when T_net < 0)
    'momentum'   the equation above, for a canted stack

Example helicopter, p. 308: T_net = 2(-11) = -22 lb at 115 kt, q = 45 psf,
f_ex = 0.5 ft^2.

    rho, V, [T_res, n_eng | m_dot, V_ex, chi] --> q, f_ex
"""

import numpy as np
import openmdao.api as om


class ExhaustDragComp(om.ExplicitComponent):
    """Exhaust residual drag, p. 308."""

    def initialize(self):
        self.options.declare('mode', default='thrust', values=('thrust', 'momentum'))

    def setup(self):
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('V', val=1.0, units='ft/s', desc='reference flight speed')
        self.add_output('q', val=1.0, units='lbf/ft**2', desc='free stream dynamic pressure')
        self.add_output('f_ex', val=0.0, units='ft**2', desc='exhaust equivalent flat plate area')

        if self.options['mode'] == 'thrust':
            self.add_input('T_res', val=0.0, units='lbf', desc='residual thrust per engine')
            self.add_input('n_eng', val=1.0, desc='number of engines')
            self.declare_partials('f_ex', ['rho', 'V', 'T_res', 'n_eng'])
        else:
            self.add_input('m_dot', val=0.0, units='slug/s', desc='engine mass flow, all engines')
            self.add_input('V_ex', val=0.0, units='ft/s', desc='exhaust velocity')
            self.add_input('chi', val=0.0, units='rad', desc='exhaust cant angle from rearward')
            self.declare_partials('f_ex', ['rho', 'V', 'm_dot', 'V_ex', 'chi'])
        self.declare_partials('q', ['rho', 'V'])

    def compute(self, inputs, outputs):
        rho, V = inputs['rho'][0], inputs['V'][0]
        q = 0.5 * rho * V ** 2
        if self.options['mode'] == 'thrust':
            D_ex = -inputs['n_eng'][0] * inputs['T_res'][0]
        else:
            D_ex = inputs['m_dot'][0] * (V - inputs['V_ex'][0] * np.cos(inputs['chi'][0]))
        outputs['q'] = q
        outputs['f_ex'] = D_ex / q

    def compute_partials(self, inputs, partials):
        rho, V = inputs['rho'][0], inputs['V'][0]
        q = 0.5 * rho * V ** 2
        partials['q', 'rho'], partials['q', 'V'] = 0.5 * V ** 2, rho * V

        if self.options['mode'] == 'thrust':
            n, T = inputs['n_eng'][0], inputs['T_res'][0]
            D_ex = -n * T
            partials['f_ex', 'T_res'] = -n / q
            partials['f_ex', 'n_eng'] = -T / q
            dD_dV = 0.0
        else:
            m, Vex, chi = inputs['m_dot'][0], inputs['V_ex'][0], inputs['chi'][0]
            D_ex = m * (V - Vex * np.cos(chi))
            partials['f_ex', 'm_dot'] = (V - Vex * np.cos(chi)) / q
            partials['f_ex', 'V_ex'] = -m * np.cos(chi) / q
            partials['f_ex', 'chi'] = m * Vex * np.sin(chi) / q
            dD_dV = m

        partials['f_ex', 'rho'] = -D_ex / q * (0.5 * V ** 2) / q
        partials['f_ex', 'V'] = dD_dV / q - D_ex / q * (rho * V) / q
