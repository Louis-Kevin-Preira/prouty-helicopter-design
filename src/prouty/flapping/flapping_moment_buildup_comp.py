"""Build-up of the c.g. moment per radian of flapping.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", p. 479.
"""

import numpy as np
import openmdao.api as om


class FlappingMomentBuildupComp(om.ExplicitComponent):
    """Split the flapping moment derivative into hub couple and rotor force.

    p. 479 tabulates the moment about the centre of gravity per radian of
    flapping for the example helicopter, in two parts::

        dM_CG/da_1s = dM_M/da_1s + (dH/da_1s) h_M

    The ``-T l_M`` term of p. 476 does not appear: it does not depend on the
    flapping. The in-plane force derivative comes from p. 479::

        dH/da_1s = [d(C_H/sigma)/da_1s] sigma rho pi R^4 Omega^2

    Only the second output varies with flight condition; rotor stiffness is
    a property of the blade and the rotor speed.

    The printed table
    -----------------
    ==================== ============ ============= ============
    Flight condition     Hub couple   Rotor force   Total
    ==================== ============ ============= ============
    Hover                200,940      73,500        274,440
    115 knots            200,940      123,000       323,940
    160 knots            200,940      108,000       308,940
    ==================== ============ ============= ============

    The rotor force contribution rises from hover to 115 knots and then falls
    again. Two effects compete inside ``lambda' = mu alpha_TPP -
    v_1/(Omega R)``: the induced velocity collapses with speed, which raises
    ``lambda'``, while the tip path plane pitches further nose down, which
    lowers it. The first wins up to roughly 115 knots, the second beyond.

    Only the hover entry can be checked without a full trim, since
    ``lambda' = -sqrt(C_T/2)`` there. It comes out at 69,400 against the
    73,500 printed, 5.6 % low, most likely because the table's ``theta_75``
    comes from the Chapter 1 hover analysis rather than from the idealised
    momentum value.

    Notes
    -----
    The hub couple contribution is ``dMM_da1s`` itself, so it is not
    duplicated as an output. The roll axis is the symmetric analogue, using
    ``d(C_Y/sigma)/db_1s``, which p. 479 states is the same expression.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('dMM_da1s', val=np.zeros(nn), units='lbf*ft/rad',
                       desc='Rotor stiffness.')
        self.add_input('dCHsigma_da1s', val=np.zeros(nn),
                       desc='d(C_H/sigma)/da_1s.')
        self.add_input('dCYsigma_db1s', val=np.zeros(nn),
                       desc='d(C_Y/sigma)/db_1s.')
        self.add_input('rho', val=np.full(nn, 0.002378), units='slug/ft**3',
                       desc='Air density.')
        self.add_input('Omega', val=np.ones(nn), units='rad/s',
                       desc='Rotor angular speed.')
        self.add_input('R', val=1.0, units='ft', desc='Rotor radius.')
        self.add_input('sigma', val=0.08, desc='Rotor solidity.')
        self.add_input('h_M', val=0.0, units='ft',
                       desc='Hub height above the centre of gravity.')

        self.add_output('dMCG_da1s_force', val=np.zeros(nn),
                        units='lbf*ft/rad',
                        desc='Rotor force contribution, pitch.')
        self.add_output('dLCG_db1s_force', val=np.zeros(nn),
                        units='lbf*ft/rad',
                        desc='Rotor force contribution, roll.')
        self.add_output('dMCG_da1s', val=np.zeros(nn), units='lbf*ft/rad',
                        desc='Total pitching moment per radian of flapping.')
        self.add_output('dLCG_db1s', val=np.zeros(nn), units='lbf*ft/rad',
                        desc='Total rolling moment per radian of flapping.')

        ar = np.arange(nn)
        pitch = ['dCHsigma_da1s', 'rho', 'Omega']
        roll = ['dCYsigma_db1s', 'rho', 'Omega']
        self.declare_partials(['dMCG_da1s_force', 'dMCG_da1s'], pitch,
                              rows=ar, cols=ar)
        self.declare_partials(['dLCG_db1s_force', 'dLCG_db1s'], roll,
                              rows=ar, cols=ar)
        self.declare_partials(['dMCG_da1s_force', 'dLCG_db1s_force',
                               'dMCG_da1s', 'dLCG_db1s'],
                              ['R', 'sigma', 'h_M'])
        self.declare_partials(['dMCG_da1s', 'dLCG_db1s'], 'dMM_da1s',
                              rows=ar, cols=ar, val=1.0)

    def _scale(self, inputs):
        """sigma rho A (Omega R)^2, the force that a unit coefficient gives."""
        return (inputs['sigma'] * inputs['rho'] * np.pi * inputs['R'] ** 4
                * inputs['Omega'] ** 2)

    def compute(self, inputs, outputs):
        arm = self._scale(inputs) * inputs['h_M']

        outputs['dMCG_da1s_force'] = inputs['dCHsigma_da1s'] * arm
        outputs['dLCG_db1s_force'] = inputs['dCYsigma_db1s'] * arm
        outputs['dMCG_da1s'] = outputs['dMCG_da1s_force'] + inputs['dMM_da1s']
        outputs['dLCG_db1s'] = outputs['dLCG_db1s_force'] + inputs['dMM_da1s']

    def compute_partials(self, inputs, J):
        scale, h_M = self._scale(inputs), inputs['h_M']
        arm = scale * h_M
        dCH, dCY = inputs['dCHsigma_da1s'], inputs['dCYsigma_db1s']

        for force, coef in (('dMCG_da1s_force', dCH),
                            ('dLCG_db1s_force', dCY)):
            name = 'dCHsigma_da1s' if force[1] == 'M' else 'dCYsigma_db1s'
            d = {
                name: arm,
                'rho': coef * arm / inputs['rho'],
                'Omega': 2.0 * coef * arm / inputs['Omega'],
                'R': (4.0 * coef * arm / inputs['R']).reshape(-1, 1),
                'sigma': (coef * arm / inputs['sigma']).reshape(-1, 1),
                'h_M': (coef * scale).reshape(-1, 1),
            }
            total = 'dMCG_da1s' if force[1] == 'M' else 'dLCG_db1s'
            for key, val in d.items():
                J[force, key] = val
                J[total, key] = val
