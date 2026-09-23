"""
LoadingsComp -- the five spanwise loadings of the numerical method.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 209 (thrust), p. 211 (moments), p. 212 (torque and H-force).

    d(C_T/sigma)/d(r/R) = (U_B^2/2) c_N
    d(C_M/sigma)/d(r/R) = -(U_B^2/2)(r/R) cos(psi) c_N
    d(C_R/sigma)/d(r/R) = -(U_B^2/2)(r/R) sin(psi) c_N
    d(C_Q/sigma)/d(r/R) = (U_B^2/2)(r/R) c_c
    d(C_H/sigma)/d(r/R) = (U_B^2/2)[ c_c sin(psi) + c_f (mu/U_T) cos^2(psi) ]

Each is emitted already multiplied by U_B^2 through cN_UB2 and the two parts
of the chordwise force, so nothing here divides by U_B.

Split by integration domain. The torque and H-force loadings come out in two
pieces, because p. 212 integrates them over two different spans: the induced
part between the root cutout and the tip loss station, the pressure drag and
skin friction over the whole blade. The consumer applies w_r_lift to the
_lift outputs and w_r_full to the _full ones. Thrust and the two moments are
lift-domain only.

The spanwise friction term, and a departure from the text. The second term of
the H-force loading is the skin friction acting along the span, resolved onto
the flight direction by cos(psi). Prouty writes it as
(U_B^2/2) c_f mu cos^2(psi) / U_T, which carries 1/U_T and is therefore
singular on the reverse flow boundary -- and U_B^2 does not vanish there to
save it, since U_P survives. Rederiving it from the skin friction of p. 211,

    S.F. = (rho/2) c dr (U_T^2 + U_R^2) c_f
    spanwise component  = S.F. (U_R / U_TR)
    along the flight path = that times cos(psi)

gives (1/2) c_f U_TR mu cos^2(psi), which is regular everywhere and agrees
with Prouty's form wherever U_P and U_R are small against U_T, since both
U_B^2/U_T and U_TR tend to |U_T| there.

The default is the rederived form, and the measurement is why. On the example
helicopter at mu = 0.3, integrating the two versions over the disc gives

    book, U_B^2/U_T   C_H/sigma = -0.0452
    rederived, U_TR   C_H/sigma = +0.0010
    closed form       C_H/sigma = +0.0016

A factor of 43 and a change of sign. The singularity is not a local blemish:
it dominates the H-force integral outright, because a grid that samples the
reverse flow boundary lands arbitrarily close to U_T = 0 and the term has no
U_B^2 left to suppress it there. Nothing else in the model moves -- thrust,
torque and both moments are identical between the two -- so the whole
discrepancy is this one term. The book form is kept under spanwise_form for
comparison, not for use.

    cN_UB2, cc0_UB2, ccind_UB2, r_R, psi, mu, UT_bar, UTR_bar, UB_bar, c_f
        --> dCT_dr, dCM_dr, dCR_dr,
            dCQ_dr_lift, dCQ_dr_full, dCH_dr_lift, dCH_dr_full
"""

import numpy as np
import openmdao.api as om

LIFT_OUTPUTS = ('dCT_dr', 'dCM_dr', 'dCR_dr', 'dCQ_dr_lift', 'dCH_dr_lift')
FULL_OUTPUTS = ('dCQ_dr_full', 'dCH_dr_full')


class LoadingsComp(om.ExplicitComponent):
    """Spanwise loadings, split by integration domain, p. 209-212."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('spanwise_form', values=('book', 'regular'),
                             default='regular',
                             desc="'regular' uses U_TR and is the default; "
                                  "'book' uses U_B^2/U_T as printed on p. 212 "
                                  'and is singular on the reverse flow '
                                  'boundary -- kept for comparison only')
        self.options.declare('epsilon', types=float, default=1.0e-8)

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)
        rows = np.arange(nn * n_psi * n_r)

        for name in ('cN_UB2', 'cc0_UB2', 'ccind_UB2', 'UT_bar', 'UTR_bar',
                     'UB_bar'):
            self.add_input(name, shape=field)
        self.add_input('r_R', shape=(n_r,))
        self.add_input('psi', shape=(n_psi,), units='rad')
        self.add_input('mu', shape=(nn,))
        self.add_input('c_f', val=0.006)

        for name in LIFT_OUTPUTS + FULL_OUTPUTS:
            self.add_output(name, shape=field)

        radial = np.tile(np.arange(n_r), nn * n_psi)
        azimuth = np.repeat(np.tile(np.arange(n_psi), nn), n_r)
        node = np.repeat(np.arange(nn), n_psi * n_r)
        zeros = np.zeros(rows.size, dtype=int)

        for out in ('dCT_dr',):
            self.declare_partials(out, 'cN_UB2', rows=rows, cols=rows, val=0.5)
        for out in ('dCM_dr', 'dCR_dr'):
            self.declare_partials(out, 'cN_UB2', rows=rows, cols=rows)
            self.declare_partials(out, 'r_R', rows=rows, cols=radial)
            self.declare_partials(out, 'psi', rows=rows, cols=azimuth)
        self.declare_partials('dCQ_dr_lift', 'ccind_UB2', rows=rows, cols=rows)
        self.declare_partials('dCQ_dr_lift', 'r_R', rows=rows, cols=radial)
        self.declare_partials('dCQ_dr_full', 'cc0_UB2', rows=rows, cols=rows)
        self.declare_partials('dCQ_dr_full', 'r_R', rows=rows, cols=radial)
        self.declare_partials('dCH_dr_lift', 'ccind_UB2', rows=rows, cols=rows)
        self.declare_partials('dCH_dr_lift', 'psi', rows=rows, cols=azimuth)
        self.declare_partials('dCH_dr_full', 'cc0_UB2', rows=rows, cols=rows)
        self.declare_partials('dCH_dr_full', 'psi', rows=rows, cols=azimuth)
        self.declare_partials('dCH_dr_full', 'mu', rows=rows, cols=node)
        self.declare_partials('dCH_dr_full', 'c_f', rows=rows, cols=zeros)
        source = 'UT_bar' if self.options['spanwise_form'] == 'book' else 'UTR_bar'
        self.declare_partials('dCH_dr_full', source, rows=rows, cols=rows)
        if self.options['spanwise_form'] == 'book':
            self.declare_partials('dCH_dr_full', 'UB_bar', rows=rows, cols=rows)

    def _grid(self, inputs):
        r = inputs['r_R'][np.newaxis, np.newaxis, :]
        psi = inputs['psi'][np.newaxis, :, np.newaxis]
        mu = inputs['mu'][:, np.newaxis, np.newaxis]
        return r, np.sin(psi), np.cos(psi), mu

    def _spanwise(self, inputs):
        """Skin friction acting along the span, resolved onto the flight path."""
        _, _, cos_psi, mu = self._grid(inputs)
        c_f = inputs['c_f'][0]

        if self.options['spanwise_form'] == 'book':
            UT = inputs['UT_bar']
            safe = np.where(np.real(UT) >= 0.0, UT + self.options['epsilon'],
                            UT - self.options['epsilon'])
            velocity = inputs['UB_bar'] ** 2 / safe
        else:
            velocity = inputs['UTR_bar']
        return 0.5 * c_f * mu * cos_psi ** 2 * velocity, velocity

    def compute(self, inputs, outputs):
        r, sin_psi, cos_psi, _ = self._grid(inputs)
        cN, cc0, ccind = (inputs['cN_UB2'], inputs['cc0_UB2'],
                          inputs['ccind_UB2'])
        spanwise, _ = self._spanwise(inputs)

        outputs['dCT_dr'] = 0.5 * cN
        outputs['dCM_dr'] = -0.5 * r * cos_psi * cN
        outputs['dCR_dr'] = -0.5 * r * sin_psi * cN
        outputs['dCQ_dr_lift'] = 0.5 * r * ccind
        outputs['dCQ_dr_full'] = 0.5 * r * cc0
        outputs['dCH_dr_lift'] = 0.5 * sin_psi * ccind
        outputs['dCH_dr_full'] = 0.5 * sin_psi * cc0 + spanwise

    def compute_partials(self, inputs, partials):
        shape = inputs['cN_UB2'].shape
        r, sin_psi, cos_psi, mu = self._grid(inputs)
        cN, cc0, ccind = (inputs['cN_UB2'], inputs['cc0_UB2'],
                          inputs['ccind_UB2'])
        c_f = inputs['c_f'][0]
        full = lambda x: np.broadcast_to(x, shape).ravel()

        partials['dCM_dr', 'cN_UB2'] = full(-0.5 * r * cos_psi)
        partials['dCM_dr', 'r_R'] = full(-0.5 * cos_psi * cN)
        partials['dCM_dr', 'psi'] = full(0.5 * r * sin_psi * cN)
        partials['dCR_dr', 'cN_UB2'] = full(-0.5 * r * sin_psi)
        partials['dCR_dr', 'r_R'] = full(-0.5 * sin_psi * cN)
        partials['dCR_dr', 'psi'] = full(-0.5 * r * cos_psi * cN)

        partials['dCQ_dr_lift', 'ccind_UB2'] = full(0.5 * r)
        partials['dCQ_dr_lift', 'r_R'] = full(0.5 * ccind)
        partials['dCQ_dr_full', 'cc0_UB2'] = full(0.5 * r)
        partials['dCQ_dr_full', 'r_R'] = full(0.5 * cc0)

        partials['dCH_dr_lift', 'ccind_UB2'] = full(0.5 * sin_psi)
        partials['dCH_dr_lift', 'psi'] = full(0.5 * cos_psi * ccind)

        spanwise, velocity = self._spanwise(inputs)
        base = 0.5 * c_f * mu * cos_psi ** 2
        partials['dCH_dr_full', 'cc0_UB2'] = full(0.5 * sin_psi)
        partials['dCH_dr_full', 'psi'] = full(
            0.5 * cos_psi * cc0 - c_f * mu * cos_psi * sin_psi * velocity)
        partials['dCH_dr_full', 'mu'] = full(spanwise / mu)
        partials['dCH_dr_full', 'c_f'] = full(spanwise / c_f)

        if self.options['spanwise_form'] == 'book':
            UT = inputs['UT_bar']
            safe = np.where(np.real(UT) >= 0.0, UT + self.options['epsilon'],
                            UT - self.options['epsilon'])
            partials['dCH_dr_full', 'UT_bar'] = full(-spanwise / safe)
            partials['dCH_dr_full', 'UB_bar'] = full(
                2.0 * spanwise / inputs['UB_bar'])
        else:
            partials['dCH_dr_full', 'UTR_bar'] = full(base)
