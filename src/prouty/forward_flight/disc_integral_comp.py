"""
DiscIntegralComp -- radial integration then azimuth average.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 209 and p. 211.

    delta C/sigma  = int_{x_0}^{B} (dC/sigma / d(r/R)) d(r/R)
    C/sigma        = (1/N) sum_{n=1}^{N} delta C/sigma_n

Five coefficients come out: thrust, pitching and rolling moment, torque and
H-force. Torque and H-force are assembled from two integrals apiece, the
induced part over x_0 to B and the profile part over the whole blade, which is
Prouty's instruction on p. 212 rather than a modelling choice.

A sixth output exists for the stall indicator of p. 228: the profile torque
coefficient integrated over the full blade at each azimuth separately,

    delta C_Q/sigma_0 = int_0^1 (dC_Q/sigma_0 / d(r/R)) d(r/R)

whose maximum around the azimuth is "another indicator that has been used" of
how close the rotor runs to its limit. The maximum is left to the consumer,
since taking it here would either introduce a non-differentiable max or force
a smoothing choice on everyone.

Both quadratures are plain sums against the weights RotorDiscGridComp
supplies: partial-cell trapezoid in radius, equal weights in azimuth. The
azimuth average is exact for every harmonic below N, which is why nothing
fancier is needed there.

    the seven loadings, w_r_lift, w_r_full, w_psi
        --> CT_sigma, CM_sigma, CR_sigma, CQ_sigma, CH_sigma, CQ0_sigma_psi
"""

import numpy as np
import openmdao.api as om

# output name -> (lift-domain loading, full-domain loading)
COEFFICIENTS = {
    'CT_sigma': ('dCT_dr', None),
    'CM_sigma': ('dCM_dr', None),
    'CR_sigma': ('dCR_dr', None),
    'CQ_sigma': ('dCQ_dr_lift', 'dCQ_dr_full'),
    'CH_sigma': ('dCH_dr_lift', 'dCH_dr_full'),
}


class DiscIntegralComp(om.ExplicitComponent):
    """Integrate the loadings over the disc, p. 209 and 211."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)

        for lift, full in COEFFICIENTS.values():
            for name in (lift, full):
                if name is not None:
                    self.add_input(name, shape=field)
        self.add_input('w_r_lift', shape=(nn, n_r))
        self.add_input('w_r_full', shape=(n_r,))
        self.add_input('w_psi', val=1.0 / n_psi)

        for name in COEFFICIENTS:
            self.add_output(name, shape=(nn,))
        self.add_output('CQ0_sigma_psi', shape=(nn, n_psi),
                        desc='profile torque per azimuth, p. 228')

        size = nn * n_psi * n_r
        node_of_field = np.repeat(np.arange(nn), n_psi * n_r)
        node_of_weight = np.repeat(np.arange(nn), n_r)
        radial = np.tile(np.arange(n_r), nn * n_psi)

        for out, (lift, full) in COEFFICIENTS.items():
            self.declare_partials(out, lift, rows=node_of_field,
                                  cols=np.arange(size))
            self.declare_partials(out, 'w_r_lift', rows=node_of_weight,
                                  cols=np.arange(nn * n_r))
            self.declare_partials(out, 'w_psi', rows=np.arange(nn),
                                  cols=np.zeros(nn, dtype=int))
            if full is not None:
                self.declare_partials(out, full, rows=node_of_field,
                                      cols=np.arange(size))
                # summing over azimuth collapses this to one dense (nn, n_r)
                # block; declaring it per field element would repeat entries
                self.declare_partials(out, 'w_r_full', rows=node_of_weight,
                                      cols=np.tile(np.arange(n_r), nn))

        sweep = np.repeat(np.arange(nn * n_psi), n_r)
        self.declare_partials('CQ0_sigma_psi', 'dCQ_dr_full', rows=sweep,
                              cols=np.arange(size))
        self.declare_partials('CQ0_sigma_psi', 'w_r_full', rows=sweep,
                              cols=radial)

    def compute(self, inputs, outputs):
        w_lift = inputs['w_r_lift'][:, np.newaxis, :]
        w_full = inputs['w_r_full'][np.newaxis, np.newaxis, :]
        w_psi = inputs['w_psi'][0]

        for name, (lift, full) in COEFFICIENTS.items():
            total = (inputs[lift] * w_lift).sum(axis=2)
            if full is not None:
                total = total + (inputs[full] * w_full).sum(axis=2)
            outputs[name] = w_psi * total.sum(axis=1)

        outputs['CQ0_sigma_psi'] = (inputs['dCQ_dr_full'] * w_full).sum(axis=2)

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        w_lift = inputs['w_r_lift'][:, np.newaxis, :]
        w_full = inputs['w_r_full'][np.newaxis, np.newaxis, :]
        w_psi = inputs['w_psi'][0]

        for name, (lift, full) in COEFFICIENTS.items():
            partials[name, lift] = np.broadcast_to(
                w_psi * w_lift, (nn, n_psi, n_r)).ravel()
            partials[name, 'w_r_lift'] = w_psi * inputs[lift].sum(axis=1).ravel()

            total = (inputs[lift] * w_lift).sum(axis=2)
            if full is not None:
                partials[name, full] = np.broadcast_to(
                    w_psi * w_full, (nn, n_psi, n_r)).ravel()
                partials[name, 'w_r_full'] = (
                    w_psi * inputs[full].sum(axis=1)).ravel()
                total = total + (inputs[full] * w_full).sum(axis=2)
            partials[name, 'w_psi'] = total.sum(axis=1)

        partials['CQ0_sigma_psi', 'dCQ_dr_full'] = np.broadcast_to(
            w_full, (nn, n_psi, n_r)).ravel()
        partials['CQ0_sigma_psi', 'w_r_full'] = inputs['dCQ_dr_full'].ravel()
