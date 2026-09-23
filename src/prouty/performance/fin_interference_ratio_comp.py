"""
FinInterferenceRatioComp -- thrust interference of the fin on the tail rotor in hover.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Tail Rotor-Fin Interference in Hover", pp. 283-286, Figure 4.9
p. 286.

The fin in the tail rotor wake (tractor) or inflow (pusher) carries a force F
against the rotor thrust T, so that T_net = T_actual (1 - F/T). Figure 4.9
(Lynn et al., JAHS 1970; Morris, NASA 1971) gives F/T against the separation
x/R between rotor and fin, for fin area in the disc over disc area S/A of
0.10 to 0.25:

    tractor   F/T about 0.05 to 0.22, rising as the rotor nears the fin
    pusher    F/T about 0.07 to 0.16 at x/R = 0.2, zero at x/R = 1

Example helicopter, p. 286: pusher, S/A = 0.25, x/R = 0.3, F/T = 0.125.

Digitization (C4-14): pixel columns of each curve, smoothing spline (rms at
most 0.0008), tabulated every 0.05 in x/R; the pusher curves are closed on
zero at x/R = 1. Akima in x/R along each curve, linear in S/A between curves.
Outside the drawn range (S/A 0.10-0.25, x/R 0.2-1.0) the nearest edge is held
and a warning is raised: nothing is extrapolated.

    S_A, x_R --> F_T
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

S_A = np.array([0.10, 0.15, 0.20, 0.25])
X_R = np.round(np.arange(0.2, 1.0001, 0.05), 2)

# Figure 4.9 p. 286: F/T, rows S/A = 0.10, 0.15, 0.20, 0.25; columns X_R
_F_T = {
    'tractor': np.array([
        [0.0817, 0.0780, 0.0747, 0.0716, 0.0689, 0.0665, 0.0643, 0.0624, 0.0606,
         0.0590, 0.0576, 0.0563, 0.0552, 0.0541, 0.0530, 0.0520, 0.0510],
        [0.1283, 0.1238, 0.1196, 0.1159, 0.1125, 0.1094, 0.1065, 0.1039, 0.1015,
         0.0993, 0.0972, 0.0953, 0.0935, 0.0917, 0.0900, 0.0883, 0.0866],
        [0.1729, 0.1672, 0.1620, 0.1572, 0.1529, 0.1489, 0.1454, 0.1421, 0.1392,
         0.1364, 0.1339, 0.1316, 0.1294, 0.1273, 0.1253, 0.1233, 0.1213],
        [0.2214, 0.2155, 0.2102, 0.2052, 0.2008, 0.1967, 0.1930, 0.1896, 0.1865,
         0.1836, 0.1809, 0.1784, 0.1760, 0.1737, 0.1715, 0.1693, 0.1670]
    ]),
    'pusher': np.array([
        [0.0686, 0.0602, 0.0525, 0.0453, 0.0387, 0.0327, 0.0272, 0.0223, 0.0179,
         0.0140, 0.0106, 0.0077, 0.0053, 0.0033, 0.0018, 0.0007, 0.0000],
        [0.0936, 0.0820, 0.0714, 0.0615, 0.0524, 0.0442, 0.0367, 0.0299, 0.0239,
         0.0186, 0.0140, 0.0101, 0.0068, 0.0042, 0.0022, 0.0008, 0.0000],
        [0.1263, 0.1111, 0.0970, 0.0841, 0.0722, 0.0614, 0.0516, 0.0428, 0.0348,
         0.0278, 0.0216, 0.0162, 0.0116, 0.0077, 0.0045, 0.0019, 0.0000],
        [0.1591, 0.1405, 0.1233, 0.1076, 0.0932, 0.0801, 0.0682, 0.0574, 0.0477,
         0.0390, 0.0312, 0.0243, 0.0181, 0.0127, 0.0079, 0.0037, 0.0000]
    ]),
}
INSTALLATIONS = tuple(_F_T)
_CURVES = {name: [InterpND(method='akima', points=X_R, values=row, extrapolate=False)
                  for row in table] for name, table in _F_T.items()}


def fin_interference_ratio(S_A_val, x_R_val, installation):
    """F/T from Figure 4.9 and its derivatives d/d(S/A), d/d(x/R); edges held."""
    s = float(np.clip(np.real(S_A_val), S_A[0], S_A[-1]))
    x = float(np.clip(np.real(x_R_val), X_R[0], X_R[-1]))
    vals, slopes = zip(*(c.interpolate(np.array([x]), compute_derivative=True)
                         for c in _CURVES[installation]))
    vals, slopes = np.ravel(vals), np.ravel(slopes)

    k = min(int(np.searchsorted(S_A, s, side='right')) - 1, len(S_A) - 2)
    t = (s - S_A[k]) / (S_A[k + 1] - S_A[k])
    F_T = (1 - t) * vals[k] + t * vals[k + 1]
    s_in = S_A[0] <= np.real(S_A_val) <= S_A[-1]          # one-sided on the edges
    x_in = X_R[0] <= np.real(x_R_val) <= X_R[-1]
    dF_ds = (vals[k + 1] - vals[k]) / (S_A[k + 1] - S_A[k]) if s_in else 0.0
    dF_dx = ((1 - t) * slopes[k] + t * slopes[k + 1]) if x_in else 0.0
    return F_T, dF_ds, dF_dx


class FinInterferenceRatioComp(om.ExplicitComponent):
    """Figure 4.9: fin force over tail rotor thrust in hover."""

    def initialize(self):
        self.options.declare('installation', default='pusher', values=INSTALLATIONS)

    def setup(self):
        self.add_input('S_A', val=0.1, desc='fin area in the tail rotor disc over disc area')
        self.add_input('x_R', val=0.3, desc='rotor to fin separation over tail rotor radius')
        self.add_output('F_T', val=0.0, desc='fin force over tail rotor thrust')
        self.declare_partials('F_T', ['S_A', 'x_R'])

    def compute(self, inputs, outputs):
        s, x = np.real(inputs['S_A'][0]), np.real(inputs['x_R'][0])
        if not (S_A[0] <= s <= S_A[-1]) or not (X_R[0] <= x <= X_R[-1]):
            warnings.warn(f'S/A = {s:.3f}, x/R = {x:.3f} outside Figure 4.9 '
                          '(S/A 0.10-0.25, x/R 0.2-1.0); the nearest edge is used.', stacklevel=2)
        outputs['F_T'] = fin_interference_ratio(s, x, self.options['installation'])[0]

    def compute_partials(self, inputs, partials):
        _, dF_ds, dF_dx = fin_interference_ratio(inputs['S_A'][0], inputs['x_R'][0],
                                                 self.options['installation'])
        partials['F_T', 'S_A'] = dF_ds
        partials['F_T', 'x_R'] = dF_dx
