"""
Figures 5.14 and 5.15, pp. 365-366: maximum acceleration and deceleration of
the example helicopter, sea level, 20,000 lb.

Writes figures_5_14_5_15_model.png: G3 (rotor tilted forward at the available
main rotor power, 3,600 hp) and G4 (autorotation at 120 % rotor speed) against
the digitized figures.
"""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

from prouty.special_performance import MaxAccelerationGroup, RotorForceLimitGroup
from prouty.special_performance.book_figures import FIG_5_14, FIG_5_15

HERE = pathlib.Path(__file__).parent


def run(system, V):
    p = om.Problem(reports=False)
    p.model.add_subsystem('g', system, promotes=['*'])
    p.setup()
    p.set_val('V', V, units='kn')
    p.set_val('sigma', 0.0849)
    p.set_val('cd_bar', 0.01 * np.ones(len(V)))
    p.set_val('R', 30.0)
    p.run_model()
    return p


Va = np.arange(20.0, 161.0, 10.0)
Vd = np.arange(40.0, 161.0, 10.0)
acc = run(MaxAccelerationGroup(num_nodes=len(Va)), Va).get_val('acc_max')
dec = run(RotorForceLimitGroup(num_nodes=len(Vd), mode='decel'), Vd).get_val('decel')

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Va, acc, 'k', label='G3 acceleration')
ax.plot(list(FIG_5_14), list(FIG_5_14.values()), 'ko', mfc='none', label='Figure 5.14')
ax.plot(Vd, dec, 'b', label='G4 deceleration (above 37 kt)')
ax.plot(list(FIG_5_15), list(FIG_5_15.values()), 'bs', mfc='none', label='Figure 5.15')
ax.set(xlabel='speed, kt', ylabel='capability, ft/s^2', ylim=(0, 35))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / 'figures_5_14_5_15_model.png', dpi=130)
