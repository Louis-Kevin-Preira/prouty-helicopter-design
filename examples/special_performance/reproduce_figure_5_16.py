"""
Figure 5.16, p. 369: takeoff at high gross weight (28,000 lb, sea level).

Writes figure_5_16_model.png: optimum rotation speed and minimum distance to
clear an obstacle (G5, low-speed power join), against the figure. The hover
power, installed power, IGE thrust and V_max are the Chapter 4 values of
EXAMPLE_28K (docs/validation_special_performance.md, G5).
"""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

from prouty.special_performance import OptimumTakeoffGroup
from prouty.special_performance.book_figures import (EXAMPLE_28K, EXAMPLE_POWER, EXAMPLE_ROTOR,
                                                     FIG_5_16)

HERE = pathlib.Path(__file__).parent
rotor = {k: v for k, v in EXAMPLE_ROTOR.items() if k != 'GW'}

h_list = np.array([25.0, 50.0, 100.0, 175.0, 250.0, 375.0, 500.0])
V_rot, x_min = [], []
for h in h_list:
    p = om.Problem(reports=False)
    p.model.add_subsystem('g', OptimumTakeoffGroup(), promotes=['*'])
    p.setup()
    for k, v in {**rotor, **EXAMPLE_POWER, 'GW': 28000.0, 'CT_sigma': 0.12, 'h': h,
                 'P_hover': EXAMPLE_28K['P_hover'], 'P_avail': EXAMPLE_28K['P_avail'],
                 'T_max_IGE': EXAMPLE_28K['T_max_IGE']}.items():
        p.set_val(k, v)
    p.set_val('alpha_F', np.deg2rad(-20.0) * np.ones(3))
    p.set_val('V_max', EXAMPLE_28K['V_max_kt'], units='kn')
    p.run_model()
    V_rot.append(p.get_val('V_rot', units='kn')[0])
    x_min.append(p.get_val('x_min')[0])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(h_list, x_min, 'k', label='G5 minimum distance')
ax.plot(list(FIG_5_16), [d for _, d in FIG_5_16.values()], 'ko', mfc='none', label='Figure 5.16')
for h, v in zip(h_list, V_rot):
    ax.annotate(f'{v:.0f} kt', (h, x_min[list(h_list).index(h)]), fontsize=7)
ax.set(xlabel='obstacle height, ft', ylabel='distance to clear obstacle, ft')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / 'figure_5_16_model.png', dpi=130)
