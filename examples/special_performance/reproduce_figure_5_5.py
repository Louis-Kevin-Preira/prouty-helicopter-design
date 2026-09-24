"""
Figure 5.5, p. 351: rate of descent in autorotation of the example helicopter.

Writes figure_5_5_model.png next to this script: R/D(V) of G2b against the
digitized figure, with the speeds for minimum rate and minimum angle.
"""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

from prouty.special_performance import AutorotationDescentGroup, BestAutorotationSpeedGroup
from prouty.special_performance.book_figures import EXAMPLE_ROTOR, FIG_5_5

HERE = pathlib.Path(__file__).parent


def run(system, **values):
    p = om.Problem(reports=False)
    p.model.add_subsystem('g', system, promotes=['*'])
    p.setup()
    for k, v in {**EXAMPLE_ROTOR, **values}.items():
        p.set_val(k, v[0], units=v[1]) if isinstance(v, tuple) else p.set_val(k, v)
    p.run_model()
    return p


V = np.arange(50.0, 161.0, 10.0)
sweep = run(AutorotationDescentGroup(num_nodes=len(V)), V=(V, 'kn'))
best = {t: run(BestAutorotationSpeedGroup(target=t)) for t in ('min_rate', 'min_angle')}

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(V, sweep.get_val('RD', units='ft/min'), 'k', label='G2b, closed-form trim')
ax.plot(list(FIG_5_5), list(FIG_5_5.values()), 'o', mfc='none', label='Figure 5.5, digitized')
for t, p in best.items():
    v, rd = p.get_val('V', units='kn')[0], p.get_val('RD_nodes', units='ft/min')[1]
    ax.plot(v, rd, 's', label=f'{t}: {v:.0f} kt, {rd:.0f} ft/min, L/D {p.get_val("LD_nodes")[1]:.2f}')
ax.set(xlabel='forward speed, kt', ylabel='rate of descent, ft/min', ylim=(0, 3000))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / 'figure_5_5_model.png', dpi=130)
