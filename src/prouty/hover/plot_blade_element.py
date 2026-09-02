"""
plot_blade_element -- Figure 1.45 style summary of a hover blade element run.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, Figure 1.45, "Results of Sample Calculation", p. 76.

The book overlays every radial distribution of the calculation on one set of
axes, scaling each so that they share a single ordinate: angles in degrees as
they are, the Mach number times 10, the thrust loading times 1,000, the profile
torque loading times 10,000. That scaling is reproduced here.

This is post-processing, not a component. A figure is a side effect with no
derivatives, so putting it in the model would place it in the dependency graph
and redraw it at every optimiser iteration.

Curves are drawn only if the model actually provides them, so the same call
works on a partial chain (angles and Mach alone) and on the complete method.

    plot_figure_145(problem, 'run.png', title='...')
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# name in the model, scale factor, legend label
CURVES = (
    ('phi', 1.0, r'$\tan^{-1}(v_1/\Omega r)$, deg'),
    ('theta', 1.0, r'$\theta$, deg'),
    ('alpha', 1.0, r'$\alpha$, deg'),
    ('M', 10.0, r'$10\,M$'),
    ('cl', 10.0, r'$10\,c_l$'),
    ('cd', 100.0, r'$100\,c_d$'),
    ('dCT_dr', 1000.0, r'$1{,}000\;dC_T/d(r/R)$'),
    ('dCQ0_dr', 10000.0, r'$10{,}000\;dC_{Q_0}/d(r/R)$'),
    ('dCQi_dr', 10000.0, r'$10{,}000\;dC_{Q_i}/d(r/R)$'),
)


def _get(problem, name):
    """Return an output if the model has it, else None."""
    try:
        return np.asarray(problem.get_val(name)).ravel()
    except (KeyError, RuntimeError):
        return None


def plot_figure_145(problem, path, title=None, annotations=None, B=None):
    """Draw the radial distributions of a hover run, Figure 1.45 style."""
    r_R = _get(problem, 'r_R')
    if r_R is None:
        raise RuntimeError('the model has no r_R output to plot against')

    fig, ax = plt.subplots(figsize=(8.0, 8.0))

    for name, scale, label in CURVES:
        y = _get(problem, name)
        if y is None or y.shape != r_R.shape:
            continue
        ax.plot(r_R, scale * y, marker='o', markersize=3, linewidth=1.2,
                label=label)

    if B is not None:
        ax.axvline(float(B), color='0.4', linestyle='--', linewidth=1.0)
        ax.annotate('B', xy=(float(B), ax.get_ylim()[1]), xytext=(-10, -14),
                    textcoords='offset points', color='0.3')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel('Radius Station, $r/R$')
    ax.set_ylabel('Scaled quantities')
    ax.set_title(title or 'Blade element distributions in hover')
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    if annotations:
        text = '\n'.join(f'{k} = {v}' for k, v in annotations.items())
        ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=9,
                va='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def case_annotations(problem):
    """Small dictionary of the scalar case parameters, for the figure corner."""
    wanted = (('sigma', '{:.4f}'), ('theta_0', '{:.1f} deg'),
              ('theta_1', '{:.1f} deg'), ('b', '{:.0f}'),
              ('V_tip', '{:.0f} ft/s'), ('x0', '{:.2f} R'))
    out = {}
    for name, fmt in wanted:
        val = _get(problem, name)
        if val is not None:
            out[name] = fmt.format(float(val[0]))
    return out
