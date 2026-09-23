"""Figures for presenting a stability and control analysis.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9, Figure 9.15 (p. 619) and Figure 9.12 (p. 609).

This is post-processing, not a component. A figure is a side effect with no
derivatives, so putting it in the model would place it in the dependency graph
and redraw it at every optimiser iteration -- the same reasoning
``plot_blade_element`` sets out for Chapter 1.

Nothing here evaluates the model on a grid. The characteristic coefficients
are affine in the two derivatives a stability map is drawn in, so three model
evaluations fix the whole plane and everything after that is arithmetic on
arrays. See ``stability_map``.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prouty.stability.polynomial_discriminant_comp import discriminant
from prouty.stability.stability_map import (
    affine_coefficients,
    constant_term_line,
    evaluate_conic,
    routh_conic,
)

#: Region codes returned by :func:`classify_grid`, in Prouty's own words.
REGIONS = ('stable', 'unstable oscillations', 'unstable divergences')

#: Figure 9.23 names the same three codes after the lateral modes they are.
LATERAL_REGIONS = ('stable', 'unstable Dutch roll', 'spiral dive')

#: Fills for the three regions: unshaded, hatched lightly, hatched heavily,
#: following Figure 9.15's convention of leaving the acceptable region clear.
REGION_COLOURS = ('#ffffff', '#cfe0f3', '#f6cfcf')


def roots_on_grid(coefficients):
    """Roots of a grid of monic polynomials, batched.

    ``coefficients`` is ``(..., n + 1)`` in **descending** powers with a
    leading 1. Uses companion-matrix eigenvalues, which ``numpy`` batches,
    where ``np.roots`` would need a Python loop over every grid point.
    """
    order = coefficients.shape[-1] - 1
    batch = coefficients.shape[:-1]
    companion = np.zeros(batch + (order, order))
    companion[..., 0, :] = -coefficients[..., 1:]
    rows, cols = np.arange(1, order), np.arange(order - 1)
    companion[..., rows, cols] = 1.0
    return np.linalg.eigvals(companion)


def classify_grid(coefficients, tolerance=1e-9):
    """Region code for each point of a grid, matching ``classify``.

    Stable where every root is in the left half plane; otherwise an unstable
    oscillation if any growing root is complex, and a divergence if they are
    all real. Prouty's right-hand boundary on Figure 9.15 is the line between
    the last two, and it falls out of this rather than being drawn separately.
    """
    roots = roots_on_grid(coefficients)
    growing = roots.real > tolerance
    oscillatory = growing & (np.abs(roots.imag) > tolerance)

    region = np.zeros(coefficients.shape[:-1], dtype=int)
    region[np.any(growing, axis=-1)] = 2
    region[np.any(oscillatory, axis=-1)] = 1
    return region


def plot_stability_map(
        evaluate, path, points=(), first_range=None, second_range=None,
        resolution=240, title=None, first_label='first derivative',
        second_label='second derivative', regions=REGIONS,
        invert_first=False):
    """A stability map for an aircraft of your own.

    ``evaluate(first, second)`` returns the characteristic polynomial in
    descending powers for a given pair of derivatives -- the same callable
    :func:`affine_coefficients` takes. It is called **four times in total**,
    three to fix the affine expansion and one to verify it.

    ``points`` is a sequence of ``(first, second, label)`` to mark: the
    natural use is one per candidate empennage size, which turns the figure
    into the justification for the size you chose.

    Three boundaries, all closed form:

    - ``E = 0``, a straight line through the origin, dashed;
    - ``R.D. = 0``, a conic, solid;
    - the oscillation/divergence line, which emerges from the region shading
      rather than being drawn -- p. 618 found it by a root search and this is
      the same search done on a grid, batched.

    :func:`plot_longitudinal_stability_map` and
    :func:`plot_lateral_stability_map` are this with Figure 9.15's and
    Figure 9.23's labels and orientation.
    """
    constant, d_first, d_second = affine_coefficients(evaluate)
    conic = routh_conic(constant, d_first, d_second)

    marks = np.array([[point[0], point[1]] for point in points]) if points \
        else np.zeros((0, 2))
    first_range = first_range or _span(marks[:, 0] if len(marks) else [0.0])
    second_range = second_range or _span(marks[:, 1] if len(marks) else [0.0])

    x = np.linspace(*first_range, resolution)
    y = np.linspace(*second_range, resolution)
    X, Y = np.meshgrid(x, y)

    coefficients = (constant + X[..., None] * d_first
                    + Y[..., None] * d_second)
    region = classify_grid(coefficients)
    routh = evaluate_conic(conic, X, Y)
    constant_term = coefficients[..., -1]

    figure, axes = plt.subplots(figsize=(7.0, 5.5))
    axes.contourf(X, Y, region, levels=[-0.5, 0.5, 1.5, 2.5],
                  colors=REGION_COLOURS)
    axes.contour(X, Y, routh, levels=[0.0], colors='k', linewidths=1.2)
    axes.contour(X, Y, constant_term, levels=[0.0], colors='k',
                 linewidths=1.2, linestyles='--')

    for centre, label in zip(marks, [point[2] for point in points]):
        axes.plot(*centre, 'o', color='k', markersize=5)
        axes.annotate(label, centre, textcoords='offset points',
                      xytext=(7, 5), fontsize=9)

    _legend(axes, regions)
    if invert_first:
        axes.invert_xaxis()
    axes.set_xlabel(first_label)
    axes.set_ylabel(second_label)
    axes.set_title(title or 'Stability map')
    axes.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_longitudinal_stability_map(evaluate, path, **kwargs):
    """Figure 9.15, p. 619: angle-of-attack against speed stability."""
    kwargs.setdefault('first_label',
                      r'$\partial M/\partial\dot z$, ft-lb/(ft/sec)')
    kwargs.setdefault('second_label',
                      r'$\partial M/\partial\dot x$, ft-lb/(ft/sec)')
    kwargs.setdefault('title', 'Longitudinal stability map')
    return plot_stability_map(evaluate, path, **kwargs)


def plot_lateral_stability_map(evaluate, path, **kwargs):
    """Figure 9.23, p. 634: dihedral effect against directional stability.

    The same three region codes, named after the lateral modes they are: an
    unstable real root is a spiral dive and an unstable pair is the Dutch
    roll. Prouty plots the dihedral effect increasingly negative to the
    right, and ``invert_first`` matches that so the two figures can be laid
    side by side.

    p. 634 is worth keeping in view when reading it: for a helicopter "the
    direct relationships between these derivatives and easily changed
    geometric parameters are not so straightforward", unlike the fin and
    dihedral of an aeroplane. The map illustrates; it does not size.
    """
    kwargs.setdefault('first_label',
                      r'dihedral effect $\partial R/\partial\dot y$, '
                      r'ft-lb/(ft/sec)')
    kwargs.setdefault('second_label',
                      r'directional stability $\partial N/\partial\dot y$, '
                      r'ft-lb/(ft/sec)')
    kwargs.setdefault('title', 'Lateral-directional stability map')
    kwargs.setdefault('regions', LATERAL_REGIONS)
    kwargs.setdefault('invert_first', True)
    return plot_stability_map(evaluate, path, **kwargs)


def plot_step_response(times, response, path, single_dof=None, title=None,
                       label='pitch rate, free to translate',
                       ylabel=r'$q/B_1$, deg/sec per deg'):
    """Figure 9.12: the time history following a one-degree control step.

    ``single_dof``, when given, is the second curve Figure 9.12 carries -- the
    aircraft on trunnions, free to pitch only -- as ``(steady_rate,
    time_constant)``. p. 608: "the two time histories are essentially identical
    during the first quarter cycle of the oscillation. After that point, the
    effects of horizontal translation become dominant."
    """
    times, response = np.asarray(times), np.asarray(response)
    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    axes.plot(times, response, color='k', linewidth=1.6, label=label)

    if single_dof is not None:
        steady, tau = single_dof
        axes.plot(times, steady * (1.0 - np.exp(-times / tau)), color='k',
                  linewidth=1.2, linestyle='--',
                  label='pitch only, on trunnions')

    axes.axhline(0.0, color='k', linewidth=0.8)
    axes.set_xlabel('time, seconds')
    axes.set_ylabel(ylabel)
    axes.set_title(title or 'Response to a longitudinal control step')
    axes.grid(alpha=0.25, linewidth=0.5)
    axes.legend(fontsize=9, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_response_box(steady_rate, time_constant, path, axis='longitudinal',
                      role='any', label=None, title=None, margin=0.35):
    """Figure 9.14, p. 613: where an aircraft sits against the desirable box.

    The plane p. 612 arrives at -- steady rate per inch of stick against the
    time constant, which is the inverse of damping over inertia. Both come
    straight out of ``MilResponseRequirementsComp``.

    Unlike Figure 9.15 this needs no grid: the boundaries are the box itself,
    and the verdict is :func:`figure_9_14_violations`. For the yaw axis
    ``role`` picks between the utility and armed boxes, and both are drawn so
    the tighter one is visible.

    p. 613 makes the case for the format: "flight test data in the form of
    time histories following step control inputs can yield the information
    required to judge the flying qualities directly" -- no derivative
    estimation at all.
    """
    from prouty.stability.mil_response_requirements_comp import (
        FIGURE_9_14, figure_9_14_violations)

    boxes = FIGURE_9_14[axis]
    figure, axes = plt.subplots(figsize=(6.5, 4.5))

    for name, (rate_min, rate_max, time_max) in boxes.items():
        axes.add_patch(plt.Rectangle(
            (rate_min, 0.0), rate_max - rate_min, time_max,
            facecolor='#e8f0e4', edgecolor='k', linewidth=1.2))
        axes.annotate('desirable' if name == 'any' else f'desirable, {name}',
                      ((rate_min + rate_max) / 2.0, time_max / 2.0),
                      ha='center', fontsize=9)

    rates = [value for box in boxes.values() for value in box[:2]]
    times = [box[2] for box in boxes.values()] + [time_constant]
    axes.set_xlim(0.0, (1.0 + margin) * max(rates + [steady_rate]))
    axes.set_ylim(0.0, (1.0 + margin) * max(times))

    axes.plot(steady_rate, time_constant, 'o', color='k', markersize=7)
    axes.annotate(label or 'aircraft', (steady_rate, time_constant),
                  textcoords='offset points', xytext=(9, 6), fontsize=9)

    violations = figure_9_14_violations(steady_rate, time_constant, axis, role)
    verdict = ', '.join(violations) if violations else 'inside the box'
    axes.text(0.02, 0.97, verdict, transform=axes.transAxes, fontsize=9,
              va='top', style='italic')

    axes.set_xlabel('steady rate, deg/sec per inch')
    axes.set_ylabel('time constant, seconds')
    axes.set_title(title or f'Hover manoeuvrability, {axis} axis')
    axes.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_load_factor_margin(rotor, airframe, path, limit=None, label=None,
                            title=None, span=(1.0, 2.0)):
    """Figure 9.18, p. 623: angle-of-attack stability against load factor.

    One straight line, ``n (rotor) + (airframe)``, with the zero crossing
    marked. Everything above the axis is the region p. 622 describes, where
    "the rotor would overpower the stabilizer and would pitch the helicopter
    nose-up unless prevented by an alert pilot".

    ``limit`` marks the load factor the aircraft is designed to hold, which is
    the comparison that matters: the crossing has to lie beyond it.
    """
    factors = np.linspace(*span, 200)
    derivative = rotor * factors + airframe

    figure, axes = plt.subplots(figsize=(6.5, 4.5))
    axes.axhspan(0.0, max(derivative.max(), 1.0) * 1.15, facecolor='#f6cfcf',
                 zorder=0)
    axes.plot(factors, derivative, color='k', linewidth=1.6,
              label=label or r'$\partial M/\partial\dot z$')
    axes.axhline(0.0, color='k', linewidth=0.8)

    if rotor > 0.0:
        neutral = -airframe / rotor
        if span[0] <= neutral <= span[1]:
            axes.plot(neutral, 0.0, 'o', color='k', markersize=6)
            axes.annotate(f'neutral at {neutral:.2f} g', (neutral, 0.0),
                          textcoords='offset points', xytext=(-95, 12),
                          fontsize=9)
    if limit is not None:
        axes.axvline(limit, color='k', linewidth=1.0, linestyle=':')
        axes.annotate(f'limit {limit:.2f} g', (limit, derivative.min()),
                      textcoords='offset points', xytext=(6, 12), fontsize=9)

    axes.set_xlim(*span)
    axes.set_xlabel('load factor, g')
    axes.set_ylabel(r'$\partial M/\partial\dot z$, ft-lb/(ft/sec)')
    axes.set_title(title or 'Angle-of-attack stability against load factor')
    axes.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _span(values, margin=0.6):
    """A plotting range around some marked points."""
    values = np.asarray(values, dtype=float)
    low, high = float(np.min(values)), float(np.max(values))
    if high - low < 1e-9:
        low, high = low - 1.0, high + 1.0
    pad = margin * (high - low)
    return low - pad, high + pad


def _legend(axes, regions=REGIONS):
    from matplotlib.patches import Patch

    axes.legend(handles=[Patch(facecolor=colour, edgecolor='0.4', label=name)
                         for colour, name in zip(REGION_COLOURS, regions)],
                fontsize=9, loc='best', frameon=False)
