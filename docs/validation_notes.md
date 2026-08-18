# Validation notes — `prouty.airfoil`

All figures reproduced with `validation/validate_hover_group.py` and
`validation/validate_fwd_group.py`.

## Hover group vs. Fig. 6.43 (p. 427)

### Coefficient anchors, p. 429-430

| M | `K1` model | `K1` book | `K2` model | `K2` book |
|---|---|---|---|---|
| 0.2 | 0.0233 | 0.0233 | 1.86 | 1.95 |
| 0.5 | 0.0257 | 0.0257 | 1.57 | 1.57 |
| 0.7 | 0.0500 | 0.0497 | 1.38 | 1.38 |

`K1` matches the three tabulated values. The `K2` gap at M = 0.2 is expected:
p. 429 states the straight line `2.05 - 0.95M` is fitted favouring the
high-Mach points, so it departs from the low-Mach one by design.

### Stall behaviour

| M | `alpha_L` | `cl_max` | angle at `cl_max` | `alpha_D` | `cd(0)` |
|---|---|---|---|---|---|
| 0.10 | 13.40 | 1.444 | 15.65 | 14.66 | 0.00810 |
| 0.30 | 10.20 | 1.183 | 13.45 | 9.98 | 0.00810 |
| 0.50 | 7.00 | 1.004 | 12.75 | 5.30 | 0.00810 |
| 0.70 | 3.80 | 0.707 | 9.25 | 0.62 | 0.00810 |
| 0.75 | 3.40 | 1.092 | no peak below 18 | 0.00 | 0.00826 |
| 0.80 | 3.40 | 1.109 | no peak below 18 | 0.00 | 0.01338 |
| 0.85 | 3.40 | 0.803 | no peak below 18 | 0.00 | 0.03516 |

Lift matches Fig. 6.43 up to M = 0.70: `cl_max` falls monotonically from 1.44 to
0.71 and the peak angles track the measured curves. Above the break the model
loses its stall — see the limitations section of the README.

Drag matches throughout. `cd(0)` stays at 0.0081 until drag divergence, then
rises to 0.0134 and 0.0352, and the knee shifts left with increasing Mach
exactly as in the figure.

### Cross-check of the two incompressible drag series

The asymmetric series of p. 432 (hover) and the even series of p. 433 (forward
flight) are two representations of the same fit. At positive angles they agree
to within 9.4e-4, dropping to 1.8e-4 at the extreme control point of 14.7 deg —
which validates both transcriptions at once.

| alpha | forward (even) | hover (asymmetric) |
|---|---|---|
| 2.0 | 0.008360 | 0.008536 |
| 6.0 | 0.010391 | 0.011327 |
| 14.7 | 0.058181 | 0.058356 |
| -6.0 | 0.010391 | 0.042872 |
| -14.7 | 0.058181 | 0.470794 |

At negative angles the hover series is off by a factor of eight — the odd terms
change sign. This is precisely why p. 433 mandates the even form.

## Forward flight group vs. Fig. 6.47 (p. 434)

### Segment anchors, p. 433-434, at M = 0.30

| alpha | `cl` | expected | `cd` | expected |
|---|---|---|---|---|
| 45 | 1.1500 | 1.15 | 1.0300 | 1.03 |
| 90 | 0.0000 | 0.00 | 2.0500 | 2.05 |
| 167 | -0.7000 | -0.70 | 0.1132 | — |
| 180 | 0.0000 | 0.00 | 0.0100 | 0.01 |
| 270 | 0.0000 | 0.00 | 2.0500 | 2.05 |
| 315 | -1.1500 | -1.15 | 1.0300 | 1.03 |

### Symmetry

```
max |cl(a) + cl(360-a)| = 2.8e-14
max |cd(a) - cd(360-a)| = 2.8e-14
```

This is the most discriminating test in the suite. The two `1.15 sin 2a`
segments are declared separately in the book table, and the two generated
branches pass through distinct component instances, so agreement at machine
precision simultaneously validates the angle wrapping, the antisymmetry of the
lift assembly and the parity of the even drag series.

### Ranges

```
cl in [-1.183, 1.183]
cd in [ 0.0081, 2.0500]
```

Both bracket the axes of Fig. 6.47.

### Junction overshoot at 20 and 340 deg

Lift peaks at 1.18 near 17 deg, above the `1.15 sin 2a` branch, then dips to
0.73 before rising again. Drag shows a matching bump to 0.39 near 19 deg. These
are real features of the model: Fig. 6.47 plots the generated low-angle data and
the approximate high-angle fit as two distinct dashed curves precisely because
they do not meet cleanly. The 2 deg blend softens the junction without
pretending to remove it. Widening the band would flatten the bump further at the
cost of a larger local departure from both branches.

## Derivative verification

| Component | Method | Largest residual |
|---|---|---|
| `LiftModelCoefsComp` | fd | 7.4e-6 |
| `LiftCoefComp` | fd | 1.0e-7 |
| `DragModelCoefsComp` | fd | 1.7e-3 (on a slope of -14.3) |
| `IncompDragHoverComp` | cs | 0 |
| `DragCoefHoverComp` | fd | 8.0e-9 |
| `AlphaWrapComp` | fd | 0 |
| `IncompDragFwdComp` | cs | 0 |
| `LiftCoefFwdComp` | fd | 3.8e-8 |
| `DragCoefFwdComp` | fd | 1.5e-7 |
| `AirfoilHoverGroup` (totals) | fd | 3.1e-6 |
| `AirfoilForwardFlightGroup` (totals) | fd | 6.3e-6 |

All residuals are finite-difference truncation. Components using `np.abs`,
`np.sign`, `np.where` or `np.maximum` are **not** complex-step safe; analytic
partials are supplied, but wrapping a group in `approx_totals(method='cs')`
requires switching to `'fd'`.
