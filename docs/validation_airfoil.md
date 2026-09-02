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

---

# Validation notes — `prouty.hover`

Chapter 1, combined momentum and blade element theory with empirical
corrections, steps 1 to 21, p. 69-72. Reproduced with
`validation/validate_complete_method.py` and
`validation/validate_hover_rotor_group.py`.

## Sample calculation vs. Fig. 1.45 (p. 76)

Example helicopter main rotor, p. 669: R = 30 ft, c = 2 ft, b = 4,
V_tip = 650 ft/s, cutout 0.15 R, theta_1 = -10 deg, theta_0 = 17.5 deg,
NACA 0012, sea level standard day, 10 elements.

| Quantity | Model | Book | Error |
|---|---|---|---|
| `CT_no_tip_loss` | 0.00739 | 0.00741 | −0.33 % |
| `B` | 0.9794 | 0.98 | −0.06 % |
| `CT` | 0.00702 | 0.00719 | −2.42 % |
| `CQ0` | 1.1097e-4 | 1.11e-4 | −0.03 % |
| `CQi` | 4.3925e-4 | 4.59e-4 | −4.30 % |
| `swirl_ratio` | 0.0169 | 0.017 | −0.38 % |
| `DL` | 7.05 | 7.2 | −2.14 % |
| `CT_sigma` | 0.0827 | 0.0846 | −2.30 % |
| `power_factor` | 1.0458 | 1.05 | −0.40 % |
| `CQ_sigma` | 0.00687 | 0.0070 | −1.85 % |
| `T` | 19,923 lb | 20,400 lb | −2.34 % |
| `power_hp` | 1,957 | 1,990 | −1.65 % |

## The step 11 discrepancy

Every quantity that does not pass through the tip loss truncation matches to
better than 0.4 %; `CQ0`, which integrates on the untruncated weights, is exact
to 0.03 %. The single source of error is `CT` at step 11.

The book's own numbers are not self-consistent there. Integrating our loading
on a 200 element grid gives C_T = 0.00684 at B = 0.97, 0.00702 at 0.98 and
0.00720 at 0.99. The printed pair 0.00741 → 0.00719 therefore corresponds to
B ≈ 0.99, not to the 0.98 printed beside it. The printed ratio
0.00719/0.00741 = 0.9703 is instead close to B itself, which suggests the
thrust was scaled by a factor near B rather than reintegrated to B. Step 11
says to integrate from x0 to B, and that is what is implemented.

Both forms of step 11 were checked and agree to machine zero: integrating
directly from x0 to B, and subtracting the integral from B to 1 from the
untruncated result, differ by 9e-19.

## Tip loss model

Two models are provided. `general`, B = 1 − sqrt(2 C_T)/b, gives 0.970 and
matches p. 35 and p. 200. `effective_radius`, the two branch form of the note
to step 10, gives 0.979 and matches Fig. 1.45. The second is the default
**because** the airfoil data come from `prouty.airfoil`, i.e. from reference
1.1, which is the condition the note attaches to those equations. The 0.06
constant is calibrated so that b = 2 returns exactly 0.97 R, the assumption the
note says the data were synthesized under; this is checked in the test suite.

The branch point at C_T = 0.006 is smoothed by a cubic blend placed entirely
above the break. Centring the blend on the break produced a non-monotone B,
because the two branches actually cross at C_T = 0.005991.

## Digitised charts

Both were read by eye and both are confirmed by the book's own sample values.

| Chart | Page | Anchor | Model |
|---|---|---|---|
| Fig. 1.29, swirl ratio | 52 | 0.017 at C_T = 0.00719 | 0.0173 |
| Fig. 1.34, power factor | 58 | 1.05 at 0.61 | 1.0492 |

Reading precision is about ±0.002 and ±0.005 respectively, but the test points
on Fig. 1.34 scatter roughly ±0.03 about the fitted curve, so that scatter, not
the digitisation, sets the real uncertainty on any computed power.

Fig. 1.29 carries three curves; the spread between them is at most 1.4 % of
C_Q, so the choice of curve is not critical. `approximate` is the default.

## Known limits

- The empirical charts cover roughly theta_0 in [14, 21] deg for this rotor.
  Outside it the splines extrapolate and warn.
- Trim mode requires theta_0 > −theta_1 for linear twist about the centre; the
  closed form inflow of step 5 has no real root below that and Newton stalls.
- `check_totals` with `method='fd'` does not reconverge the Newton in trim
  mode and reports the derivative at frozen collective. Analytic totals are
  correct and are verified against retrimmed central differences.
- Ideal twist gives exactly uniform inflow only at constant lift curve slope.
  Coupled to the Mach dependent airfoil model the spread is about 6 %.
- Step 13 asks for the profile torque integral from 0, not from the cutout.
  The grid starts at the cutout and no hub drag model exists; the r^3 weighting
  makes the missing span worth 0.05 % of C_Q0.
