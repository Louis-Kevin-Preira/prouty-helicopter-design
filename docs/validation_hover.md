# Validation notes — `prouty.hover`

Chapter 1, combined momentum and blade element theory with empirical
corrections, steps 1 to 21, p. 69-72. Reproduced with
`examples/hover/reproduce_figure_1_45.py` and
`examples/hover/validate_hover_rotor_group.py`.

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
