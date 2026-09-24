# Chapter 9 — Stability and Control Analysis: validation notes

Discrepancies between the printed book and the implementation, and the reason
for each departure. Nothing here is a silent correction.

## C9-1 — Routh's discriminant for a quintic (p. 556)

The book prints, for `A s⁵ + B s⁴ + C s³ + D s² + E s + F = 0`:

    R.D.(5) = D(BC − AD)(BE − AF) − B(BF − AF)² − F(BC − AD)²

The middle term is a misprint. Building the Routh array for the quintic gives,
with `P = BC − AD` and `Q = BE − AF`, a last-row numerator of

    D·P·Q − B·Q² − F·P²

so the squared factor is `(BE − AF)`, i.e. `Q`, not `(BF − AF)`. The two
printed occurrences of `Q` in the same expression confirm the intent: `BF − AF`
is not a Routh-array quantity at all, and would factor as `F(B − A)`, which is
dimensionally inconsistent with the other terms.

`RouthDiscriminantComp(degree=5)` implements the corrected form. No result in
Chapter 9 depends on it — Prouty only ever uses the cubic (p. 602) and the
quartic (p. 618) — so this is a latent misprint rather than an error affecting
the book's numbers.

## C9-2 — Rate-flapping derivatives drop a factor the cyclic ones keep (p. 565)

Writing `c16 = 16/[gamma(1-e/R)^2]` and `kappa = 12(e/R)/[gamma(1-e/R)^3]`, the
Chapter 7 2×2 flapping system (pp. 468–473) evaluated at `mu = 0` gives

    da1s/dq = -(c16 + kappa) / [Omega (1 + kappa^2)]
    da1s/dp =  (1 - kappa c16) / [Omega (1 + kappa^2)]

Table 9.1 prints both **without** the `1 + kappa^2` denominator, while
`da1s/dB1 = -1/(1 + kappa^2)`, three rows further down the same page, keeps it.
The two rate rows are therefore a small-`kappa` approximation and the cyclic
rows are not.

`BasicRotorDerivativesHoverComp` defaults to `rate_flapping='table'` so the
printed numbers are reproduced; `rate_flapping='exact'` restores the factor and
agrees with `RateFlappingComp`. For the example helicopter `kappa = .0933`, so
the two differ by 0.87 % — invisible in the two significant figures Table 9.1
prints, and it propagates to `dX/dq` and `dX/dp` in Table 9.2.

## C9-3 — The two inflow rows of Table 9.1 do not have the same sign (p. 564)

Not a book error, but a trap. The first two rows read

    (dmu/dxdot)_M , ( dlambda'/dzdot)_M  =  (1/Omega R)_M     main rotor
    (dmu/dxdot)_T , (-dlambda'/dydot)_T  =  (1/Omega R)_T     tail rotor

The minus sign appears on the **tail rotor row only**. At scan resolution the
main rotor fraction bar is broken on its left and reads as a minus, which makes
the two rows look alike; at higher magnification the tail rotor minus is a
full-width dash in the numerator and the main rotor row has nothing. Hence

    dlambda'/dzdot =  1/(Omega R)   main rotor
    dlambda'/dydot = -1/(Omega R)   tail rotor

The asymmetry is the axis mapping: main rotor thrust acts along `-z`, tail rotor
thrust along `+y`, so only one of the two flips.

Independent confirmation from the book's own numbers: Table 9.8 (p. 580) gives
`dZ/dzdot = -rho A_b (Omega R)^2 (dCT/sigma/dlambda') (dlambda'/dzdot) = -261
lb/ft/sec`. With `dCT/sigma/dlambda' > 0`, that result is negative only if
`dlambda'/dzdot` is positive. A negative one would turn heave damping into heave
divergence — a helicopter that accelerates downward once it starts to drop.

## C9-6 — `dN/dr` in Table 9.2 is missing its `1/Omega` (p. 569)

The row is printed as

    (dN/dr)_M = 2 rho A_b (Omega R)^2 R C_Q/sigma    (governed engine,
                                                     counterclockwise rotation)

with a stated unit of ft-lb/rad/sec. The expression as written has the
dimensions of a moment, not a moment per unit rate, and evaluates to 96,872 —
21.7 times the printed 4,471, which is exactly the rotor speed `Omega = 21.67
rad/sec`. Restoring the `1/Omega` reproduces 4,471 with
`C_Q_bar/sigma = .0067`.

The corrected form is the physically recognisable one: `dN/dr = 2 Q_M/Omega`.
A governed engine holds `Omega` fixed while the airframe yaws, so the rotor
sees a shaft-speed perturbation; with torque going as `Omega^2`,
`dQ/dOmega = 2Q/Omega`. It is positive for counterclockwise rotation, i.e.
yaw-rate *divergence*, which is why the tail rotor's -17,797 is what makes the
total -13,326 stable (Table 9.4, p. 573).

Table 9.8 prints the same construction for forward flight correctly, as
`(2/Omega) rho A_b (Omega R)^2 C_T_bar/sigma` (p. 580), so the omission is
confined to Table 9.2.

## C9-7 — Stray subscript in the `dM/dtheta0` row (p. 568)

The row reads `-(dX_A/dtheta0)_M h_M + (dZ/dtheta0)_M l_M`. No `X_A` is defined
anywhere in the chapter and every other moment row of the table uses plain
`(dX/d.)_M`. Reading it as `dX/dtheta0 = 3,677` gives 45,966 against the printed
45,967, so the subscript carries no numerical content.

## C9-8 — `dR/dydot` in Table 9.3 has the wrong sign (p. 569)

The row is printed as

    (dR/dydot)_T = (dY/dydot)_T h_T          value: +78 ft-lb/ft/sec

`dY/dydot = -13` sits two rows above it and `h_T = +6`, so the printed
expression is **-78**. The sign of `h_T` is not in doubt: the three sibling
rows are built the same way and all three are exact against the book —
`dR/dp = (dY/dp) h_T = -468`, `dR/dr = (dY/dr) h_T = 2,886`,
`dR/dtheta0 = (dY/dtheta0) h_T = 58,476`. Nor is the sign of `dY/dydot`, which
is sideslip damping and must be negative; `dN/dydot = -(dY/dydot) l_T = +481`
confirms it independently.

So the equation is right and the value is a sign misprint. It propagates:
Table 9.4 (p. 572) lists `dR/dydot` as `-143` main rotor, `+78` tail rotor,
`-65` total, where `-221` belongs. This is the dihedral-effect derivative and
it does enter the hover lateral mode of p. 604, so it is not a harmless
misprint — the lateral roots there were computed from `-65`.

## C9-9 — The two torque rows of Table 9.3 have opposite signs (p. 570)

    (dM/dydot)_T   = rho A_b (Omega R)^2 R (dCQ/sigma/dlambda')(dlambda'/dydot)
                     value: -7 ft-lb/ft/sec
    (dM/dtheta0)_T = rho A_b (Omega R)^2 R (dCQ/sigma/dtheta0)
                     value: +11,276 ft-lb/rad

Both are the tail rotor's torque reacting on the airframe about the shaft
axis, so they cannot have opposite signs. Evaluating the first as printed,
with `dCQ/sigma/dlambda' = -.038` and `dlambda'/dydot = -1/(Omega R)_T`, gives
**+7.4**, not -7: two negatives multiply. Evaluating the second as printed
gives +11,276, matching.

The missing piece is the rotation-direction factor of Chapter 8, p. 487, where
`M_T = -s Q_T` with `s = +1` when the tail rotor blade nearest the main rotor
travels upward. Table 9.3's equations omit `-s` in both rows. Applying it
consistently with `s = +1` gives -7.4 and -11,276, which matches the printed
`dM/dydot = -7` and, through `dM/dr = -(dM/dydot) l_T`, the printed
`dM/dr = 274` exactly — and contradicts the printed `dM/dtheta0`.

`s = -1` would salvage `dM/dtheta0` alone and break the other two. So the
implementation takes `blade_closest='up'` (`s = +1`) as its default, matching
two of the three printed values, and exposes the option. Neither row enters
the hover analyses of pp. 596-605, which are decoupled.

## C9-10 — Unit misprint on `dR/dtheta0` in Table 9.3 (p. 570)

The row is annotated `58,476 ft-lb/rad/sec`. Collective is an angle, not a
rate, so the unit is `ft-lb/rad`. The sibling row it is built from,
`(dY/dtheta0)_T = 9,746 lb/rad` on p. 569, is annotated correctly, as are the
other three collective rows of the two tables. Values are unaffected; the
implementation declares `lbf*ft/rad`.

## C9-11 — Sign of the mass term in the hover Z-force equation (p. 596)

The three longitudinal equations of p. 596 carry their inertia terms as

    X:  -(G.W./g) x_ddot ...
    Z:  +(G.W./g) z_ddot ...
    M:  ... - I_yy q_dot ...

The Z equation is the odd one out, and it also omits `dZ/dz_ddot`. The matrix
on the facing page, p. 597, has `[dZ/dz_ddot - G.W./g]s` in that position:
minus, and the missing term restored.

The matrix is right. With a minus the middle row contributes a plunge root at
`dZ/dzdot / m = -182/621 = -0.293`, and p. 598 lists `-.28` among the four
roots. With a plus it would be `+0.293`, a plunge divergence, and the printed
quartic's `s^3` coefficient would change sign as well. The implementation
follows p. 597.

## C9-12 — `da1s/dq = 16/(gamma R)` on p. 601

The damping derivative used to reduce the period to a function of rotor radius
is printed as `16/(gamma R)`. It should be `16/(gamma Omega)`, which is
Table 9.1 at `e/R = 0`. Three reasons:

- **Dimensions.** `da1s/dq` is an angle per angular rate, i.e. seconds.
  `16/(gamma Omega)` is seconds; `16/(gamma R)` is per foot.
- **The result.** The printed period on the same page is
  `P = 2 pi sqrt(R)/sqrt(g (C_T/sigma) gamma/a)`. Substituting
  `16/(gamma Omega)` into the flapping form gives exactly that.
  `16/(gamma R)` gives `P` proportional to `1/sqrt(R)` instead — the opposite
  direction — and the whole point of the page is that the period grows with
  rotor size.
- **Table 9.1.** p. 565 gives `da1s/dq = -16/[gamma Omega (1-e/R)^2] - ...`,
  which at zero offset is `-16/(gamma Omega)`.

## C9-13 — Table 9.6 drops the flapping determinant Table 9.1 keeps

Set `mu = 0` in Table 9.6 (pp. 576-577) and six of its eight flapping rows
collapse exactly onto Table 9.1 (p. 565). Two do not:

| | Table 9.6 at `mu = 0` | Table 9.1 |
|---|---|---|
| `da1s/dB1` | -1 | `-1/(1 + kappa^2)` |
| `db1s/dA1` | +1 | `+1/(1 + kappa^2)` |

Table 9.1 keeps the flapping determinant on its cyclic rows and drops it on
its rate rows, which is entry C9-2. Table 9.6 drops it everywhere, so the two
tables are each internally inconsistent in a different place. For the example
helicopter `kappa = .0933` and the gap is 0.7 %, invisible at the printed
precision.

`BasicMainRotorDerivativesFFComp` reproduces Table 9.6 as printed.

## C9-14 — `da1s/dB1` is transposed in two rows of Table 9.8

Table 9.6 prints `da1s/dB1 = -1.188` and its formula,
`-(1 + 1.5 mu^2)/(1 - mu^2/2)`, gives -1.18848 at `mu = .30`.

Two rows of Table 9.8 are reproduced **exactly** by -1.118 instead:

| row | printed | with -1.188 | with -1.118 |
|---|---|---|---|
| `dX/dB1` | 18,601 | 19,766 | 18,601 |
| `dM/dB1` | -364,158 | -386,961 | -364,159 |

and a third row of the same table needs -1.188:

| `dZ/dB1` | 67,855 | 67,891 | 63,892 |

So the X and M columns and the Z column of Table 9.8 were computed with
different values of the same derivative. The formula is unambiguous, so -1.118
is a digit transposition. It propagates: Table 9.19's control column (p. 615)
lists -18,601 for `x(s)` against `B_1`, and Table 9.20 inherits it.

The component uses the formula value and the two rows come out 6.3 % high.

## C9-15 — `dN/dr` changes sign between hover and forward flight

    p. 569, Table 9.2:   +2 rho A_b (Omega R)^2 R C_Q/sigma        +4,471
    p. 582, Table 9.8:  -(2/Omega) rho A_b (Omega R)^2 R C_Q/sigma  -3,204

Same governed-engine term, `2 Q/Omega`, opposite sign. (p. 582 also prints the
`2/Omega` the hover row omits, which is entry C9-6.)

The forward-flight sign gives yaw damping, the hover sign yaw divergence — and
the hover total on p. 573 needed the tail rotor's -17,797 to overcome the main
rotor's +4,471, where in forward flight the two work together. One of the two
signs is wrong and the chapter does not say which; the hover row carries the
note "counterclockwise rotation" and the forward-flight row carries none.

The components reproduce each table as printed, so the sign flips between the
hover and forward-flight groups. Anyone building a single model across both
regimes has to pick.

## C9-16 — `dX/dzdot` uses `C_T_bar/sigma` where its neighbours use `dCH/da1s`

p. 578 prints three rows in the same structural shape. Two of them,
`dX/dxdot` and `dX/dtheta0`, put `dCH/sigma/da1s` in front of the flapping
derivative. The third, `dX/dzdot`, puts `C_T_bar/sigma` there.

Table 9.6 defines `dCH/sigma/da1s = C_T_bar/sigma + (a/8) lambda'`, so the two
differ by `(a/8)(-.023) = -.017`, 25 %. The row's sign turns on it: the printed
-6 requires `C_T_bar/sigma`, and with `dCH/sigma/da1s` the row comes out
**+0.3**. Equation and value agree inside the row; both disagree with the rest
of the column. The component follows p. 578.

## C9-18 — The coupled characteristic equation has two `s^2` terms (p. 616)

The eighth-order equation of the fully coupled six-degree-of-freedom system is
printed as

    s^8 + 10.02 s^7 + 28.88 s^6 + 48.98 s^5 + 26.28 s^4 - 137.88 s^3
        - 4.627 s^2 + 4.315 s^2 + .1675 = 0

A degree-eight polynomial has nine coefficients, and nine are printed, but two
of them sit on `s^2`. The second must be `s`.

The same page prints the eight roots — `-6.602, -2.907, -.7822 +/- 2.4432i,
-.1710, -.0391, .1828, 1.085` — and rebuilding the polynomial from them gives

    [1, 10.0157, 28.8753, 48.9629, 26.2328, -137.9079, -4.6198, 4.3154, .1675]

matching every printed coefficient to three or four figures, with **4.3154 in
the `s` position**. The misprint is typographic and carries no consequence.

p. 617 has a lighter version of the same slip: the longitudinal quartic is
printed as `s^4 + 1.545 s^3 - 2.618 s^2 + .0228 + .0949 = 0`, with the `s`
dropped from the fourth term altogether.

## C9-19 — Sign in the phugoid characteristic equation (p. 623)

The phugoid reduction drops the Z equation, leaving speed and pitch:

    | -m s^2 + dX/dxdot s      (dX/dq - m V Theta_bar)s - G.W. |
    | dM/dxdot s               -I_yy s^2 + dM/dq s             |

Expanding that determinant and stripping the one rigid-body root gives an `s`
coefficient of

    (1/(G.W./g)) [ (dX/dxdot)(dM/dq) **-** (dX/dq - (G.W./g) V Theta)(dM/dxdot) ]

p. 623 prints a **plus** on the second product.

**The book's own numbers settle it.** p. 624 tabulates the phugoid for the
54 ft² stabilizer at `omega = .342 rad/sec`, `P = 18.4 s`. The minus gives
`.3420` and `18.37 s`; the printed plus gives `.3487` and `18.0 s`.

The same sign carries into p. 624's one-degree-of-freedom reduction, which is
p. 623's cubic with `I_yy` set to zero. The hover version of the same 2x2
(p. 598) has the minus, and it is what makes that cubic's `s` term cancel.

`PhugoidMatrixComp` builds the matrix rather than the cubic, so the
determinant produces the corrected form by construction.

## Transcription audit

Tables 9.1 to 9.4 were re-read against the page scans a second time, cell by
cell, after the four components were written. Result: **no transcription
error found.** What was checked:

- **Table 9.1** (pp. 564-565), 13 rows. The two ambiguous cells were
  re-examined at high magnification: the `da1s/dq` cell is a *sum* of two
  stacked terms, both negative, and `dCT/sigma/dlambda'` is
  `1/[8/a + sqrt(sigma/2)/sqrt(C_T/sigma)]` with the second radical in the
  denominator of the second term, not under the first.
- **Table 9.2** (pp. 566-569), 35 rows: 11 on p. 566, 11 on p. 567, 11 on
  p. 568, 2 on p. 569. Every equation, value and sign confirmed.
- **Table 9.3** (pp. 569-570), 15 rows: 7 on p. 569, 8 on p. 570. Confirmed,
  including the two anomalies C9-8 and C9-9, which are in the book and not in
  the transcription.
- **Table 9.4** (pp. 571-573), 43 rows: 14 on p. 571, 13 on p. 572, 16 on
  p. 573. Every value *and every blank cell* confirmed. The main and tail
  columns were transcribed independently of Tables 9.2 and 9.3 and agree with
  them on all 43 rows.

A second, independent check was run on the flattening. Tables 9.2 and 9.3 are
stored as monomials in the inputs, whereas the book writes many rows in terms
of other rows (`dR/dq = (dR/db1s)(db1s/dq) + (dY/dq) h_M`). All 50 rows were
re-evaluated in the book's nested form by a separate script and compared
against the components: zero mismatch to 1e-9 relative.

## Validated anchors (G0)

| Quantity | Book | Model | Page |
|---|---|---|---|
| 2-DOF spring-weight-damper quartic | symbolic | exact match | 556 |
| Longitudinal characteristic equation, 115 kt | s⁴ + 1.545s³ − 2.618s² + .0228s + .0949 | 1.5445, −2.6182, .02280, .09488 | 617–618 |
| Longitudinal roots, 115 kt | −2.564, −.1782, .2106, .9867 | same to 2·10⁻³ | 617 |
| Routh's discriminant, same case | −.32 (negative → unstable) | negative | 617 |

## Conventions adopted in the implementation

- **Coefficient order.** Polynomials are stored in *ascending* powers of `s`:
  `char_coeffs[k]` multiplies `s**k`. Prouty writes them descending
  (`A s⁴ + B s³ + …`), so `char_coeffs[-1]` is his `A` and `char_coeffs[0]` his
  constant term `E`. The constant term is exactly the spiral-stability
  criterion of p. 633, so it is directly usable as a constraint.
- **Zero roots.** Prouty writes the equations of motion in displacements, so
  every determinant carries a factor `s**k` from the rigid-body position modes
  (the degree-6 determinant of p. 618 versus the quartic of p. 617). These
  factors are divided out through the `n_zero_roots` option, and a test asserts
  that the removed coefficients are numerically zero rather than merely small.
- **Normalisation.** The determinant is divided by its leading coefficient, as
  the book always does before quoting a characteristic equation.
- **Roots stay off-model.** Root ordering changes discontinuously with the
  coefficients, so roots and modal metrics (period, time to half/double
  amplitude, damping ratio) live in `prouty.stability.postprocess` and are
  never part of the OpenMDAO graph.

## Validated anchors (G1)

Table 9.1, main rotor column (pp. 564–565), example helicopter at
`Omega R = 650`, `e/R = .05`, `gamma = 8.1`, `a = 6`, `sigma = .085`,
`A_b = 240`, `C_T/sigma = .0849`, `theta_0 = .2794`, `theta_1 = -.1396`,
`v_1/(Omega R) = .062`, `a_0 = .075`. Every printed value is reproduced to the
precision it is printed with; the largest relative gap is `dCQ/dlambda'`
(-.0757 against -.076, 0.4 %).

The aerodynamic state above is not printed as such in Chapter 9. It was
recovered by inverting the table itself: `C_T/sigma` from `dCT/dlambda' = .49`,
`theta_75` from `dCH/da1s = .040`, `v_1/(Omega R)` from `dCQ/dlambda' = -.076`,
and `theta_0`, `theta_1` from `da1s/dmu = .34` with the -8° twist of Chapter 7.
The set is over-determined and closes to better than half a printed digit on
all five rows, so it is the state Prouty used.

The eight mirrored outputs (`d_lambda_d_zdot`, `d_lambda_d_ydot`, `d_b1s_dp`,
`d_b1s_dq`, `d_b1s_dA1`, `d_b1s_dB1`, `dCY_sigma_db1s`, `dR_db1s`) are checked
against their Table 9.1 partners and, downstream, against the two symmetries
Table 9.2 prints on p. 566: `dY/dq = dX/dp = -355` and `dY/dp = -dX/dq =
-1,008`. The model gives -360.8 and -1,012.7, the 1.5 % gap being Prouty
rounding `da1s/dp` to .037 and `da1s/dq` to -.105 before multiplying.

## Validated anchors (Table 9.2)

All 35 printed rows of Table 9.2 (pp. 566-569), reproduced from the printed
Table 9.1 values plus the geometry the table itself implies. Rows hold to a
relative tolerance of 0.2 %, which is what two-significant-figure inputs
support, with one exception noted below.

Inputs not printed in Chapter 9 and recovered from the table:

| quantity | value | recovered from |
|---|---|---|
| `h_M` | 7.5 ft | `dR/dp = (dR/db1s)(db1s/dp) + (dY/dp) h_M = -28,659` |
| `l_M` | -0.5 ft | `dM/dzdot = (dZ/dzdot) l_M = 91` |
| `y_M` | 0 | `dR/dzdot = (dZ/dzdot) y_M = 0` |
| `dCH/sigma/da1s` | .0398 | `dX/dq`, `dX/dp`, `dX/dA1`, `dX/dB1`, all four exact |
| `a1s_bar + i_M` | -.025 | `dX/dtheta0 = 3,677` |
| `b1s_bar` | -.027 | `dY/dtheta0 = -3,971` |
| `C_Q_bar/sigma` | .0067 | `dN/dr = 4,471`, with C9-6 applied |

`h_M = 7.5` is confirmed independently by `dR/dtheta0 = (dY/dtheta0) h_M =
-29,783` and by `dR/dA1` and `dR/dB1`, all three exact.

Only the **sum** `a1s_bar + i_M` is recoverable; `dX/dtheta0` and `dM/dtheta0`
are the only rows either appears in, and both see the sum. The split has to
come from the Chapter 8 trim solution.

### The one row that cannot be held

`dR/dxdot`, and with it `dM/dydot`, which Table 9.2 defines as equal to it.
The printed formula is `(dR/db1s)(db1s/dmu)(dmu/dxdot) + (dY/dxdot) h_M`. The
first term is `200,940 x .10 x .00154 = 30.9`; the second needs `dY/dxdot`,
whose unrounded value is 1.48 and whose printed value in the row above is 1.
Using 1.48 gives 42.0, using 1 gives 38.4, and the book prints 39.

The neighbouring `dM/dxdot` row settles which Prouty meant elsewhere: it needs
`dX/dxdot`, printed as -5, unrounded -5.02, and `200,940 x .34 x .00154 +
5.02 x 7.5 = 142.9` matches the printed 143 exactly — so that row used the
unrounded value. `dR/dxdot` is the inconsistent one. The model uses unrounded
values throughout and returns 42.0.

Nothing downstream depends on it in hover: `dR/dxdot` and `dM/dydot` are
longitudinal-to-lateral cross terms, and both hover analyses Prouty runs
(pp. 596-605) are decoupled.

## Validated anchors (Table 9.3)

Thirteen of the fifteen printed rows of Table 9.3 (pp. 569-570); the two
exceptions are C9-8 and C9-9 above.

Geometry recovered from the table, each value fixed by three separate rows:

| quantity | value | recovered from |
|---|---|---|
| `h_T` | 6 ft | `dR/dp`, `dR/dr`, `dR/dtheta0` |
| `l_T` | 37 ft | `dN/dr`, `dN/dtheta0`, `dM/dr` — the same 37 ft Chapter 8 uses |
| `rho A_b (Omega R)^2` | 19,492 | `dY/dtheta0 = 9,746` |
| `R_T` | 6.5 ft | `dM/dtheta0 = 11,276` |

Rows downstream of `dY/dydot` all sit 1.5 % high. The cause is a single
rounding: the unrounded `dY/dydot` is -13.19 and Prouty prints -13, then
carries the printed -13 onto the arms. `dY/dr = -(dY/dydot) l_T` is `481` in
the book, which is exactly `13 x 37`; the unrounded chain gives 488. Nothing
can reproduce both `dY/dtheta0 = 9,746` and `dY/dydot = -13` exactly, since
the two fix `rho A_b (Omega R)^2` at 19,492 and 19,204 respectively. The
scale is anchored on `dY/dtheta0`, which no rounding touches.

## Validated anchors (Table 9.4)

All 43 printed totals of Table 9.4 (pp. 571-573), reproduced from the printed
Table 9.2 and 9.3 columns. The main and tail columns of Table 9.4 were
transcribed independently of Tables 9.2 and 9.3 and agree with them on every
one of the 43 rows, which is the cross-check that the three transcriptions are
right.

Structure of the table, worth recording because it is not obvious from the
printed layout:

- **43 rows in the book, 44 in the component.** Table 9.4 has no `dR/dzdot`.
  Table 9.2 does, as `(dZ/dzdot) y_M = 0`, zero only because the example
  helicopter's main rotor is on the centreline. The row is kept so a laterally
  offset rotor is not silently dropped.
- **Only six rows take both rotors**: `dY/dydot`, `dY/dp`, `dR/dydot`,
  `dR/dp`, `dM/dydot`, `dN/dr`. Every other row has a blank in one column.
- **Collective is never summed.** Each rotor has its own, so the six main rotor
  collective rows and the four tail rotor ones stay apart, as `theta_0M` and
  `theta_0T` in the book and `_M` / `_T` suffixes here.

### Two totals that depart from the book

| row | main | tail (book) | total (book) | tail (here) | total (here) |
|---|---|---|---|---|---|
| `dR/dydot` | -143 | +78 | -65 | -79 | **-222** |
| `dM/dtheta0_T` | — | +11,276 | +11,276 | -11,276 | **-11,276** |

Both originate in `TailRotorDerivativesHoverComp` (C9-8, C9-9); the totals
component only adds up what it is given and has no switch of its own.

`dR/dydot` is the one to watch. It is the dihedral-effect derivative, it is
three times larger once C9-8 is applied, and it enters the hover lateral mode
of p. 604 — whose published roots were computed from -65.

## Validated anchors (HoverDerivativesGroup)

End-to-end: the example helicopter's physical parameters go in, Table 9.4 comes
out through Table 9.1 for each rotor and then Tables 9.2 and 9.3. No printed
intermediate value is fed in anywhere.

**Excluding the six rows below, the worst error across all 43 printed totals is
2.0 %**, which is what two-significant-figure inputs support through three
layers of multiplication.

| row | group | book | cause |
|---|---|---|---|
| `dR/dydot` | -222.9 | -65 | C9-8, tail rotor sign misprint |
| `dM/dtheta0_T` | -11,274 | +11,276 | C9-9, tail rotor torque sign |
| `dY/dxdot`, `dX/dydot` | ±1.48 | ±1 | Prouty rounds 1.48 to 1 |
| `dR/dxdot`, `dM/dydot` | 42.0, 34.6 | 39, 32 | the same 1.48 carried onto `h_M` |

### The tail rotor state

Chapter 9 prints no tail rotor parameters. They were recovered the same way as
the main rotor's, by inverting Table 9.1's tail column:

| quantity | value | recovered from |
|---|---|---|
| `sigma_T`, `CT_sigma_T` | .1461, .0828 | `dCT/sigma/dlambda' = .44` with `A_b_T` from Table 9.3 |
| `theta_0_T` | .1888 rad (10.8 deg) | `dCQ/sigma/dlambda' = -.038` and `da1s/dmu = .34` together |
| `v_1/(Omega R)_T` | .0817 | the same pair |
| `a_0_T` | .0375 rad | `db1s/dmu = (4/3) a_0 = .05` |
| `e/R_T` | 0 | the four zeros in the tail column: teetering |

The first attempt took `v_1/(Omega R)_T` from momentum theory instead, giving
`dCQ/sigma/dlambda' = -.0454` against the printed -.038 and putting `dM/dr`
19 % out. Solving the two printed rows simultaneously closes both exactly. The
lesson generalises: where Chapter 9 does not print a state, invert its own
table for it rather than reconstructing it from theory, since the table is what
the printed values were computed from.

### The two chart entries

p. 564 gives no equation for `dCT/sigma/dtheta0` and `dCQ/sigma/dtheta0`; it
says to take them from the Chapter 1 hover charts. They are ordinary inputs of
the group, defaulting to the Table 9.1 values, so it runs standalone.

`derivative_source='model'` replaces them with `HoverChartSlopesComp`, which
derives both in closed form from Chapter 1. See "What lambda-prime is" below
for the derivation and its accuracy.

## What lambda-prime is, and the factor of two

Appendix D, p. 707, defines the two inflow ratios:

| symbol | definition |
|---|---|
| `lambda` | inflow ratio with respect to the **swashplate** |
| `lambda'` | inflow ratio with respect to the **tip path plane** |

Chapter 1, p. 19, gives the thrust of an ideally twisted blade as
`C_T/sigma = (a/4)(theta_t - phi_t)` with `phi_t = sqrt(C_T/2)`, and p. 20
relates it to the collective of a linearly twisted blade,
`theta_0 = (3/2) theta_t - (3/4) theta_1`.

### The trap

Perturbing the inflow by a climb rate and letting the induced velocity respond
through `lambda_i = sqrt(sigma x/2)` gives, with `x = C_T/sigma` and
`k = dlambda_i/dx = lambda_i/(2x)`:

    dx/dlambda' = -1 / [4/a + k]  =  -0.982     for the example helicopter

Table 9.1 prints `+1/[8/a + sqrt(sigma/2)/sqrt(x)]` = **0.491** — exactly half,
which looked at first like an error in the book.

It is not. Momentum theory **in climb** gives

    v_1 = -V_c/2 + sqrt((V_c/2)^2 + T/(2 rho A))

so the total inflow through the disc changes by only half the climb rate: the
induced velocity absorbs the other half. `dlambda/dlambda_c = 1/2` at hover, and

    dx/dlambda_c = -(a/8) / [1 + (a/4)k] = -1 / [8/a + sqrt(sigma/2)/sqrt(x)]

which is Table 9.1's expression, to 0.2 % on the printed value. The sign is
convention: `lambda'` grows with descent, and descent adds thrust.

The lesson for the rest of the chapter: a perturbation that changes the rotor's
own inflow cannot be differentiated with the induced velocity held frozen.

### The closed-form slopes

With the factor understood, both chart entries follow from the same
denominator. Using `dx/dlambda'` from Table 9.1 itself:

    dCT/sigma/dtheta_0 = (4/3) dx/dlambda'
    dCQ/sigma/dtheta_0 = dx/dlambda' [2 k_induced lambda_i + (dcd/dalpha)/a]

the second from `C_Q/sigma = lambda_i x + c_d/8` (p. 22) and
`alpha_bar = 6x/a` (p. 24).

Accuracy at the example helicopter's `C_T/sigma = .086`, against both the
printed chart and this repository's own Chapter 1 model differentiated at the
same operating point (`theta_0 = 17.82 deg`):

| quantity | closed form | Chapter 1 model | Table 9.1 |
|---|---|---|---|
| `dCT/sigma/dlambda'` | .4911 | — | .490 |
| `dCT/sigma/dtheta0` | .6548 | .5986 | .610 |
| `dCQ/sigma/dtheta0` | .0641 | .0748 | .078 |

Two conclusions. The Chapter 1 model reproduces both chart entries to within
4 %, so the charts are what Table 9.1 says they are. And the closed form is
7 % high on thrust and 18 % low on torque at `k_induced = 1`, the ideal
momentum value; the shortfalls are tip loss and the empirical corrections of
pp. 69-76, which no closed form carries. `k_induced = 1.2` brings torque within
2.6 %, but 1.2 is fitted to this one rotor, so the default stays at 1.

`derivative_source='table'` therefore remains the default. `'model'` is for
letting an optimiser see the slopes move with the design, not for reproducing
Table 9.4.

## Validated anchors (G4, longitudinal hover)

`HoverLongitudinalMatrixComp` feeding `PolyDeterminantComp(degree_out=4)`, from
the Table 9.4 totals with `G.W. = 20,000 lb` and `I_yy = 40,000 slug ft^2`.
Both inertias are confirmed by Table 9.20's own `-621 s^2` and `-40,000 s^2`
(p. 614).

| | model | book (pp. 597-598) |
|---|---|---|
| characteristic equation | s⁴ + 1.0175 s³ + .2123 s² + .1151 s + .0337 | s⁴ + 1.02 s³ + .215 s² + .12 s + .034 |
| real roots | -.875, -.293 | -.89, -.28 |
| unstable pair | .0752 ± .3548i | .076 ± .360i |
| period | 17.71 s | 17.5 s |
| time to double | 9.22 s | 9.1 s |

The `s` coefficient is the widest gap, .1151 against a printed .12: that
coefficient is exactly `(g/I_yy)(dM/dxdot) = 32.2 x 143 / 40,000 = .1151`, and
.12 is it rounded to two figures.

### Two structural facts the tests pin down

`dM/dzdot` is 91 in Table 9.4 and has **no effect** on the determinant. With
`dZ/dxdot` and `dZ/dq` zero in hover the middle row is `[0, ., 0]`, and
expanding along it removes the whole `zdot` column from every other term. This
is why the printed quartic contains only `dX/dxdot`, `dZ/dzdot`, `dM/dq` and
`dM/dxdot`.

The cross terms `(dM/dq)(dX/dxdot) - (dM/dxdot)(dX/dq)` cancel exactly, as
p. 597 says they do for a single rotor, which is why the quartic's `s`
coefficient is the gravity term alone. Nothing in the implementation enforces
it — it falls out of Table 9.2's structure, where both `dX` rows carry
`dCH/sigma/da1s` and both `dM` rows carry `dM/da1s` and `-h_M`. Rebuilt from
the Table 9.1 pieces it closes to one part in 10^9, so the printed quartic is
an independent check on Table 9.2.

### Vectorisation

`PolyDeterminantComp`, `MatrixColumnSubstitutionComp` and
`RouthDiscriminantComp` now take `num_nodes` and the polynomial helpers batch
over leading axes. The forward-flight stability maps and root loci of
pp. 619-634 sweep a parameter, so the alternative was reworking G4 later.

`PolyDeterminantComp` also gained `degree_out`. The hover matrix has one
quadratic entry and eight linear or constant ones, so its determinant is a
quartic while the entrywise bound says degree six. The two top coefficients
vanish identically, and without `degree_out` `normalize` would divide by one of
them.

## Validated anchors (G4, two-degree-of-freedom hover)

`HoverLongTwoDofMatrixComp` feeding `PolyDeterminantComp(n=2, degree_out=3)`
and `RouthDiscriminantComp(degree=3)`, from the same Table 9.4 totals.

| | model | book (pp. 598-602) |
|---|---|---|
| characteristic equation | s³ + .72453 s² − .000034 s + .11511 | s³ + .724 s² + .115 |
| real root | −.8749 | −.87 |
| unstable pair | .0752 ± .3548i | .075 ± .355i |
| period | 17.71 s | 17.7 s |
| time to double | 9.22 s | 9.2 s |
| R.D.(3) | −.11514 | −.115 |

Every line lands inside the last printed digit.

### The signs, checked three ways

p. 598 writes the two equations with `−(G.W./g) x_ddot`, `−G.W.·Θ` and
`−I_yy q_dot`. Figure 9.9 (p. 599) prints the same pair with the inertia moved
to the right-hand side, and Figure 9.8 feeds `−G.W.` from `Θ` into the X-force
summer and `1/I_yy` into `q_dot`. All three agree. Worth recording because the
*three*-degree-of-freedom Z-force equation on p. 596 does not agree with its own
matrix (C9-11) — the 2-DOF pair has no such problem.

### The missing s term

The printed cubic has no `s` term. The determinant does produce one,
`[(dX/dxdot)(dM/dq) − (dX/dq)(dM/dxdot)]/(m I_yy)`, which p. 597 shows cancels
identically for a single rotor. Nothing in the implementation forces it to
zero; with two-significant-figure Table 9.4 inputs it lands at −3.4e−5 beside
an `s²` coefficient of .72.

That cancellation is structural, not incidental, and a test pins the
consequence: zeroing `dM/dxdot` on its own is **not** a physical case. Table
9.2 builds `dM/dxdot` and `dM/dq` from the same `dM/da1s` and the same `−h_M`,
so a rotor with no hub spring sitting on the c.g. zeroes both, and only then do
R.D. and the constant term go to zero together — Table 9.17's tests 3 and 5.
Zeroing `dM/dxdot` alone leaves `s` coefficient `(dX/dxdot)(dM/dq)/(m I_yy)`,
and R.D. picks it up as `BC` instead of `−AD`.

### Table 9.17 verdicts reproduced

For the example helicopter: test 4 fires (R.D. negative, unstable), test 2
fails, test 3 does not hold — exactly the column Prouty fills in on p. 602.
Reversing the sign of `dM/dxdot`, the teetering-rotor case of p. 603 and
Figure 9.10, turns R.D. positive so that test 2 passes, and pushes the constant
term negative so that test 6 fires instead: the unstable oscillation is traded
for a pure divergence, which is the drift into translational flight Prouty
describes.

## Validated anchors (G4, Hohenemser period)

| output | model | book (pp. 600-601) |
|---|---|---|
| `omega_N_squared` | .1607 rad²/s² | — |
| `period` | 15.675 s | 15.7 s |
| `period_flapping` | 15.688 s | 15.7 s |
| `period_radius` | 17.799 s | — |
| coefficient of √R | 3.2496 | 3.2 |

### The second printed form is exact, not an approximation

p. 600 gives `omega_N` twice, once from `dM/dxdot` and `dM/dq` and once from
`da1s/dmu` and `da1s/dq`. It looks as though the second drops the mast height,
since Table 9.2 builds

    dM/dxdot = (dM/da1s)(da1s/dmu)(dmu/dxdot) - (dX/dxdot) h_M
    dM/dq    = (dM/da1s)(da1s/dq)             - (dX/dq)    h_M

It does not. Both `dX` derivatives carry the same
`-rho A_b (Omega R)^2 (dCH/sigma/da1s)` in front of the same flapping
derivative, so each moment factors as that flapping derivative times
`(dM/da1s + K h_M)`, and the bracket cancels in the ratio. Rebuilt from the
Table 9.1 pieces the two forms agree to machine precision; on the rounded
Table 9.4 values they differ by 0.08 %.

This is the third place in the chapter where the same structural cancellation
appears — after the missing `s` term of the hover cubic and the
`(dM/dq)(dX/dxdot) - (dM/dxdot)(dX/dq)` of p. 597.

### The radius form rests on two substitutions, not one

Prouty names only the first, "a small white lie concerning the coefficient of
the induced velocity ratio":

| step | `da1s/dmu` | `da1s/dq` | period |
|---|---|---|---|
| exact | .34 | -.105 | 15.69 s |
| white lie only | .229 | -.105 | 19.10 s |
| plus `e/R = 0` | .229 | -.0912 | 17.80 s |

The white lie comes from Chapter 3 at `mu = 0` carrying `4 v_1/(Omega R)`
where `da1s/dmu` carries `2 v_1/(Omega R)`. The second substitution, unnamed,
is the zero hinge offset behind `16/(gamma Omega)` — the example helicopter has
`e/R = .05`. The two errors partly cancel: separately +22 % and -7 %, together
+13 %.

### Not oscillatory everywhere

`omega_N^2` goes negative when `dM/dxdot` changes sign, the teetering
low-rotor case of p. 603. The mode is then a divergence and has no period, so
the three period outputs return zero with zero partials and a warning.
`omega_N_squared` stays exact and differentiable throughout and is what to
constrain.

## Validated anchors (G4, lateral and directional hover)

### Yaw, p. 605

| | model | book |
|---|---|---|
| root | -.38074 | -.38 |
| time to half | 1.821 s | 1.82 s |

Both exact. `I_zz = 35,000 slug ft^2` is not printed in the section; it comes
from Table 9.20's `-35,000 s^2` (p. 615) and reproduces the published root to
four figures, which confirms the reading.

### Lateral, p. 604

p. 604 prints **no equations at all** — it says the lateral mode "could have
been treated in the same manner by using the moment of inertia in roll instead
of pitch". One sign does not follow from that instruction.

**The gravity term is `+G.W.`, not `-G.W.`** Transposing the longitudinal
equations symbol for symbol gives the wrong sign. Table 9.19 (p. 615) writes
the lateral equations out: the Y row of the `Phi` column is
`(dY/dp + (G.W./g) V_bar Theta_bar)s + G.W.`, against `(dX/dq - ...)s - G.W.`
in the X row of the `Theta` column. Table 9.20 confirms it numerically on the
facing page — `-3964 s + 20000` for Y against `3927 s - 20000` for X. It is
also what the axes require: with `x` forward, `y` right and `z` down, a nose-up
`Theta` puts `-W sin(Theta)` along body `x` while a right-wing-down `Phi` puts
`+W sin(Phi)` along body `y`.

The consequence is that the lateral cubic's constant term is
`-(g/I_xx)(dR/dydot)` where the longitudinal one is `+(g/I_yy)(dM/dxdot)`.

With Table 9.4 as printed and `I_xx = 5,000 slug ft^2` (Table 9.20's
`-5000 s^2`):

    s^3 + 5.8544 s^2 + .14609 s + .41860 = 0
    roots: -5.842,  -.0064 +/- .2676i     P = 23.5 s

a fast roll convergence and a slow, barely damped lateral oscillation.

**The `s` term does not cancel here.** The longitudinal cubic loses it because
`dX/dxdot` and `dX/dq` share a factor and so do `dM/dxdot` and `dM/dq`
(p. 597). Laterally, `dY/dydot` and `dR/dydot` carry tail rotor contributions
proportional to nothing the roll derivatives contain. Restricting the inputs to
the main rotor alone brings the cancellation back — the residual drops to
-2.7e-4, the same -849 numerator the longitudinal case leaves, scaled by
`I_xx` instead of `I_yy`. So the difference is the tail rotor, not an error.

### C9-8 decides the lateral verdict

This is where the tail rotor sign misprint lands, as flagged when it was found.

| `dR/dydot` | R.D.(3) | lateral oscillation |
|---|---|---|
| -65, Table 9.4 as printed | +.437 | stable, real part -.0064, t½ = 109 s |
| -222, from Table 9.3's own equation | -.896 | **unstable**, real part +.0129, t_double = 54 s |

Both are within a hundredth of neutral, so the character of the mode barely
changes — a nearly undamped 12 to 24 second lateral oscillation either way,
which is what a hovering pilot actually flies. What does change is the verdict
Table 9.17 returns: test 2 passes on the printed value and fails on the
corrected one. Anyone using this chapter's lateral hover result as a stability
criterion gets the opposite answer depending on which value they take.

## Validated anchors (HoverStabilityGroup)

Five analyses of the same hover, from one derivative set. The group runs
standalone on the Table 9.4 defaults and reproduces every number pp. 596-605
print:

| output | model | book |
|---|---|---|
| `char_coeffs_long` | s⁴ + 1.0175 s³ + .2123 s² + .1151 s + .0337 | p. 597 |
| `char_coeffs_long_2dof` | s³ + .7245 s² + .1151 | p. 598 |
| `routh_discriminant_long_2dof` | −.1151 | −.115, p. 602 |
| `period` | 15.675 s | 15.7 s, p. 600 |
| `char_coeffs_lateral` | s³ + 5.854 s² + .1461 s + .4186 | p. 604 (unprinted) |
| `root_yaw` | −.3807 | −.38, p. 605 |
| `time_to_half_yaw` | 1.821 s | 1.82 s, p. 605 |

`routh_discriminant_long` is R.D.(4) on the quartic. Prouty does not compute
it — p. 602 works on the cubic because R.D.(3) is what he introduces — but the
quartic supports it and it comes out negative, agreeing with the unstable
oscillation.

### The quartic factors exactly

p. 600 says the two-degree-of-freedom roots are "almost the same" as the full
set's, with "the only root missing" being the plunge mode. They are not almost
the same: they are **exactly** the same.

In hover `dZ/dxdot` and `dZ/dq` are zero, so the heave equation decouples
completely and the determinant is

    quartic = (s - dZ/dzdot / m) x cubic

which the tests verify to 2e-16. Prouty's printed roots differ (−.89, −.28,
.076 ± .360i against −.87, .075 ± .355i) only because he solved two separately
rounded polynomials, and the 2-DOF period he quotes as 17.7 s against 17.5 s is
the same number twice. Give `dZ/dq` a value and the factorisation breaks, which
is what makes it a statement about hover rather than about the algebra.

So the ladder of simplification on pp. 596-601 has one real step, not two:
dropping the Z equation costs nothing at all, and dropping the fuselage inertia
costs 13 %.

### Chaining

`HoverDerivativesGroup` feeds `HoverStabilityGroup` by promotion alone: rotor
geometry and trim state in one end, hover stability out the other. Two Table
9.1 outputs are promoted out of the derivative group for it, `d_a1s_d_mu_M` and
`d_a1s_dq_M`, because p. 600 writes the Hohenemser period in terms of them.

One name is kept deliberately apart. `CT_sigma_bar_M`, the trim thrust
coefficient the p. 601 radius formula uses, is .086 including the vertical drag
penalty (Chapter 1, p. 18), while the value that reproduces Table 9.1 is .0849.
Calling both `CT_sigma_M` would make the two groups fight over one variable.

Chained, the lateral branch inherits `dR/dydot = -222` from Table 9.3's own
equation rather than the printed -65, so `routh_discriminant_lateral` comes out
negative — the C9-8 verdict, arrived at without anyone typing the number.

## Validated anchors (Table 9.6)

All fourteen printed rows of Table 9.6 (pp. 576-577), example helicopter in
level flight at 115 knots, `mu = .30`. Every row lands inside the last printed
digit.

| row | model | book |
|---|---|---|
| `d_mu_d_xdot` | .0015385 | .00154 |
| `d_lambda_d_xdot` | .0000370 | .000037 |
| `d_lambda_d_zdot` | .0013836 | .00138 |
| `d_beta_d_ydot` | .0051520 | .00515 |
| `dCH_sigma_da1s` | .06925 | .069 |
| `d_a1s_dq` | -.109773 | -.1098 |
| `d_a1s_dp` | .0395834 | .0396 |
| `d_a1s_dA1` | .0904674 | .0905 |
| `d_a1s_dB1` | -1.18848 | -1.188 |
| `d_b1s_dq` | -.0354211 | -.0354 |
| `d_b1s_dp` | -.100663 | -.101 |
| `d_b1s_dA1` | 1 | 1 |
| `d_b1s_dB1` | .0982588 | .0983 |
| `dM_da1s` | 200,941 | 200,940 |

### The eight flapping rows share four factors

Table 9.6 prints them as eight unrelated fractions. Writing
`c16 = 16/[gamma Omega (1-e/R)^2]`, `kappa = 12(e/R)/[gamma(1-e/R)^3]`,
`k192 = kappa c16`, and `f1 = 1 - mu^2/2`, `f2 = 1 + mu^2/2`,
`f3 = 1 - mu^4/4`, `f4 = 1 + (3/2)mu^2`, they become

    da1s/dq  = -c16/f1 - (kappa/Omega)/f3    db1s/dq  = -1/(Omega f2) + k192/f3
    da1s/dp  =  1/(Omega f1) - k192/f3       db1s/dp  = -c16/f2 - (kappa/Omega)/f3
    da1s/dA1 =  kappa f2/f3                  db1s/dA1 =  1
    da1s/dB1 = -f4/f1                        db1s/dB1 =  kappa f4/f3

`kappa` and `k192 = kappa c16` are the Chapter 7 quantities `flapping_2x2`
already provides, so the module does not recompute them. The rewriting is what
makes the `mu = 0` comparison with Table 9.1 (entry C9-13) a two-line check
rather than an exercise.

### Recovered inputs

Chapter 9 prints neither the trim tip-path-plane angle nor `C_T/sigma` at 115
knots. `lambda' = -.023` is the Table 9.5 trim condition (p. 574);
`C_T/sigma = .0865` follows from `dCH/sigma/da1s = C_T/sigma + (a/8)lambda' =
.069`; and `alpha_TPP = -.0366 rad = -2.1 deg` follows from
`dlambda'/dxdot = .000037`. All three are consistent with level flight at that
speed.

### Table 9.6 cannot be run down to hover

Both `lambda'` rows carry `sigma/(2 mu)`, and the thrust term carries
`(C_T/sigma)/mu`, from the Glauert high-speed induced velocity
`v_1/(Omega R) = C_T/(2 mu)`. As `mu` falls, `dlambda'/dxdot` grows without
bound — 65 times larger at `mu = .05` than at `mu = .30` — while
`dlambda'/dzdot` falls off proportionally to `mu`. The flapping rows are
perfectly well behaved at `mu = 0`; it is the inflow rows that break.

This is the concrete reason Chapter 9 keeps Tables 9.1 to 9.4 for hover and
9.5 to 9.9 for forward flight, and why they are separate groups here rather
than one set with `mu` swept through zero.

## Where Table 9.5 comes from, and how close the Chapter 3 model gets

Table 9.5 (p. 574) holds thirty numbers read off the Chapter 3 charts by hand.
The question is whether the repository's own Chapter 3 model can produce them,
and what the charts are tied to.

### The charts are one blade, not one helicopter

p. 570 states the premise: for a given rotor the forces and flapping are
uniquely determined by `mu`, `theta` and `lambda'`. Chapter 3, pp. 229-230,
fixes the blade at

| parameter | value |
|---|---|
| twist | -5 deg linear |
| airfoil | NACA 0012 |
| advancing tip Mach | 0.7 |
| chord/radius | 0.079 |
| tip loss factor | 0.97 |

and says plainly that the charts "are flexible enough to be used for rotors
with different parameters", listing the corrections.

**Solidity is not on that list, and it does not belong there.** Pushing `sigma`
from .0849 to .120 in `NumericalRotorGroup` at fixed `(mu, theta_0, lambda')`
moves `C_T/sigma` by 0.02 %. Solidity enters *downstream*, in Table 9.6's
inflow rows, as the `sigma/(2 mu)` of `dlambda'/dzdot`. The split is clean:
blade aerodynamics in the charts, momentum closure in Table 9.6.

The twist correction of p. 230, `theta_0_chart = theta_0 + .75(theta_1 + 5)`,
is a shift and not a scaling, so `dtheta_0_chart/dtheta_0 = 1` and the
*derivatives* transfer to another twist unchanged — only the trim point moves.
Checked: at -10 deg twist the equivalent chart collective reproduces
`C_T/sigma` to 5 %.

### Grid convergence: not the problem

`RotorChartGenerator` differentiated at `mu = .30`, `theta_0 = 13.5 deg`,
`lambda' = -.023`, narrow step:

| grid | `C_T/sigma` | `dCT/dlambda'` | `dCH/dlambda'` | `dCQ/dlambda'` |
|---|---|---|---|---|
| 12x15 | .09420 | .7572 | -.0563 | -.0034 |
| 16x20 | .09408 | .7553 | -.0564 | -.0034 |
| 24x24 | .09400 | .7543 | -.0565 | -.0035 |
| 32x32 | .09394 | .7531 | -.0566 | -.0034 |
| **book** | .0865 | **.79** | **-.070** | **+.010** |

Refining the grid by a factor of seven in points moves the derivatives by half
a per cent. The remaining 5 % on `dCT/dlambda'` is not discretisation.

### Step size: that *is* the problem

Table 9.5 holds **secants over wide windows, not tangents**. p. 576 says the
`mu` partials were taken as the difference between the `mu = .25` and
`mu = .35` charts — a secant over `Delta mu = .10` — and Figure 9.7 marks
`Delta lambda' = .020` and `Delta theta_0 = 2 deg` on the other two.

At 24x24, varying only the step:

| step | `dCT/dlambda'` | `dCH/dlambda'` | `dCQ/dlambda'` |
|---|---|---|---|
| .005 | .7673 | -.0524 | -.0078 |
| .010 | .7543 | -.0565 | -.0035 |
| **.020, Prouty's own** | **.6975** | **-.0727** | **+.0141** |
| book | .79 | -.070 | +.010 |

The small rows only agree at Prouty's step. `dCH/sigma/dlambda'` goes from
-.056 to -.073 against a printed -.070, and `dCQ/sigma/dlambda'` **changes
sign**, from -.0035 to +.0141 against a printed +.010.

So the sign disagreement flagged earlier is a step-size artefact, not a
disagreement between the model and the book. The function is curved over the
window Prouty reads across. He is aware of it for this very row: p. 629 says
the `C_Q/sigma` slope "will either have a negative slope, a positive slope, or
be almost flat as it is for the example helicopter", and spends a page on what
it does to Dutch roll damping through `dN/dzdot`.

`dCT/sigma/dlambda'` goes the other way, moving *away* from the book at the
wide step. The two effects are not reconcilable at a single step, which is the
honest summary: Table 9.5 is a hand reading, and the residual 5 to 10 % is the
reading, not the model.

### What this means for a forward-flight `derivative_source='model'`

It is defensible here in a way it was not in hover, because
`NumericalRotorGroup` is the same method that produced the charts rather than
a closed form standing next to them. But a model source must use Prouty's own
step sizes to land on his numbers, and `dCQ/sigma/dlambda'` will stay badly
conditioned whatever is done, because it is genuinely near zero and curved.

`RotorChartDerivativesComp` is therefore an `IndepVarComp` carrying the thirty
printed values — an `IndepVarComp` being what a model subsystem substitutes for
cleanly, and the swap point a `'model'` source would replace.

## Validated anchors (Table 9.7)

Four rows, p. 578, tail rotor at 115 knots.

| row | model | book | |
|---|---|---|---|
| `d_mu_d_xdot` | .0015385 | .00154 | |
| `d_lambda_d_xdot` | .0001690 | .000169 | |
| `d_lambda_d_ydot` | -.0012275 | -.00121 | **1.4 %** |
| `d_beta_d_ydot` | .0051520 | .00515 | |

### Two differences from Table 9.6, both from the axis the disc faces

A tail rotor thrusts along `+y` where the main rotor thrusts along `-z`, and
the two tables differ exactly where that matters and nowhere else.

The row pairing with the main rotor's `dlambda'/dzdot` is `dlambda'/dydot`, and
it carries a **leading minus sign** — the same asymmetry Table 9.1 prints for
hover (entry C9-3), repeated here. Fed the same solidity, advance ratio and
chart slope, the two rows are exactly equal and opposite, which a test checks.

`dlambda'/dxdot` uses `a1s_bar` where the main rotor uses `alpha_TPP_bar`.
Forward speed is *in the plane* of a tail rotor disc and contributes no inflow
directly; what tilts flow through it is the tail rotor's own longitudinal
flapping. The induced term behind it is identical in both tables.

### The one row that does not close

`dlambda'/dydot` depends on nothing but `sigma_T`, `mu` and
`dCT/sigma/dlambda' = 1.04`. Table 9.3 pins the tail rotor geometry —
`A_b = 19.40 ft^2` from `dY/dtheta0 = 9,746` and `R = 6.5 ft` from
`dM/dtheta0 = 11,276` — giving `sigma_T = .1461` and a row value of -.001228.

Inverting the printed -.00121 instead requires `sigma_T = .157`, seven per cent
larger, which contradicts Table 9.3. The geometry is taken as correct and the
1.4 % as rounding in Prouty's arithmetic. A test asserts both branches so the
choice is visible rather than buried.

### Recovered inputs

`C_T/sigma = .0339` from `T_T = 661 lb`, the Chapter 8 value already
established in this repository (p. 510), over the `rho A_b (Omega R)^2 =
19,492` Table 9.3 implies. `a1s_bar = .0653 rad = 3.74 deg` is recovered by
inverting the printed `dlambda'/dxdot = .000169`; it is the only unknown left
in that row once `C_T/sigma` is fixed.

## Validated anchors (Table 9.8)

All 41 rows, pp. 578-582, example helicopter at 115 knots. **Thirty-six land
within 5 % of the printed value.** Of the five that do not:

- `dX/dzdot` (-6.47), `dY/dxdot` (-0.57) and `dM/dydot` (-8.45) are single
  digits in the book and each **rounds to the printed integer**.
- `dX/dB1` and `dM/dB1` are the C9-14 transposition, 6.3 % high by
  construction.

So nothing in the table is unaccounted for.

### Recovered trim state

Chapter 9 prints none of these; each is inverted from one row of Table 9.8:

| quantity | value | from |
|---|---|---|
| `a1s_bar + i_M` | -.0174 rad | `dX/dtheta0 = -6,727` |
| `b1s_bar` | -.0136 rad | `dY/dtheta0 = 2,650` |
| `B_1 + a1s_bar` | .1362 rad (7.8°) | `dR/dydot = -246` |
| `A_1 - b1s_bar` | -.0361 rad | `dM/dydot = -8` |
| `C_H_bar/sigma` | -.00045 | `dY/dydot = -14` |
| `C_Q_bar/sigma` | .00480 | `dN/dr = -3,204` |

The last is the check worth having: `C_Q_bar/sigma = .0048004` gives
`Q_M = 34,726 ft-lb`, which is the Chapter 8 torque at 115 knots already
established in this repository. Two independent chapters agreeing on the same
number through a derivative row.

`C_T_bar/sigma = .086` comes from `dZ/dr = 1,914` and agrees with the hover
reading.

### One misread character caught

`dY/dydot` is printed as
`-rho A_b (Omega R)^2 [C_H_bar/sigma + C_?_bar/sigma (B_1 + a1s_bar)]
dbeta/dydot`. The second subscript renders as a small `r` in the scan. It is
`T`, not `Y`: the thrust vector tilted by `B_1 + a1s_bar` giving a side force
in sideslip, and it is also what the printed -14 requires. `C_Y_bar/sigma` was
tried first and left the row 17 % short.

### Three structural additions over hover

**Rate rows pick up a kinematic term.** `dX/dq` carries `-(dX/dxdot) h_M`: a
pitch rate puts a fore-and-aft velocity `-q h_M` at a hub `h_M` above the c.g.,
and in forward flight that velocity moves the H-force. Table 9.2 omits it
because in hover `dX/dxdot` is -5 against -12 here. It is 4 % of `dX/dq`.

**`dlambda'/da1s = mu`.** Tilting the tip path plane by `a1s` changes the
inflow through it by `mu a1s`, which is the whole of `dZ/dB1`. In hover the row
does not exist; a test confirms it scales linearly with `mu` and vanishes at
zero.

**`dZ/dr` and `dN/dr`** are the governed-engine `2 Q/Omega` terms, printed here
with the `2/Omega` explicit.

### Machinery

Many rows are printed in terms of other rows. `MonomialRowsComp` gained
`ref()` and `expand()`, so a table declares the nesting directly —
`'dM_dq': [term(...), ref(-1.0, 'dX_dq', ('h_M',))]` — and the flattening is
done once, mechanically, instead of by hand for 41 rows.

## Validated anchors (Table 9.9)

All fifteen rows, pp. 582-583, tail rotor at 115 knots. Every row inside
0.1 % except the two single-digit ones, `dY/dxdot` (-2.10 against -2) and
`dR/dxdot` (-12.59 against -13), which round correctly.

### The simplest table in the chapter

Three rows are aerodynamic — `dY/dxdot`, `dY/dydot`, `dY/dtheta0` — and the
other twelve are those three carried to the c.g. on `h_T` for rolling moment
and `-l_T` for yawing moment. In the implementation that is twelve `ref()`
entries and nothing else.

### It settles C9-8

Table 9.3 prints `(dR/dydot)_T = (dY/dydot)_T h_T` and a value of **+78**,
where that expression gives -78. Table 9.9 prints **the same equation for the
same derivative** with the same `h_T = +6`, and a value of **-147**, which is
`-24.5 x 6` exactly.

So the forward-flight table is consistent with the equation and the hover table
is not. That is independent confirmation that the hover +78 is a misprint
rather than a sign convention this implementation has misread — and it means
the corrected hover total of -222, and the lateral-mode verdict that follows
from it, stand.

### It sidesteps C9-9

Table 9.3 and Table 9.9 both have fifteen rows, but not the same fifteen.
Forward flight **adds** the three `xdot` rows, since a tail rotor in forward
flight sees its own advance ratio change, and **drops all three M rows** — the
tail rotor torque reacting about the shaft axis, whose two hover rows carried
contradictory signs. Chapter 9 never resolves that contradiction; it stops
printing the rows.

### One internal rounding

`dR/dr = (dY/dr) h_T = 5,442` and `dN/dp = -(dY/dp) l_T = 5,439` are the same
product, `24.5 x 6 x 37`, reached two ways. The book prints both, differing by
0.06 % because `dY/dr` was rounded to 907 before being multiplied. The model
gives one number for both, 5,444.

## Validated anchors (Table 9.10)

All seven rows, p. 584, horizontal stabilizer at 115 knots.

| row | model | book |
|---|---|---|
| `d_gammaC_d_zdot` | -.005152 | -.00515 |
| `d_epsMH_d_xdot` | -.000761 | -.00076 |
| `d_epsMH_d_zdot` | .000773 | .00077 |
| `d_epsFH_d_zdot` | .001080 | .00108 |
| `d_alphaH_d_xdot` | .000761 | .00076 |
| `d_alphaH_d_zdot` | .003299 | .00331 |
| `d_alphaH_d_zddot` | **-.000132** | **-.00014** |

### The one row Chapter 9 adds to Chapter 8

p. 584 says the Chapter 8 stabilizer equations "can be used almost as is". The
exception, given on p. 586, is a term for the time the main rotor downwash
takes to reach the tail:

    alpha_H = Theta + i_H - (eps_MH + eps_FH) - gamma_c
              - (deps_MH/dzdot) zddot (l_H/V)

`Delta t = l_H/V` is 0.17 seconds for the example helicopter, and it is the
whole of `dalpha_H/dzddot`. It matters only in unsteady motion, which is
exactly what Chapter 9 is about and Chapter 8 is not.

### The lag row rounds the wrong way

`dalpha_H/dzddot` comes out -1.32e-4 against a printed -1.4e-4. The row is
`(deps_MH/dzdot)(l_H/V)`, and `l_H` is pinned at **33.0 ft** by Table 9.11's
`dM/dq = -7,161` over `dZ/dq = -217`, both printed to three figures.
Reproducing -1.4e-4 instead would need `l_H = 35.2 ft`, which contradicts that
pair. The 6 % is Prouty rounding 1.31 up to 1.4 in a two-figure entry.

### Recovered inputs

Chapter 9 prints none of the stabilizer's own parameters. Three come from the
table itself and the rest from the Chapter 8 work already in this repository
(`q_H/q = .6`, `A_H = 18 ft^2`, `a_H = 4.0`, `A.R. = 4.5`, `delta = .02`,
`C_D0 = .0064`, `i_H = -.052 rad`, `alpha_H = -7.9 deg`, Table 8.5 p. 523):

| quantity | value | from |
|---|---|---|
| `v_H/v_1` | 1.49 | `deps_MH/dzdot = .00077` |
| `Z_M_bar` | -20,543 lb | `deps_MH/dxdot = -.00076` |
| `deps_F/dalpha_F` | 0.233 | `deps_FH/dzdot = .00108` |
| `l_H` | 33.1 ft | Table 9.11, `dM/dq` over `dZ/dq` |
| `h_H` | -2.97 ft | Table 9.11, `dM/dzdot = -219` |

`Z_M_bar = -20,543` against a 20,000 lb gross weight is the weight plus the
vertical drag download, which is the independent check that the recovery is
sound rather than a curve fit.

With those, Table 9.11's `dM/dxdot` reconstructs to 41.8 against a printed 42
without any further adjustment — the Chapter 8 stabilizer definition and the
Chapter 9 derivative table describe the same surface.

## Validated anchors (Table 9.11)

All eleven rows, pp. 585-586, horizontal stabilizer at 115 knots. Every row
that carries a number lands inside 1 %; three are printed only as "<1".

| row | model | book |
|---|---|---|
| `dX_dxdot` | -.362 | <1 |
| `dX_dzdot` | -.941 | -1 |
| `dX_dzddot` | .038 | <1 |
| `dZ_dxdot` | 1.292 | 1 |
| `dZ_dzdot` | -6.506 | -7 |
| `dZ_dzddot` | .260 | <1 |
| `dZ_dq` | -215.5 | -217 |
| `dM_dxdot` | 41.70 | 42 |
| `dM_dzdot` | -218.3 | -219 |
| `dM_dzddot` | 8.73 | 9 |
| `dM_dq` | -7,137 | -7,161 |

### Two brackets do all the work

    lift  (alpha_H - alpha_LO)[1 - 2 a_H (1+delta)/(pi A.R.)] + (alpha_H - i_H)
    drag  1 + [a_H(1+delta)/(pi A.R.)][2(alpha_H - alpha_LO)(alpha_H - i_H)
                                       + (alpha_H - alpha_LO)^2] + C_D0

evaluating to -0.1473 and 1.0192. Everything else is those two times
`(q_H/q) q A_H a_H` and then carried onto `l_H` and `h_H`.

The stabilizer's own parameters are the Chapter 8 ones already in this
repository (Table 8.5, p. 523) and were not adjusted: `q_H/q = .6`,
`A_H = 18 ft^2`, `a_H = 4.0`, `A.R. = 4.5`, `delta = .02`, `C_D0 = .0064`,
`i_H = -.052 rad`, `alpha_H = -7.9 deg`. Two arms and two trim forces come
from Table 9.11 itself: `l_H = 33.1 ft`, `h_H = -2.97 ft`,
`X_H_bar = -14.1 lb`, `Z_H_bar = 271 lb`.

### The `2/V` terms are dynamic pressure, not angle of attack

`dX/dxdot` and `dZ/dxdot` each carry `(2/V)` times the trim force. Forward
speed changes `q` as `V^2`, so the force the surface already carries grows as
`2/V`. It is the larger part of `dZ/dxdot`: 2.79 against -1.50 from the
angle-of-attack term, netting 1.29.

### Where Table 9.20's `9 s^2` comes from

`dM/dzddot = 8.73` is the only unsteady term in the airframe tables, and it is
the `9 s^2` in the M row of Table 9.20 (p. 614). It exists because of the
downwash lag row of Table 9.10: `dX/dzddot` and `dZ/dzddot` are the `zdot`
rows scaled by `(dalpha_H/dzddot)/(dalpha_H/dzdot) = -0.040`. Set the lag to
zero and the row vanishes.

### A printing artefact and a machinery fix

The two Z rows must share the drag bracket. p. 585 prints it clearly as
`(alpha_H - alpha_LO)^2` in `dZ/dxdot`; in `dZ/dzdot` the minus is broken by
the scan and reads `(alpha_H alpha_LO)^2`. No consequence.

`MonomialRowsComp` needed one fix for this table. `dX/dzddot` is `dX/dzdot`
times `(dalpha_H/dzddot)/(dalpha_H/dzdot)`, and `dX/dzdot` already carries
`dalpha_H/dzdot`, so after reference resolution the same input sat above and
below the line in one monomial. The derivative rule handled only one of the
two occurrences. `reduce_monomial` now cancels common factors at expansion
time, and `check_partials` agrees to 1e-8 across the table.

## Validated anchors (Table 9.12)

All six rows, p. 587, vertical stabilizer at 115 knots.

| row | model | book | |
|---|---|---|---|
| `d_beta_d_ydot` | .005152 | .00515 | |
| `d_etaTV_d_xdot` | .000617 | .00061 | 1.2 % |
| `d_alphaV_d_xdot` | .000617 | .00061 | 1.2 % |
| `d_etaTV_d_ydot` | -.001717 | -.00167 | **2.8 %** |
| `d_etaFV_d_ydot` | .000309 | .00031 | |
| `d_alphaV_d_ydot` | -.003744 | -.00379 | 1.2 % |

### The mirror of Table 9.10

Tail rotor sidewash on the vertical stabilizer stands where main rotor
downwash stood on the horizontal one, with `A_T` for `A_M` and `T_T` for
`Z_M`. The momentum relation is identical: doubling the disc halves the
coupling either way.

`deta_F/dbeta = 0.06` recovered from the printed `deta_FV/dydot = .00031` is
the same 0.06 the Chapter 8 work in this repository already uses (entry C8-5
of its own notes), so the two chapters agree on the fuselage sidewash without
anyone connecting them.

### The two sidewash rows do not share a sign convention

This is the trap in the table, and it is **not** a misprint.
`deta_TV/dxdot` carries a leading minus inside its own equation and
`deta_TV/dydot` does not, so the two angle-of-attack rows treat them
differently:

    dalpha_V/dxdot = +deta_T/dxdot
    dalpha_V/dydot = -(dbeta/dydot + deta_T/dydot + deta_F/dydot)

The first looks wrong next to Table 9.10, whose equivalent row is
`dalpha_H/dxdot = -deps_MH/dxdot`, and "correcting" it would be easy and
wrong.

Table 9.13 settles it from outside. Its `dY/dxdot = (2/V) Y_V_bar +
(q_V/q) q A_V a_V (dalpha_V/dxdot) = 5` inverts to `Y_V_bar = 287 lb` with
the positive value — which is the Chapter 8 vertical stabilizer force to three
figures — and to 606 lb with the negative one. An independent chapter, and one
not involved in writing Table 9.12, picks the printed sign.

### The row 3 % out is two tables disagreeing

`deta_TV/dydot` is `(dY/dydot)_T / [4 (q_V/q) q A_T]` and nothing else.
The same denominator makes `deta_TV/dxdot` land within 1 %, so the gap is in
the numerator: reproducing -.00167 needs `(dY/dydot)_T = -23.8`, where
Table 9.9 prints **-24.5**. Two tables of the same chapter differ by 3 % on
one derivative. The component uses Table 9.9's value.

## Validated anchors (Table 9.13)

All fourteen rows, pp. 587-589, vertical stabilizer at 115 knots.

| row | model | book | |
|---|---|---|---|
| `dX_dxdot` | .284 | <1 | |
| `dX_dydot` | **-4.21** | **-3** | contradicted |
| `dY_dxdot` | 4.58 | 5 | |
| `dY_dydot` | -14.23 | -14 | 1.7 % |
| `dY_dp` | -42.7 | -42 | 1.7 % |
| `dY_dr` | 498 | 490 | 1.7 % |
| `dR_dxdot` | **13.74** | **12** | contradicted |
| `dR_dydot` | -42.7 | -42 | 1.7 % |
| `dR_dp` | -128 | -126 | 1.7 % |
| `dR_dr` | 1,495 | 1,470 | 1.7 % |
| `dN_dxdot` | -160.3 | -161 | 0.4 % |
| `dN_dydot` | 498 | 490 | 1.7 % |
| `dN_dp` | 1,495 | 1,470 | 1.7 % |
| `dN_dr` | -17,437 | -17,150 | 1.7 % |

Nine rows sit 1.7 % high **together**, because they are one row carried on the
arms. That is one number out by 1.7 %, not nine.

Nothing about the fin was adjusted: `q_V/q = .6`, `A_V = 33 ft^2`,
`a_V = 3.0`, `A.R. = 3.2`, `delta = .01`, `alpha_LO = -.1012 rad`,
`Y_V_bar = 287 lb`, `X_V_bar = -58 lb` and `dD_int = 42.8 lb` are all Chapter 8
values already in this repository. `h_V = 3.0 ft` and `l_V = 35.0 ft` come from
Table 9.13 itself, each confirmed by three separate rows.

### Interference drag is not a correction term

`dD_int` is the biplane interference between the tail rotor and the fin
(Chapter 8, p. 509): **42.8 lb against 14.9 lb of clean fin drag**. It appears
in three of the four core rows, and in `dY/dydot` it appears twice — directly,
and as the factor `1/(1 - dD_int/Y_V_bar) = 1.175`. That factor exists because
the interference drag is itself proportional to the fin's side force, so a
sideslip perturbation feeds back on itself. A 17 % amplification.

It is also the one row of any airframe table that is not a straight product,
which is why this component is written out rather than declared as monomials.

Inverting the printed `dY/dydot = -14` for `dD_int` gives **40.7 lb** against
the Chapter 8 value of 42.8. Two chapters, 5 % apart, on a quantity neither
prints in the other's units.

### C9-17 — `dR/dxdot` contradicts `dN/dxdot`

Both are `dY/dxdot` carried on an arm, and the two arms are each confirmed
three ways elsewhere in the table:

    dR/dxdot = (dY/dxdot) h_V  = 12     needs dY/dxdot = 4.0
    dN/dxdot = -(dY/dxdot) l_V = -161   needs dY/dxdot = 4.6

The model gives 4.58, reproducing `dN/dxdot` to 0.4 % and overshooting
`dR/dxdot` by 15 %. The book's own two rows cannot both be right.

### `dX/dydot` cannot be reached

-4.21 against a printed -3. Reaching -3 needs the bracket `K_V = -.070`, hence
`alpha_V = -4.5 deg`. But `Y_V_bar = 287 lb` over `(q_V/q) q A_V a_V` fixes
`alpha_V` at `+0.38 deg`, and that same value is what makes `dY/dxdot` land on
`dN/dxdot`. The two constraints are incompatible and the row is a single digit
in the book.

## Validated anchors (Tables 9.14 and 9.15)

### Table 9.14, pp. 589-590

| row | model | book |
|---|---|---|
| `d_gammaC_d_zdot` | -.005152 | -.00515 |
| `d_beta_d_ydot` | .005152 | .00515 |
| `d_epsMF_d_xdot` | -.000507 | -.00051 |
| `d_epsMF_d_zdot` | .000515 | .00051 |
| `d_alphaF_d_xdot` | .000507 | .00051 |
| `d_alphaF_d_zdot` | .004637 | .00467 |

The same construction as Table 9.10 with two rows missing: the fuselage has no
upwash of its own to fly in, so there is no `eps_F` row, and it sits under the
rotor rather than 33 ft behind it, so there is no downwash lag row either.

**`v_F/v_1 = 1` where the horizontal stabilizer has 1.5.** The fuselage is in
the rotor's own induced velocity; the stabilizer is further aft where the wake
has contracted and sped up. Chapter 8's Table 8.4 states the 1 explicitly, and
this repository's `RotorDownwashComp` already uses it. With the same
`4 q A_M` and the same rotor `Z` derivative, the two downwash rows differ by
exactly that ratio — a test asserts it to 1e-6.

Table 9.14's other six entries — `df/dalpha_F`, `d(L/q)/dalpha_F`,
`d(S.F./q)/dbeta`, `d(M/q)/dalpha_F`, `d(N/q)/dbeta`, `d(R/q)/dbeta` — are
"from curves in Appendix A at trim conditions". They are wind tunnel data, not
equations, so they enter Table 9.15 directly as inputs with the printed values
as defaults, the same treatment Table 9.5's chart entries get.

### Table 9.15, pp. 590-591

| row | model | book |
|---|---|---|
| `dX_dxdot` | -8.18 | -8 |
| `dX_dzdot` | -.888 | -1 |
| `dY_dydot` | -54.86 | -55 |
| `dZ_dxdot` | 2.90 | 3 |
| `dZ_dzdot` | -19.16 | -19 |
| `dR_dydot` | 53.08 | 53 |
| `dM_dxdot` | -80.47 | -80 |
| `dM_dzdot` | 369.7 | 374 |
| `dN_dydot` | -189.2 | -190 |

Every row within 1.2 %. The plainest of the three airframe tables: no bracket,
no feedback, and the only one whose aerodynamic content is entirely
measurement rather than a lift curve slope.

Two kinds of row. The `2/V` ones — `dX/dxdot`, `dZ/dxdot` and half of
`dM/dxdot` — are dynamic pressure alone and need no Appendix A curve. The rest
are `q` times a curve slope times an angle derivative, and **the chordwise and
normal rows use different curves**: `dX/dzdot` follows `df/dalpha_F`, the
equivalent flat plate area, while `dZ/dzdot` follows `d(L/q)/dalpha_F`. A test
confirms each row moves with its own curve and not the other.

The trim forces are the Chapter 8 ones already in this repository,
unadjusted: `L_F_bar = -281 lb` and `D_F_bar = 794 lb` (Table 8.5, p. 524),
with `X_F_bar = -D_F_bar` and `Z_F_bar = -L_F_bar`. Only `M_F_bar` had to be
recovered, **-11,733 ft-lb** from `dM/dxdot = -80`, and it is the one fuselage
trim quantity Chapter 8 does not supply.

The fuselage is the bigger half of the sideslip damping: `dY/dydot = -55`
against the vertical stabilizer's -14.

## Validated anchors (Table 9.16)

All **fifty** rows, pp. 591-595, five contributors. Every printed total is
reproduced from the printed columns to 1e-10, and a separate test checks
Prouty's own arithmetic — that each printed total really is the sum of the
cells beside it. It is, on all fifty rows.

The forward-flight counterpart of Table 9.4 and four times its size, because
the airframe is in it. Twenty-five of the fifty rows are main rotor alone.

Collective is not summed, as in hover: six main rotor collective rows and
three tail rotor ones. There is **no** `dZ/dtheta0_T` or `dM/dtheta0_T`,
because Table 9.9 drops the tail rotor's M rows altogether (entry C9-9). The
structure of Table 9.16 confirms that omission was deliberate rather than a
printing accident.

### The chapter closes on itself

Ten of the totals appear verbatim in Table 9.20, p. 614 — the matrix the whole
forward-flight stability analysis of pp. 617-634 rests on:

| | | | |
|---|---|---|---|
| `dM/dxdot` | 144 | `dR/dp` | -33,738 |
| `dM/dzdot` | 650 | `dR/dr` | 6,912 |
| `dM/dzddot` | 9 | `dN/dxdot` | -136 |
| `dM/dq` | -43,752 | `dN/dydot` | 1,207 |
| | | `dN/dp` | 6,909 |
| | | `dN/dr` | -53,913 |

Those are **the same numbers `PolyDeterminantComp` was first validated against**
at the very top of this chapter, before any derivative table existed. A test
now builds p. 618's determinant from Table 9.16's own totals rather than from
hand-typed values and recovers p. 617's quartic,
`s⁴ + 1.545 s³ - 2.618 s² + .0228 s + .0949`, to 2e-3.

`dM/dzddot = 9` is worth singling out. It comes from the horizontal
stabilizer alone, it exists only because of the downwash lag row of Table 9.10,
and it is the `9 s²` in Table 9.20's M row. Four tables and eighty pages
between the lag and the coefficient.

### Three totals inherit a disagreement

- `dX/dB1 = 18,601` and `dM/dB1 = -364,158` are main rotor alone and carry the
  transposed `da1s/dB1 = -1.118` of entry C9-14.
- `dX/dydot = -7` takes the vertical stabilizer's -3, which entry C9-17 shows
  cannot be reached from the fin's own geometry.

None of the three originates in the totalling; the component only adds up what
it is given, exactly as in hover.

## Validated anchors (ForwardFlightDerivativesGroup)

Eleven tables wired, from the chart readings of Table 9.5 to the totals of
Table 9.16. Nothing printed is fed in as an intermediate: rotor geometry,
airframe geometry and the trim state go in one end.

**Forty-six of the fifty totals land within 5 %**, most within 2 %. The four
that do not are exactly the four rows the book already contradicts itself on:
`dX/dB1` and `dM/dB1` from entry C9-14, `dX/dydot` and `dR/dxdot` from C9-17.
A test asserts that set, so a fifth appearing would be a regression.

### The ten coefficients the stability analysis is built on

| row | group | Table 9.16 | |
|---|---|---|---|
| `dM/dxdot` | 141.9 | 144 | -1.5 % |
| `dM/dzdot` | 665.3 | 650 | 2.4 % |
| `dM/dzddot` | 8.78 | 9 | -2.5 % |
| `dM/dq` | -43,526 | -43,752 | -0.5 % |
| `dR/dp` | -34,552 | -33,738 | 2.4 % |
| `dR/dr` | 6,997 | 6,912 | 1.2 % |
| `dN/dxdot` | -137.0 | -136 | 0.7 % |
| `dN/dydot` | 1,222.6 | 1,207 | 1.3 % |
| `dN/dp` | 6,997 | 6,909 | 1.3 % |
| `dN/dr` | -54,459 | -53,913 | 1.0 % |

All within 2.5 % end to end, which is what a chain of eleven tables of two-
and three-figure numbers supports.

Built into p. 618's determinant, those give

    group  s⁴ + 1.5400 s³ - 2.6950 s² + .01844 s + .09507
    book   s⁴ + 1.545  s³ - 2.618  s² + .0228  s + .0949

with roots -2.581, -.1768, .2057 and 1.0124 against the printed -2.564,
-.1782, .2106 and .9867. **Rotor geometry in one end, p. 617's characteristic
equation out the other**, with nothing hand-typed but the three non-M rows of
the matrix. The `s` coefficient is the smallest and shows the accumulated
rounding most; the two unstable roots that decide the aircraft's behaviour are
within 2.5 %.

### The graph is coupled, not a stack

    main_chart --> main_basic --> main_table ----+--> total
    tail_chart --> tail_basic --> tail_table --+ |
    main_table --> horiz_nondim --> horiz_table--+
    main_table --> fuse_nondim  --> fuse_table --+
    tail_table --> vert_nondim  --> vert_table --+

Three of the five airframe tables read a rotor table. The horizontal
stabilizer and the fuselage fly in the main rotor's downwash, so Tables 9.10
and 9.14 take `dZ/dxdot` and `dZ/dzdot` out of Table 9.8; the vertical
stabilizer flies in the tail rotor's sidewash, so Table 9.12 takes `dY/dxdot`
and `dY/dydot` out of Table 9.9. Those four are the only explicit connections
in the group; everything else meets by promotion.

A test drives it the long way round: changing `dCT/sigma/dlambda'` on the main
rotor moves the horizontal stabilizer's and the fuselage's contributions, and
changing it on the tail rotor moves the vertical stabilizer's.

Another follows the downwash lag all the way through. `dM/dzddot = 9`, the
`9 s²` of Table 9.20's M row, exists only because of the `l_H/V` term Chapter 9
adds to Chapter 8's stabilizer equation. Set `l_H` to zero and the coefficient
vanishes — four tables and eighty pages away.

### Naming

Every variable takes its subsystem's tag — `_M`, `_T`, `_H`, `_V`, `_F` —
unless it is one quantity for the whole aircraft (`rho`, `V`, `q`, `A_M`,
`Z_M_bar`, `T_T`) or already says which component it belongs to.

The tags are not decoration. `sigma` is .085 on the main rotor and .146 on the
tail; `A_b` is 240 ft² and 19.4; `A_R`, `delta` and `alpha_LO` are different
surfaces on the two stabilizers; and `d_beta_d_ydot` is computed **four**
separate times, by Tables 9.6, 9.7, 9.12 and 9.14, which all agree on `1/V`
but are four outputs all the same.

The five dimensional tables promote their outputs with Table 9.16's own column
word instead — `_main`, `_tail`, `_horiz`, `_vert`, `_fuse` — so they meet the
totalling component by promotion and need no connect statement.

## Validated anchors (G7, longitudinal in forward flight)

`LongMatrixFFComp` feeding `PolyDeterminantComp(n=3, degree=2, n_zero_roots=2)`
and `RouthDiscriminantComp(degree=4)`, from the Table 9.16 totals.

| | model | book (pp. 616-617) |
|---|---|---|
| characteristic equation | s⁴ + 1.5445 s³ − 2.6162 s² + .02286 s + .09484 | s⁴ + 1.545 s³ − 2.618 s² + .0228 s + .0949 |
| R.D.(4) | −.3191 | −.32 |
| roots | −2.5631, −.1782, .2106, .9861 | −2.564, −.1782, .2106, .9867 |
| time to double | 0.70 s | "less than one second" |

Every line inside the last printed digit. All four roots are real: the example
helicopter's longitudinal instability at 115 knots is a **pure divergence**,
not an oscillation, and p. 617 attributes it to the 18 ft² horizontal
stabilizer being too small.

### Displacement form, not velocity form

Unlike the hover matrix of p. 597, every entry carries an extra `s`: the
states are `x(s)`, `z(s)`, `Theta(s)`. The determinant is degree six and
carries an `s^2` rigid-body factor, which `n_zero_roots=2` divides out. A test
checks both that the two stripped coefficients vanish and that the degree-six
term genuinely does not.

### Two terms forward flight adds

Both come from the trim velocity and neither exists in hover.

`(dX/dq - m V Theta_bar)`: a pitch rate in forward flight rotates the velocity
vector, and the trim attitude decides how much lands on the X axis. It is the
**larger** part of the entry — 1,937 of aerodynamic `dX/dq` against +1,990 of
kinematics, giving Table 9.20's 3,927.

`(dZ/dq + m V)`: the centrifugal term. A pitch rate at 194 ft/sec throws
120,555 lb of normal force against the −217 of aerodynamic `dZ/dq`. **The
aerodynamics are 0.2 % of that entry.**

`Theta_bar` is not printed in Chapter 9. It is recovered from Table 9.20's own
`3927 s` entry as −0.95°, a slightly nose-down attitude at 115 knots.

### Coupled against uncoupled

p. 617 compares the two root sets:

    uncoupled  -2.564, -.1782, .2106, .9867
    coupled    -2.907, -.1710, .1828, 1.085

They agree to within 13 %, which is what justifies studying the two submatrices
separately for the rest of the chapter.

## Validated anchors (G7, the longitudinal stability map)

Figure 9.15, p. 619, is Prouty's guide to resizing the horizontal stabilizer:
plot the aircraft in the plane of its angle-of-attack stability `dM/dzdot`
against its speed stability `dM/dxdot`, and the regions say what kind of
instability it has. Two curves bound them — Routh's discriminant vanishing,
and `E`, the constant term of the characteristic equation, vanishing.

### The `E = 0` boundary is exact, not fitted

p. 618 prints it as `dM/dzdot = (dM/dxdot)(dZ/dzdot)/(dZ/dxdot)` without
derivation. It comes straight out of the determinant. The only entry of the
matrix carrying a constant term is `-G.W.` in the X row, so only two of the six
Leibniz permutations reach the `s^2` coefficient of the degree-six determinant,
and they give

    E = G.W. [(dZ/dxdot)(dM/dzdot) - (dZ/dzdot)(dM/dxdot)] / (m^2 I_yy)

which is **.094842** for the example helicopter against the `.0949` of the
p. 617 quartic, and vanishes exactly on p. 618's line. A test sets `dM/dzdot`
to the boundary value and checks the determinant's constant term goes to zero
to 1e-10.

`dM/dzdot = -843 ft lb/(ft/sec)` is the boundary; the aircraft sits at `+650`,
a margin of 1,493 — comfortably on the oscillation side of *that* boundary even
though it is diverging for the other reason.

### p. 618's forms are exact, not fitted — a correction

These notes first called the p. 618 quadratic a fit "for drawing, not
computing", on the strength of its giving **-.3088** against the determinant's
**-.3191** at the nominal point. That was wrong, and the section below on the
map boundaries derives why: `R.D.` is *exactly* quadratic in the two
derivatives, and p. 618's six coefficients are that exact conic rounded to two
or three significant figures. Every one is within **1.5 %** of the derived
value, which is their printed precision. The 3 % at the nominal point is
rounding, not a fit.

`routh_map` and `ROUTH_MAP_COEFFICIENTS` are kept as the printed values;
`routh_conic` derives the exact ones.

### `E > 0` does not mean oscillatory

A negative constant term with a positive leading coefficient forces a positive
real root, so `E < 0` is a divergence with no further analysis. The converse
fails: the example helicopter has `E = +.095` **and four real roots**. That is
why Figure 9.15's right-hand boundary is not a formula — p. 618 says it "was
determined by finding combinations of the two derivatives that made the roots
switch from complex to real".

`classify()` therefore works on the roots, and is a post-processing function
rather than an output, since the answer is discrete.

### Doubling the stabilizer, as p. 619 predicts

p. 619: "doubling the area would improve the longitudinal flying qualities by
moving the example helicopter from a region of pure divergences to one of
unstable oscillations".

Adding the horizontal stabilizer's own Table 9.16 contributions a second time —
`dM/dzdot` by -219 and `dM/dxdot` by +42 — and re-expanding the determinant
does exactly that: `classify` returns `unstable divergence` at 18 ft² and
`unstable oscillation` at 36. The prediction is reproduced from the derivative
tables rather than read off the figure.

## Validated anchors (G7, modal approximations)

p. 624 works three approximations on the example helicopter with a 54 ft²
horizontal stabilizer -- the Table 9.16 stabilizer column counted three times.

| p. 624 | model | book |
|---|---|---|
| full 3 degrees of freedom | .3667 rad/s, 17.13 s | .365, 17.2 |
| full 2 degrees of freedom | .3420 rad/s, 18.37 s | .342, 18.4 |
| approximate 2 degrees of freedom | .3556 rad/s, 17.67 s | .356, 17.6 |

All three within 1 %, and the three periods span under 7 %, which is p. 624's
point: the simplifications cost little on frequency.

### "Natural frequency" means the damped frequency

p. 624's column is `2 pi / P`, the **imaginary part** of the root, not the
modulus. The distinction is invisible on the phugoid reduction, whose real part
is .035, and unmissable on the full quartic, whose real part is .226: there the
modulus is **.431** against a printed **.365**. Reading it as the modulus makes
the full three-degree-of-freedom row look 18 % wrong when it is 0.5 % right.

### The short-period reduction is the better one

p. 625's quadratic matches the determinant term for term -- unlike the phugoid
cubic, every sign is printed correctly. It is also the closer approximation:
for the 18 ft² aircraft it gives `-2.549` and `+1.037` against the full
quartic's `-2.564` and `+.9867`, inside 1 % and 6 %.

The phugoid reduction on the same aircraft returns a **complex pair where the
full system has four real roots** — a change of character, not just of damping.
p. 624 calls it "sacrificed reasonableness for the phugoid damping"; on this
configuration it is worse than that.

### The third approximation was already implemented

p. 624's one-degree-of-freedom limit drops `I_yy` on Hohenemser's argument and
reduces to `omega = sqrt(-g (dM/dxdot)/(dM/dq))`, which it says "is the same
equation as that derived for hover". `HohenemserPeriodComp`, written for
pp. 600-601, produces .3556 for the 54 ft² configuration without modification.
No new component was needed for that row, and a test drives it through the
hover component to make the identity explicit.

## Validated anchors (G7, handling qualities in forward flight)

### Figure 9.21, the Boeing-Vertol parameter

    (dZ/dzdot)/(G.W./g) x (dM/dq)/I_yy - V_bar (dM/dzdot)/I_yy

| area | parameter | longitudinal oscillation |
|---|---|---|
| 18 ft² | -2.649 | none — four real roots |
| 36 ft² | -1.489 | P = 41.6 s, doubling in 1.6 s |
| 54 ft² | -0.325 | P = 17.1 s, doubling in 3.1 s |
| 72 ft² | +0.843 | P = 17.0 s, doubling in 46.6 s |
| 90 ft² | +2.015 | P = 7.5 s, halving in 0.7 s |

**It is the short-period stiffness in disguise.** Expand p. 625's short-period
quadratic and its constant term is
`[(dZ/dzdot)(dM/dq) - (dZ/dq + m V)(dM/dzdot)] / [I_yy (m - dZ/dzddot)]`. Drop
`dZ/dq` and `dZ/dzddot` — -217 against an `m V` of 120,555, and zero — and the
two coincide: -2.6487 against the exact -2.6427, **0.2 %**, and a test shows
the gap is exactly the `dZ/dq` term.

So the Boeing-Vertol criterion is that the **square of the short-period natural
frequency** exceed about 1 — that the short period be a real oscillation rather
than a divergence. The sign change between 54 and 72 ft² is precisely where
that happens.

Figure 9.21 prints the last numerator as `dM/dz`, without the dot. It is
`dM/dzdot`: the figure is set in a typewriter face throughout, and `dM/dz`
would make the term `1/s^3` where the first product is `1/s^2`.

The computed values sit about 1.2 below the marks on Figure 9.21. The
stabilizer contributions are scaled linearly with area here and the aircraft is
not re-trimmed at each size, which Prouty would have done. The trend, the
ordering and the zero crossing all hold.

### Table 9.21, p. 622

| Period | Visual flight | Instrument flight |
|---|---|---|
| < 5 s | ½ amplitude in 2 cycles | ½ amplitude in 1 cycle |
| 5-10 s | at least lightly damped | ½ amplitude in 2 cycles |
| 10-20 s | not double in 10 s | at least lightly damped |
| > 20 s | no requirement | not double in 20 s |

Discontinuous thresholds keyed on a period band, so this is a classifier and
not a gradient constraint — the same treatment `classify` gets on the stability
map. `mil_h_8501a` returns the rule that fired alongside the verdict.

**p. 622's sentence, reproduced from the derivative tables.** It says the
72 ft² stabilizer "would allow the aircraft to satisfy the visual flight
requirement, but a somewhat larger tail ... would be needed for instrument
flight". The 72 ft² configuration oscillates at 17.0 seconds with amplitude
doubling in 46.6, which clears the 10-20 second band's visual rule of *not
double in 10 sec* and fails its instrument rule of *at least lightly damped*.
At 90 ft² the mode is damped and both rules pass; at 54 ft² both fail.

### A caution the classifier surfaces rather than patches

Table 9.21 imposes **no** visual requirement above 20 seconds, on the argument
that "the time is so long that the pilot instinctively corrects for any
instability". Applied literally that lets anything through, and the 36 ft²
configuration exploits it: period 41.6 seconds, amplitude doubling in **1.6
seconds**. That is a divergence wearing a very slow oscillation, not the slow
wallow the clause was written for.

`mil_h_8501a` returns `no requirement` as the deciding rule there, so a caller
can see it. Silently tightening the standard would have hidden a real feature
of MIL-H-8501A.

## Validated anchors (G8, lateral-directional in forward flight)

`LateralMatrixFFComp` feeding `PolyDeterminantComp(n=3, degree=2,
n_zero_roots=2)`, from the Table 9.16 totals.

| | model | book (p. 628) |
|---|---|---|
| characteristic equation | s⁴ + 8.4602 s³ + 17.677 s² + 45.506 s + 2.2544 | s⁴ + 8.460 s³ + 17.68 s² + 45.54 s + 2.2548 |
| roll convergence | −6.8416 | −6.842 |
| Dutch roll | −.7841 ± 2.4309i | −.7841 ± 2.4317i |
| spiral | −.0505 | −.05058 |

Every coefficient and every root inside the last printed digit.

**The lateral-directional motion is stable** where the longitudinal is a pure
divergence. All four roots are in the left half plane: a roll convergence
halving in 0.10 s, a well damped Dutch roll of 2.58 s halving in 0.88 s, and a
spiral halving in **13.7 s**, which is p. 633's "about 14 seconds".

### The gravity term is +G.W., and the kinematic term mirrors the X row

Table 9.20 prints `-3964 s + 20000` in the Y row against `3927 s - 20000` in
the X row. Both signs follow from the axes, as in hover: a right-wing-down
`Phi` puts `+W sin(Phi)` along body `y` while a nose-up `Theta` puts
`-W sin(Theta)` along body `x`.

More useful, the two kinematic terms carry the **same** 1,990 with opposite
signs — `(dY/dp + m V Theta_bar)` against `(dX/dq - m V Theta_bar)` — so the
two entries check each other, and both need `Theta_bar = -.0165 rad`. That is
the third independent confirmation of a trim attitude Chapter 9 never prints.

`(dY/dr - m V)` is the centrifugal term and swamps the aerodynamics exactly as
its longitudinal counterpart does: −120,555 against +1,397, so `dY/dr` is 1 %
of the entry.

### No product of inertia

Table 9.19 carries no `I_xz`. Roll and yaw couple only through `dR/dr` and
`dN/dp`, which for the example helicopter are **6,912 and 6,909** — equal to
0.04 %, both being the tail rotor and the vertical stabilizer's side force
carried on `h` and `-l` respectively.

## Validated anchors (G8, the Dutch roll ladder)

Three levels of approximation, pp. 628-632.

| equation | roots | book |
|---|---|---|
| full quartic, p. 628 | −.7841 ± 2.4309i | −.7841 ± 2.4317i |
| reduced cubic, p. 630 | −.7834 ± 2.3813i | — |
| Bairstow quadratic, p. 631 | −.7821 ± 2.3766i | −.7823 ± 2.3826i |
| yaw only, p. 632 | −.7702 ± 2.4700i | −.7702 ± 2.4662i |

Every real part within 2 % of the exact one and every period between 2.5 and
2.7 seconds, which is p. 632's "approximately 2.5 seconds, fairly typical of
both helicopters and airplanes of all sizes".

### p. 631's cubic is printed correctly

Expanded from the reduced determinant of p. 630 term by term, every sign
matches. Unlike the phugoid reduction of p. 623 (entry C9-19), nothing here is
misprinted, and a test compares the determinant against the printed
coefficients to 1e-10.

### The straight flight path assumption removes the spiral

p. 630 constrains the centre of gravity to a straight path, which strips the Y
equation to `y_ddot + V_bar r = 0`. The resulting cubic keeps the roll
convergence (−6.72 against −6.84) and the Dutch roll (−.7834 ± 2.3813i against
−.7841 ± 2.4309i) and **loses the spiral** — which is exactly right, since the
spiral is the one lateral mode in which the flight path curves.

### The Bairstow step is exact algebra

For a lightly damped cubic, `s^2 = -c0/c2` turns `c3 s^3 + c2 s^2 + c1 s + c0`
into `c2 s^2 + (c1 - c3 c0/c2) s + c0`. Substituting `c2 = -I_zz (dR/dp)` and
expanding, the two `V_bar I_xx (dN/dydot)` terms **cancel identically** and
what is left is precisely p. 631's printed bracket. A test performs the
substitution on the cubic's own coefficients and matches the component to
1e-10, so the approximation lives entirely in the two assumptions and not in
the manipulation.

### "Much less than" is generous

p. 631's first assumption is `(dN/dr)/I_zz << (dR/dp)/I_xx`. For the example
helicopter those are **−1.54 and −6.75**, a ratio of **0.23** — not a
negligible one. It survives because the Bairstow step is what carries the
frequency and the assumption only perturbs the damping: the cubic and the
Bairstow quadratic differ by 0.2 % on the real part.

### The simplest form keeps two derivatives

p. 632 drops everything but yaw damping and directional stability. A test
doubles `dR/dydot`, `dR/dp`, `dR/dr`, `dN/dp` and `I_xx` in turn and confirms
the simple quadratic does not move, then doubles `dN/dr` and `dN/dydot` and
confirms it does. It is the worst of the three on frequency and the best on
nothing, but it costs two numbers.

## Validated anchors (G8, the lateral-directional stability map)

Figure 9.23, p. 634, plots the aircraft in the plane of its directional
stability `dN/dydot` against its dihedral effect `dR/dydot`, bounded by Routh's
discriminant below and by `E`, the constant term, above. p. 634 names the one
that matters: "the critical boundary is that associated with the constant term
of the characteristic equation, `E`, being zero".

### `E` derived rather than taken on trust

p. 633 prints

    E = (g / (I_xx I_zz)) [(dR/dydot)(dN/dr) - (dN/dydot)(dR/dr)]

without derivation. It falls out of the determinant, exactly as its
longitudinal counterpart does: the only entry of the lateral matrix carrying a
constant term is `+G.W.` in the Y row, so two of the six Leibniz permutations
reach the `s^2` coefficient of the degree-six determinant — one odd, one even —
and they give precisely that expression. **2.2544** against the **2.2548**
p. 628 prints as the quartic's constant term.

### p. 633's table of four derivatives

| derivative | name | sign | value | product |
|---|---|---|---|---|
| `dR/dydot` | dihedral effect | − | −382 | 20.59e6 |
| `dN/dr` | yaw damping | − | −53,913 | |
| `dN/dydot` | directional stability | + | 1,207 | 8.34e6 |
| `dR/dr` | roll due to yaw rate | + | 6,912 | |
| | | | difference | **12.25e6** |

All three products reproduce. **Both are positive because each pair shares a
sign**, and the spiral is stable because the first beats the second. The
boundary sits at `dN/dydot = 2,980` and the aircraft is at 1,207, with a slope
`dN/dr / dR/dr = -7.800` — Figure 9.23 draws the line through (−150, 1170),
and the model puts it at 1,170.

### The two boundaries are crossed at opposite ends

A test walks `dN/dydot` down from 1,207. `E` stays positive and even grows,
while Routh's discriminant crosses zero near **88**, which is where Figure 9.23
draws its shallow lower line at this dihedral effect. Below that the mode that
goes is the Dutch roll, not the spiral. Raising `dN/dydot` past 2,980 does the
opposite. `classify_lateral` names which one, on the roots, since the answer is
discrete.

### The two ways to fix a spiral dive, both reproduced

p. 635 names them. Raising `dN/dr` in magnitude raises the boundary — the yaw
damper or SAS — and doubling it more than doubles the margin in a test.

Lowering `dN/dydot` moves the aircraft down the map, which is the **Bell 212's
vertical destabilizer ahead of the centre of gravity** of Figure 9.24,
deliberately *reducing* directional stability. A test takes a configuration
diverging in spiral, halves `dN/dydot` and nothing else, and `classify_lateral`
returns to `stable`.

## Validated anchors (G5, control response in hover)

`HoverControlColumnComp` with `HoverLongTwoDofMatrixComp` and the
`TransferFunctionGroup` written for G0 and not used until now.

| | model | book (pp. 606-608) |
|---|---|---|
| denominator | s³ + .7245 s² + .1151 | the p. 598 cubic |
| attitude gain `(1/I_yy)(dM/dB1)` | −6.764 | −6.78 |
| step response at t = 2 s | −6.862 | −6.870 |
| step response at t = 8 s | +8.053 | +8.072 |

### The numerator collapses to one term, for the fourth time

p. 606 notes that "some terms in the numerator cancel themselves out just as
they did in the derivation of the characteristic equation". Substituting the
control column into the attitude column gives

    (dM/dB1)(m s - dX/dxdot) + (dX/dB1)(dM/dxdot)

and the constant part vanishes because Table 9.2 builds `dM/dB1` and
`dM/dxdot` from the same `dM/da1s`, and `dX/dB1` and `dX/dxdot` from the same
`-rho A_b (Omega R)^2 (dCH/sigma/da1s)`. What survives is

    Theta(s)/B1(s) = (1/I_yy)(dM/dB1) s / [cubic]

This is the **fourth** appearance of the same structural cancellation in the
chapter, after the missing `s` term of the hover cubic, the
`(dM/dq)(dX/dxdot) - (dM/dxdot)(dX/dq)` of p. 597, and the exactness of the
Hohenemser flapping form. Nothing in the implementation enforces it; on the
rounded Table 9.4 values the residual constant term is **5e-5** of the
surviving one, and a test pins that ratio.

### The Heaviside expansion, checked two ways

p. 607 offers the expansion as a way "to make simple checks of computer
results", which is exactly the role `heaviside_step_response` plays here. It
reproduces p. 608's closed form

    q/B1 = 5.78 exp(-.874 t) - 6.85 exp(.075 t) sin(20.34 t + 57.54)

to within 1 % at every point Figure 9.12 plots, the residual being the gap
between the book's rounded gain of −6.78 and the −6.764 the recovered
derivatives give. A second test compares it against a numerical inverse
transform of the same polynomials and agrees to 1e-6, so the expansion itself
is right independently of the book's algebra.

The response starts from rest — `N(0) = 0`, so there is no steady term — builds
to about −7 deg/sec by 3 seconds and reverses, which is the first quarter cycle
of the 17.7 second unstable oscillation.

### Recovered control derivatives

Chapter 9 does not print them in hover. From Tables 9.1 and 9.2 with
`da1s/dB1 = -1/(1 + kappa^2) = -.99138`:

    dX/dB1 = -rho A_b (Omega R)^2 (dCH/sigma/da1s)(da1s/dB1)  = +9,516 lb/rad
    dM/dB1 = (dM/da1s)(da1s/dB1) - (dX/dB1) h_M               = -270,578 ft lb/rad

and `dM/dB1 / I_yy = -6.764` against the **-6.78** p. 607 uses, which is the
confirmation that the recovery is right.

## Validated anchors (G6, Table 9.18)

| axis | t (s) | min response, visual | instrument | max rate | damping, visual | instrument |
|---|---|---|---|---|---|---|
| longitudinal | 1 | 45/R | 73/R | — | 8 I⁰·⁷ | 15 I⁰·⁷ |
| lateral | .5 | 27/R | 32/R | 20 deg/s | 18 I⁰·⁷ | 25 I⁰·⁷ |
| directional | 1 | 110/R | 110/R | — | 27 I⁰·⁷ ᵃ | 27 I⁰·⁷ |

with `R` the **cube root** of `G.W. + 1,000` — the radical in Table 9.18
carries an index of 3, which is 27.589 for the example helicopter's 20,000 lb.
Footnote a is the table's only one: the visual directional damping figure is
"not a requirement, only a preference", and the component carries it as
`is_preference_only`.

All twelve cells are checked individually.

### The displacement the table is compared with

p. 611 treats each moment equation as a single degree of freedom,
`(dM/dB1) B1 = I_yy Theta_ddot - (dM/dq) Theta_dot`, and integrates it to

    Theta/inch = 57.3 (CP/I)/(D/I) [t + (1/(D/I))(exp(-(D/I) t) - 1)]

p. 611 folds `t = 1` into its printed version; the component keeps `t` because
the lateral axis is specified at half a second. A test integrates the printed
ODE numerically with `solve_ivp` and matches the closed form to 1e-6, so the
integration is verified rather than transcribed.

For the example helicopter in pitch that is **7.32 deg per inch** against a
required 2.65, and every axis clears its response requirement several times
over.

### Where the example helicopter actually stands

| axis | damping/I | required/I | |
|---|---|---|---|
| longitudinal | 0.717 | 0.624 | passes by **15 %** |
| lateral | 5.83 | 1.94 | passes 3 to 1 |
| directional | 0.381 | 1.17 | **fails 1 to 3** |

Pitch is the tight axis — 28,659 against a required 24,977 — which is the same
conclusion the longitudinal stability map reaches from the other direction, and
the same 18 ft² stabilizer behind both.

**Yaw fails by a factor of three**: 13,326 against a required 40,948
ft lb/rad/sec. That is with the tail rotor's −17,797 working *against* a main
rotor term of the opposite sign (entry C9-15), and correcting that sign to make
both damping still leaves 22,268 against 40,948. p. 612 states the example
helicopter "would satisfy the instrument flight requirements", which is hard to
reconcile with the yaw axis on these numbers.

### One number read off a figure

The abscissa of Figure 9.13 is moment per **inch of stick**, and Chapter 9
nowhere prints the control linkage gearing that converts degrees of cyclic into
inches. `FIGURE_9_13_CONTROL_POWER` holds the three values read back off the
figure — 12,800, 7,500 and 63,000 ft lb/in — and they are the only place in
this chapter's implementation where a number comes from a figure rather than
from an equation. They are inputs with those defaults, not derived quantities,
and the class docstring says so.

## Validated anchors (Figure 9.14 and ControlResponseHoverGroup)

### Figure 9.14, p. 613

Not Table 9.18 recast. p. 613 offers it as "simplified approximations of the
boundaries that are generally accepted today as the result of several flight
test and simulator studies", references 9.10 and 9.11 — a different and tighter
source. Three rectangular acceptance boxes in the plane p. 612 arrives at:

| axis | steady rate, deg/s/in | time constant, s |
|---|---|---|
| pitch | 5 to 12.3 | up to 1.0 |
| roll | 9 to 20.5 | up to 0.5 |
| yaw, utility | 15 to 24 | up to 0.5 |
| yaw, armed | 30 to 50 | up to 0.24 |

Read off the figure, so good to about half a division. `figure_9_14_violations`
returns a **tuple** of labels rather than one verdict, because more than one can
apply at once — the armed yaw box rejects (20, 0.4) as both sluggish and too
slow, and the example helicopter in pitch fails on both axes too.

p. 613's case for the format is worth recording: "flight test data in the form
of time histories following step control inputs can yield the information
required to judge the flying qualities directly" — no derivative estimation at
all, just a steady rate and a time constant off a trace.

### p. 612's two limits, as independent checks

The displacement formula of p. 611 has two asymptotes p. 612 gives separately:

    damping -> 0        Cont.Pow./Inertia = 2 (Theta/Inch)/t^2     (s = at^2/2)
    damping -> infinity Cont.Pow./Inertia = (Theta/Inch)(Damping/Inertia)

Driving `dM/dq` to -1e-2 and to -4e8 in the group recovers `CP/I = 0.32` from
both, to 0.1 %. Two checks on the integration that do not come from the
integration.

### The group

Four subsystems and two conversions: the p. 598 matrix, the p. 606 control
column, the `TransferFunctionGroup` from G0, and Table 9.18.

**Two sign conversions are done in the open.** Table 9.18 and Figure 9.13 are
written in magnitudes while the derivatives feeding them are negative —
`dM/dq = -28,659`, `dM/dB1 = -270,578`. Two `ExecComp` lines do it, so the
negation appears in `list_inputs` rather than hiding inside a component.

**One number comes from a figure.** Control power per inch needs degrees of
cyclic per inch of stick, which Chapter 9 prints nowhere.
`deg_per_inch = 2.71` is what puts the example helicopter at 0.32 on
Figure 9.13's pitch abscissa. Everything else in the group comes from an
equation.

### It passes MIL-H-8501A and fails Figure 9.14

| | value | requirement | |
|---|---|---|---|
| one-second displacement | 7.32 deg/in | 2.65 | passes |
| damping | 28,659 | 24,977 | passes by 15 % |
| steady rate | **25.6 deg/s/in** | 5 to 12.3 | **oversensitive** |
| time constant | **1.40 s** | up to 1.0 | **too slow** |

Both at once, and that is not a contradiction: it is why p. 613 prints the
second figure at all. The aircraft satisfies the 1961 specification and sits
outside the boundaries flight test had established by the time Prouty wrote.

Halving the gearing to 1.2 deg/in brings the steady rate to 11.3 and inside the
box on that axis, and leaves the time constant untouched at 1.40 — it depends
on damping alone. Fixing the pitch axis properly needs damping, not gearing,
which is the stabilizer again.

### The time history stays post-processing

`numerator_coeffs` and `denominator_coeffs` go to `heaviside_step_response` for
Figure 9.12, and `figure_9_14_violations` reads its verdict off `steady_rate`
and `time_constant`. Neither is a component. The convention was set in
Chapter 1 by `plot_blade_element`, whose docstring puts it plainly: a figure is
a side effect with no derivatives, so putting it in the model would place it in
the dependency graph and redraw it at every optimiser iteration.

## The other discriminant (G0 addition)

Chapter 9's Figure 9.15 is bounded by **two** discriminants and labels one.

| | condition | what crosses | boundary |
|---|---|---|---|
| Routh's discriminant | R.D. = 0 | a root pair crosses the imaginary axis | stable / unstable **oscillation** |
| polynomial discriminant | disc(p) = 0 | a root becomes double | complex pair / **two real roots** |

The first is a real part changing sign, the second an imaginary part
vanishing. They are different algebraic objects: R.D. comes out of the Routh
array, disc(p) is the resultant of `p` and `p'`.

p. 618 says the unlabelled right-hand boundary "was determined by finding
combinations of the two derivatives that made the roots of the characteristic
equation switch from complex to real". **That is disc(p) = 0, and it has a
closed form** — a root search where algebra will do.

### `PolynomialDiscriminantComp`

`disc(p) = (-1)^(n(n-1)/2) Res(p, p') / a_n`, the resultant taken as a
Sylvester determinant built from the coefficients alone. No roots, exact and
differentiable: the partials are cofactors of that determinant, and since each
coefficient occupies known positions in the Sylvester matrix the chain rule is
a sum over those positions.

The cofactors are taken by **explicit minors** rather than through
`det(S) inv(S).T`, which is quicker and singular precisely on the locus the
component exists to find. A test drives it to within 1e-7 of a quadruple root
and checks the partials still hold.

Validated against the textbook forms — `b^2 - 4ac` for the quadratic, the
five-term cubic formula — and against the definition
`a^(2n-2) prod (r_i - r_j)^2` for degrees 2 through 6.

### Walking the boundary across the stabilizer areas

| area | R.D.(4) | disc | roots |
|---|---|---|---|
| 18 ft² | -.3191 | **+67.92** | 4 real |
| 36 ft² | -.5345 | **-4.244** | 2 real + pair |
| 54 ft² | -.4735 | -6.464 | 2 real + pair |
| 72 ft² | -.0553 | -.5302 | 2 real + pair |
| 90 ft² | +.8025 | **+4.853** | 2 pairs |

`disc` changes sign between **18 and 36** square feet, which is exactly where
p. 619 says doubling the stabilizer moves the aircraft "from a region of pure
divergences to one of unstable oscillations". Routh's discriminant changes sign
between **72 and 90**, the other boundary. The two are independent, and both
are needed: all of 18, 36, 54 and 72 have `R.D. < 0`, and no Routh test
distinguishes the pure divergence at 18 from the unstable oscillation at 54.

### It fills the gap Table 9.17 leaves

Table 9.17 says *whether* the aircraft is unstable and nothing about the kind.
`quartic_root_character` closes that, still without root-finding: with

    P = 8 a c - 3 b^2
    D = 64 a^3 e - 16 a^2 c^2 + 16 a b^2 c - 16 a^2 b d - 3 b^4

a negative discriminant is two real roots and one complex pair; a positive one
is four real roots when `P` and `D` are both negative and two complex pairs
otherwise. It agrees with `describe_modes`' root count on all five
configurations.

**Positive is ambiguous and negative is not**: 18 ft² and 90 ft² both have
`disc > 0` and sit at opposite ends of the map, four real roots against two
complex pairs. That is why the sign alone does not classify and the companion
quantities are needed.

This also makes two of Figure 9.15's three boundaries analytic — `E = 0` and
`disc = 0` — leaving only `R.D. = 0` to be contoured on a sweep.

## All three map boundaries are closed form — a correction

These notes said twice that the oscillation/divergence boundary of Figure 9.15
had no closed form, and once that `R.D. = 0` would have to be contoured on a
sweep. Both were wrong. In the plane either map is drawn in:

| boundary | curve | why |
|---|---|---|
| `E = 0` | **straight line** through the origin | `E` is affine with no constant term |
| `R.D. = 0` | **conic** | `A` and `B` are constant, `C`, `D`, `E` affine |
| `disc = 0` | **sextic** | the quartic discriminant is degree six in the coefficients |

No sweep is needed for any of them. A sweep would only shade regions.

### The structure that makes it work

Evaluating the longitudinal determinant at three points in the
`(dM/dzdot, dM/dxdot)` plane:

    A = 1                                                        constant
    B = 1.544476                                                 constant
    C = .555011 - .00015806 dM/dxdot - .00484377 dM/dzdot         affine
    D = .017957 + .00079435 dM/dxdot - .00016844 dM/dzdot         affine
    E =           .00037197 dM/dxdot + .00006351 dM/dzdot         affine

affine to **3.5e-15**, and `A` and `B` do not move at all. So
`R.D. = BCD - AD^2 - B^2 E` is exactly quadratic. The lateral map has the
identical structure in `(dR/dydot, dN/dydot)`, affine to 3.8e-13.

Those five lines are also p. 618's printed characteristic equation with the two
derivatives left as variables, to every figure it prints.

### The derived conic against p. 618

| term | derived | p. 618 | ratio |
|---|---|---|---|
| constant | .0150705 | .015030 | 1.003 |
| `dM/dzdot` | -.000424165 | -.000425 | 0.998 |
| `dM/dxdot` | -.000239285 | -.000243 | 0.985 |
| `dM/dzdot²` | 1.23173e-6 | 1.24e-6 | 0.993 |
| cross | -5.63388e-6 | -5.55e-6 | 1.015 |
| `dM/dxdot²` | -8.24916e-7 | -8.19e-7 | 1.007 |

and `evaluate_conic` reproduces `RouthDiscriminantComp` to machine precision at
every point tested, on both maps.

### The safeguard, and where it bites

Two conditions, and they are not the same one.

`affine_coefficients` verifies **affinity** by sampling a fourth point off the
axes. That holds whenever each varied derivative appears once in the matrix, so
the determinant is linear in it — a weak condition.

`routh_conic` additionally requires **`B` constant**, and raises otherwise.
That is the condition that makes `R.D.` quadratic, and it is the stricter one.
Varying `dM/dq` shows the difference: the coefficients stay affine, because it
appears once, but `B` moves, so `BCD` is cubic and there is no conic at all. A
test drives both branches.

I had assumed affinity was the operative condition. It is not, and writing the
check found it.

## Presentation figures

Two functions in `plots.py`, post-processing as `plot_blade_element` set the
convention in Chapter 1. A test asserts the module defines no component at all.

### The whole plane costs four model evaluations

Because the characteristic coefficients are affine in the two derivatives a
map is drawn in, `affine_coefficients` fixes the expansion with three
evaluations and verifies it with a fourth. Everything after that — a
240 x 240 grid of quartics, their roots, Routh's discriminant, the region
codes — is arithmetic on arrays. A test counts the calls.

`roots_on_grid` gets the roots by **batched companion-matrix eigenvalues**,
where `np.roots` would need a Python loop over 57,600 points. `classify_grid`
then agrees with `classify` point for point, so the region shading and the
component-level verdict cannot drift apart.

Prouty's right-hand boundary, the one p. 618 found by a root search, is not
drawn separately: it **falls out of the shading**, which is the same search
done on a grid.

### The longitudinal map reproduces Figure 9.15's topology

Stable oscillations upper left, a band of unstable oscillations, divergences to
the right and again below the `E = 0` line in the lower left. The five
stabilizer marks climb up and to the left, which is p. 619's sentence made
visible: "doubling the area would improve the longitudinal flying qualities by
moving the example helicopter from a region of pure divergences to one of
unstable oscillations".

The drawn map shows a second branch of the Routh conic in the lower right.
It is real, not an artefact — Prouty's axes stop at 900 and do not reach it.

### How long the two response curves actually agree

p. 608 says the free and trunnion-mounted histories are "essentially identical
during the first quarter cycle". Measured:

| t | free | trunnions | gap |
|---|---|---|---|
| 1.0 s | -4.788 | -4.828 | 0.04 |
| 2.0 s | -6.843 | -7.187 | 0.34 |
| 3.0 s | -7.088 | -8.339 | 1.25 |
| 4.4 s | -5.046 | -9.036 | **3.99** |

The period is 17.7 seconds, so a true quarter cycle is 4.4 seconds, where the
two differ by **44 %** of the smaller. They hold within 5 % through two
seconds and part company steadily after. The claim is generous; the plot
carries both curves so the reader sees where translation takes over, which is
the useful content of Figure 9.12.

### The lateral map and the response box

`plot_stability_map` does both maps; the longitudinal and lateral wrappers
differ only in labels and orientation. Figure 9.23 names the same three region
codes after the modes they are — an unstable real root is a **spiral dive**, an
unstable pair the **Dutch roll** — and Prouty plots the dihedral effect
increasingly negative to the right, which `invert_first` matches so the two
figures can be laid side by side.

The drawn map reproduces Figure 9.23's topology: `E = 0` dashed, rising to the
left with the spiral dive above it; Routh's discriminant nearly horizontal
along the bottom with the Dutch roll below; the aircraft at (-382, 1,207) in
the stable region between them. Tests cross both boundaries from that point —
cutting `dN/dydot` to 50 gives the Dutch roll, raising it to 3,500 gives the
spiral dive — so the two instabilities sit at opposite ends of one plane.

p. 634 is worth keeping beside it: for a helicopter "the direct relationships
between these derivatives and easily changed geometric parameters are not so
straightforward", unlike the fin and dihedral of an aeroplane. The lateral map
**illustrates**; it does not size. The longitudinal one does both.

`plot_response_box` needs no grid at all — its boundaries are the Figure 9.14
rectangles and the verdict is `figure_9_14_violations`. A test asserts the
function contains no `meshgrid` and no `classify_grid`, since reaching for the
map machinery here would be the natural mistake. For the yaw axis both the
utility and armed boxes are drawn so the tighter one is visible.

On the example helicopter it prints what the numbers already said: the point
sits at 25.6 deg/sec/in and a 1.40 second time constant, outside the pitch box
on both axes, annotated *oversensitive, takes too long to respond*.

## Validated anchors (Figure 9.18, load factor)

Every other analysis in Chapter 9 is at one g. p. 622 says why that is not
enough: "a helicopter that is stable in level flight will probably be unstable
at some higher load factor at the same speed ... the higher the thrust, the
stronger the destabilizing moment due to nose-up flapping. The airframe, on
the other hand, maintains a constant stabilizing influence."

So the derivative splits and only one half moves:

    dM/dzdot(n) = n (rotor contribution) + (airframe contribution)

| | model | Figure 9.18 |
|---|---|---|
| `dM/dzdot` at 1 g | -214 | -214 |
| load factor at neutral | 1.820 | 1.82 |

For the 90 ft² stabilizer -- the one size p. 622 says "would be enough to
provide a margin of positive angle-of-attack stability in level flight at 115
knots" -- the margin is **gone at 1.82 g**, a turn at 56 degrees of bank.

### The total checks, the split does not

Summing Table 9.16's `dM/dzdot` columns with the stabilizer counted five times
gives `495 - 1,095 + 374 = -226` against the figure's **-214**, agreeing to
5 %.

The split cannot be recovered. Table 9.8's three terms are the hub spring
(333), the `h_M` arm (45) and the `l_M` arm (131); taking the last two as the
thrust-dependent pair gives **176** where the figure needs **261**. Chapter 9
never prints it, so `FIGURE_9_18` holds the values read back off the line and
they are inputs, not derived quantities. A test records the gap so they are not
later mistaken for one.

This is the second place in the chapter where a number comes from a figure
rather than an equation, the first being the control gearing of Figure 9.13.

### Constrain the limit, not the neutral point

`load_factor_at_neutral` is `-airframe/rotor` and runs to infinity as the rotor
contribution vanishes, which makes it a poor thing to put in a gradient.
`dM_dzdot_at_limit` evaluates the derivative at whatever limit load factor the
aircraft is designed to, is finite everywhere, and asks the question a designer
has: *is it still stable at 2 g?* For the example helicopter at 2 g it is
**+47**, so no.

A rotor contribution that is not destabilising means stability never vanishes;
the component warns and returns zero rather than a negative or infinite load
factor. Setting it to zero recovers p. 622's contrast with the aeroplane, whose
angle-of-attack stability is "nearly invariant" -- a horizontal line.

---

## Note — thrust damping from Chapter 2

`thrust_damping='external'` (on `BasicRotorDerivativesHoverComp` and
`HoverDerivativesGroup`) replaces the Table 9.1 dCT/σ/dλ by the Chapter 2 thrust
damping of p. 102 (`prouty.vertical.ThrustDampingComp`). Default `'table'`,
nothing changes for this chapter. See C2-7 in `docs/validation_vertical.md`.
