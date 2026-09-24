# Validation notes

Departures from the printed text, and disagreements found inside it. Each
entry says what was measured, what was decided, and where the decision lives
in the code. Nothing here is a tolerance that was widened.

Anchors that DO reproduce are not listed; they are asserted in the test suite.

---

## 1. Book misprints

**p. 171 — coning, gravity term.** Printed `g R^2`; the same equation on
p. 213 has `g R`, which is what dimensional consistency requires. Coded `g R`.
*(ConingComp)*

**p. 209 — induced velocity distribution.** Printed `v_L = v_i(1 + (r/R) sin psi)`;
p. 165, 213 and 225 all write `cos psi`. Coded `cos`, and not by majority: with
psi = 0 over the tail, `cos` puts the extra downwash at the rear of the disc,
which is where the skewed wake leaves it. The `sin` form would put maximum
downwash on the advancing side, which no wake does. *(PerpVelComp)*

**Table 3.1, p. 191 — collective for delta_3 != 0.** The tabulated theta_0 of
6.27 and 6.22 deg cannot be reconciled with the flapping printed beside them.
Feeding the book's own a_1s and b_1s back into its longitudinal flapping
equation demands theta_0 + a_0 tan(d3) = 6.62 and 6.30 deg; the model gives
6.61 and 6.29, the table prints 6.82 and 5.67. The flapping is reproduced to
0.02 deg in all three columns. The collective row is a report, not a result.
*(TailRotorTrimComp)*

**Table 3.3, p. 196 — autorotation column.** H_M = 690 lb and hp_M = -40 hp
cannot both hold. Forcing H_M = 690 reproduces the tabulated attitude exactly
but gives +127 hp; solving honestly gives -40 hp and H_M near 200 lb.
Back-solving c_d, the torque demands 0.00564 and the H-force 0.01180 — a factor
of two. The descent rate is not affected: -1792 ft/min against -1803 tabulated,
by two independent methods. Never use that column's H_M or attitude as an
anchor. *(test_autorotation_column_of_table_33_is_inconsistent)*

**Table 3.3, p. 196 — climb thrust.** p. 194 extends the numerator of
alpha_TPP by G.W. sin(gamma) but does not restate T, and the climb column is
consistent with T left in its level flight form: 21,297 lb without the climb
term against 21,290 tabulated, where the shared numerator gives 21,502. The
model keeps the shared numerator, which is the physically correct one, and
runs 1 % high against that column. Level flight and autorotation agree to
0.01 % and 0.09 %. *(TppAngleComp)*

---

## 2. Equations replaced after measurement

**p. 183 — compressibility torque integral.** Evaluating the printed double
integral reproduces Figure 3.43 at mu = 0 within 15 % but falls a factor 2 to 4
short for every mu > 0, and reverses the trend: the integral makes the penalty
decrease with mu at fixed M_ratio while the figure makes it increase by 2.7
between mu = 0 and 0.3. Figure 3.43 is what Table 3.5 and the charts were built
on, so the figure wins and the integral is kept in `scripts/` as a documented
disagreement. The printed integral also carries no factor 1/2 although the
elemental torque does; restoring it would double the disagreement, so the
figure appears to have been computed from the equation as printed.
*(CompressibilityTorqueComp)*

**p. 212 — spanwise skin friction: the exact form against its small-angle
version.** Printed as `(U_B^2/2) c_f (mu/U_T) cos^2(psi)`, which carries 1/U_T
and is singular on the reverse flow boundary where U_B^2 does not vanish to
save it. Rederiving from the skin friction of p. 211 — the force is
(rho/2) c dr U_TR^2 c_f along the in-plane velocity, its spanwise component
follows by multiplying by U_R/U_TR, its projection onto the flight direction by
cos(psi), and U_R = mu cos(psi) — gives a bracket term of
c_f (mu U_TR/U_B^2) cos^2(psi), that is (1/2) c_f mu U_TR cos^2(psi).

The two agree if and only if U_T U_TR = U_B^2, which holds to first order when
U_P and U_R are small against U_T. So this is not a choice between two
plausible forms but the exact derivation against its own small-angle version,
and the approximation becomes singular exactly where it stops being valid. The
ratio U_T U_TR / U_B^2 is 1.00 at the advancing tip, 1.17 at mid-span forward,
3.2 at the root.

Measured on the example helicopter at mu = 0.3:

| form | C_H/sigma |
|---|---|
| printed, U_B^2/U_T | -0.0452 |
| exact, U_TR | +0.0010 |
| closed form | +0.0016 |

Thrust, torque and both moments are identical between the two, so the whole
discrepancy is this one term. Default is the exact form; `spanwise_form='book'`
is kept for comparison. The term is small when regular — +0.00008, 3 % of C_H
at the chart condition — so it is not the cause of the chart 3 disagreement.

**p. 211-212 verified term by term.** After chart 3 resisted seven measured
eliminations, the H-force equation was checked against the text rather than
probed further. Three checks, three passes:

  * *chordwise skin friction.* p. 211 gives
    c_c,SF = c_f U_T sqrt(U_T^2 + U_R^2) / U_B^2. ChordForceComp computes
    `cc0_UB2 = c_dp UT UB + c_f UT UTR`, which is that form times U_B^2.
  * *no r/R factor on C_H.* p. 212 writes the torque loading with (r/R) and the
    H-force loading without it. Getting that wrong gives a factor of almost
    exactly two, since the integral of r over the blade is half the integral of
    1 — the tempting explanation, and the code already has it right.
  * *integration limits.* p. 212: "no root cutout or tip loss is applied to the
    portion of the chordwise force produced by pressure drag and skin
    friction." c_c,0 integrates over 0 to 1, c_c,ind over x_0 to B; conforms.

So the H-force equation and its integration are faithful to the text. The
chart 3 disagreement is upstream, or in what the chart plots.
---

## 3. Guards the text does not mention

These are places where a printed formula is fine as physics and unusable as
code, because the numerical method feeds it values the derivation never
contemplated.

**p. 221 — dynamic stall delay, unbounded.** Reaches -202 deg near the reverse
flow boundary, where alpha, sweep and pitch rate are all extreme at once —
p. 221 says so itself. On the lift side that is harmless because the p. 221
bounds clip c_l afterwards; on the drag side it is not, since the delay shifts
the drag divergence angle and an alpha_D of -116 deg turns
`(alpha - alpha_D)^2.54` into c_d = 110. Prouty bounds c_l and says nothing
about c_d, which works with tabulated data because the table saturates and
does not with the Chapter 6 equations. Saturated at 10 deg; between 5 and 30
deg the integrated coefficients move by less than 0.1 %.
*(StallDelayComp.max_delay)*

**p. 224 — unsteady lift, pure oscillation assumption.** The derivation assumes
delta_alpha and theta_dot are pure oscillations at rotor frequency. Near the
reverse flow boundary alpha does not oscillate, it swings between neighbouring
stations: median |d(alpha)/d(psi)| there is 0.34 against 0.046 elsewhere.
Three guards were added — `max_k`, `mean_form`, `max_rate` — and none fixes the
outcome, because the assumption itself fails. The radial decomposition of the
thrust increment settles it:

| region | contribution to C_T/sigma |
|---|---|
| r/R < 0.3 | -0.002766 |
| 0.3 - 0.7 | -0.000483 |
| r/R > 0.7 | **+0.000503** |

Outboard of 0.7, where Figure 3.61 shows the effect and where the blade
actually lifts, the correction increases thrust exactly as p. 225 says. The
sign is reversed globally only by the root region, which carries 0.9 % of the
thrust and all of the increment. `unsteady` defaults to False; the coherent fix
is to restrict the correction to the domain where its founding assumption
holds. *(UnsteadyLiftComp)*

**p. 214 / Chapter 6 — solver states are not physical values.** The numerical
method carries M and alpha as Newton states, so an overshooting step hands the
airfoil model arguments it was never fitted for: `sqrt(1 - M^2)` at M > 1,
`M**7.15` at M < 0, `x**K2` overflowing to inf. One NaN in a state poisons the
whole residual vector, and the solve is lost silently — residuals read zero on
absurd coefficients. Chapter 6 now clips M to [0, 0.99] and the stall excess to
200 deg, and returns a finite number whatever it is handed. This is a change to
validated Chapter 6 code, made deliberately: a model whose inputs are solver
states must always answer with a number.
*(LiftModelCoefsComp, DragModelCoefsComp, LiftCoefComp, DragCoefHoverComp)*

**p. 218 / 221 — negative stall angle.** With an external stall angle,
`alpha_L = alpha_L_static sec(Lambda) + delta_alpha_stall` can go negative — a
saturated delay of -10 deg against a static 3.4 deg at high Mach is enough.
LiftCoefComp applies its post-stall decrement oddly about alpha = 0 through
sgn(alpha), so a negative alpha_L makes c_l jump by twice the decrement as the
angle changes sign. Floored at zero, smoothly. *(StallAngleComp)*

---

## 4. Disagreements between chapters

**Chapter 3 charts against the Chapter 6 drag model.** The isolated rotor
charts of p. 254-271 behave as a constant c_d near 0.010, insensitive to Mach
drag rise. The Chapter 6 fit of p. 432,
`c_d = c_d,incomp + 0.00066 (alpha - 17 + 23.4 M)^2.54`, collapses the drag
divergence angle with Mach — 7.6 deg at M = 0.4, 3.6 at 0.573, 0.6 at 0.7 — and
the charts are drawn at M_1,90 = 0.7, so the advancing tip sits permanently
near divergence.

Measured at mu = 0.10, C_T/sigma = 0.081, against the chart of p. 254:

| theta_0 | C_Q/sigma model | chart | induced part | profile part |
|---|---|---|---|---|
| 12 deg | 0.00460 | 0.0045 | 0.00313 | 0.00147 |
| 16 deg | 0.00911 | 0.0072 | 0.00697 | 0.00214 |
| 20 deg | 0.01438 | 0.0108 | 0.01094 | 0.00344 |

The induced part matches; the profile part triples where the chart keeps it
near 0.00129. It is entirely outboard: r/R > 0.7 carries 0.00312 of the 0.00344
at theta_0 = 20, while the reverse flow region carries 3 % despite its c_d of
2.05. Mean c_d over r/R > 0.7 goes from 0.0122 to 0.0307, peaking at 0.097 at
psi = 270 deg, r/R = 1, alpha = 10.1 deg, M = 0.579.

Tip relief was tested as the explanation and rejected: it removes 4 points of
the 32 %, and refining to 81 radial stations does not change that. The chart
ordinate X is untouched by any of this — it depends only on lambda' and
C_T/sigma — and is reproduced to +/- 15 % everywhere, which is the reading
accuracy of the printed curves.

Most likely the p. 432 fit is being used outside the range it was calibrated
on: it was fitted at M = 0.3, 0.5 and 0.7 at moderate angles, and here it is
evaluated at M = 0.7 with alpha = 10 deg. Documented rather than calibrated
away; calibrating c_d against the charts would mean tuning Chapter 6 on
Chapter 3.

Practical consequence: **C_Q/sigma is good to 3 % while the tip stays below
drag divergence, and reads high by up to 30 % once it does not.** C_T/sigma,
the flapping and the chart ordinate are unaffected.

**Table 3.5 case 7 against the model — a chart reading amplified.** The dive at
constant collective (p. 242-244) asks for lambda' at a given theta_0 and
C_T/sigma, which is a reading off the second set of charts, not a computation.
Both of our rotor models disagree with it by a factor of two:

| route | lambda' | C_T/sigma | C_Q/sigma |
|---|---|---|---|
| closed form, B = 0.97, x_0 = 0.15 | 0.0069 | 0.083 | — |
| G2, same B and x_0 | 0.0073 | 0.083 | 0.00081 |
| book | 0.015 | 0.083 | 0.0008 |

G1 and G2 agree with each other to 6 %, so this is not a weakness of the
closed-form equations. The closed form can be made to give 0.015 by taking
B = 0.94 and x_0 = 0.20, but p. 229 states B = 0.97 and mentions no root
cutout; fitting two parameters to one point is not a calibration.

What the disagreement costs, and what it does not. lambda' drives alpha_F,
hence L_F, hence f_M, and f_M is a small difference of large numbers: a factor
of two on lambda' gives 28.0 ft^2 against the book's 41.3, and the descent rate
comes out 1,259 ft/min against 1,608. But the two robust quantities are
reproduced — C_T/sigma to 0.4 % and C_Q/sigma to 1 %, and G2 finds by itself
that this rotor at theta_0 = 13 deg in a dive is very nearly autorotating,
which is the qualitative result of case 7. Validate FixedCollectiveTrimGroup on
C_T/sigma and C_Q/sigma, not on R/D.

**Table 3.5 case 6 against Table 3.3 — the book against itself.** Both compute
the same climb, 1,000 ft/min at mu = 0.3 for the example helicopter: Table 3.3
by the closed-form equations, case 6 (p. 240-242) by the charts.

| | our G1b | Table 3.3 | Table 3.5 case 6 |
|---|---|---|---|
| theta_0, deg | 18.67 | 18.6 | 19.9 |
| alpha_F, deg | -11.52 | -11.7 | -9.5 |
| L_F, lb | -1,222 | -1,228 | -999 |
| lambda' | -0.0603 | -0.0607 | -0.046 |
| hp_M | 1,754 | 1,760 | 2,109 |

The two book methods are 20 % apart on climb power. Our model sits on the
closed-form side to within 1 % on every quantity — partly by construction,
since c_d = 0.0100 was itself backed out of Table 3.3. Back-solving case 6's
C_Q/sigma = 0.0074 through the power identity demands c_d = 0.0164, so the gap
is stall drag that a constant c_d cannot represent: at C_T/sigma = 0.087 and
mu = 0.3 the chart method is already near its lower stall limit line.

Two smaller things from the same case. f_climb, the climb folded into an
equivalent flat plate area (p. 242 step dd), is reproduced independently at
61.1 ft^2 against 61.6 — so the climb bookkeeping is right and the disagreement
is entirely in the rotor solution. And step bb prints
C_T/sigma = 20,992/241,100 = .097, where the division gives .087; the .087 is
used everywhere else in the case, so it is a slip of one digit.

A second, smaller finding from the same case: Prouty's twist correction
theta_0 + 0.75(theta_1 + 5) equates two rotors at the 75 % station, not over
the integral. In the closed form, (13 deg, -10 deg twist) needs lambda' =
0.0069 while (9.25 deg, -5 deg) needs 0.0094 — 36 % apart for two collectives
that are supposed to be equivalent. The correction has no place in this package
anyway, since both rotor models take theta_1 directly, but it is worth knowing
if a chart-based result is ever compared against one computed here.

**Chart 3 — C_H/sigma comes out half, and nothing explains it yet.** The third
chart of each plate gives C_H/sigma - (C_T/sigma) tan(a_1s). The charts were
produced by the numerical method itself (p. 229), whose rigid rotor has
a_1s = 0 by construction, so the plotted quantity IS G2's C_H/sigma and the
comparison is direct; the composed form exists only so a reader with a real
articulated rotor can recover their own value, as Table 3.5 case 3 step g does.

Digitised from p. 255 at mu = 0.10, all five chart parameters matched
(mu, theta_1 = -5 deg, M_1,90 = 0.7, theta_0, C_T/sigma):

| theta_0 | C_T/sigma | G2 | chart |
|---|---|---|---|
| 12 | 0.106 | +0.00002 | -0.0007 |
| 16 | 0.109 | -0.00106 | -0.0022 |
| 20 | 0.101 | -0.00227 | -0.0042 |

G2 returns what the chart assigns to theta_0 - 4 deg. Four candidates were
raised and all four measured out:

  * *the spanwise friction term of p. 212.* The obvious suspect, since the
    rederived U_TR form was already known to sit 40 % below the closed form at
    mu = 0.3. Decomposition kills it: the term is +0.00008 at all three points,
    3 % of the total. Running the printed U_B^2/U_T form instead gives +0.086,
    +0.658 and +2.172 -- three orders of magnitude out and the wrong sign, even
    at mu = 0.10 where the reverse flow circle is small. That form is
    unusable in quadrature, whatever the grid.
  * *grid resolution.* C_H weighs the first sin(psi) harmonic where C_T weighs
    the mean, so it is the more demanding integral. From 12x15 to 48x81 --
    twenty times the elements, 2 s to 271 s -- C_H/sigma moves 2 %, from
    -0.00227 to -0.00222. The default grid is converged, which also settles
    that generating charts at 48x81 buys nothing.
  * *root cutout and tip loss.* Sweeping B from 0.90 to 1.00 and x_0 from 0 to
    0.30 moves C_H/sigma between -0.00172 and -0.00232, +/- 15 % where +90 %
    is needed -- and cutting the root moves it the WRONG way, from 0.53 of the
    chart down to 0.41. The best of seven combinations is B = 0.97, x_0 = 0,
    which is what p. 229 states.
  * *an imbalance between the force terms.* No: induced and profile carry the
    H-force almost equally (-0.00121 and -0.00113 at theta_0 = 20) and BOTH
    are half. A single mis-coded term would tilt the balance, not scale it.

What is established: the discrepancy is a factor of about 1.9, stable across
theta_0, acting equally on both force terms, hence upstream in the load field
rather than in the force equations. C_T/sigma is correct at the same point, so
the total load is right and its AZIMUTHAL DISTRIBUTION is not -- C_T weighs the
mean, C_H the first harmonic.

Two further candidates were then tested and also cleared:

  * *the inflow gradient itself.* A kappa multiplier was added to
    PerpVelComp on the v_i (r/R) cos(psi) term. From kappa = 0, which removes
    the gradient entirely, to kappa = 3, C_H/sigma moves 4 %. Six causes
    eliminated.
  * *the phase of that gradient*, i.e. the p. 209 sin against the cos used
    everywhere else. Switching to sin makes C_H/sigma change SIGN at
    theta_0 = 12 and 16 and moves further from the chart at 20.

A ninth candidate, the frame of reference, was raised and settled by
arithmetic. The chart plots C_H/sigma - (C_T/sigma) tan(a_1s); if the
flapping-feathering equivalence makes the rigid rotor's cyclic play the part of
a_1s, that term is not zero. At theta_0 = 20 and C_T/sigma = 0.101 it has the
right SIZE — (C_T/sigma) tan(A_1) = 0.0043 against a gap of 0.0020 — but no
combination lands on the chart value: -0.00225 raw, +0.00202 subtracting
tan(A_1), -0.00652 adding it, -0.00924 subtracting tan(B_1), against -0.0042.

WHERE TO RESUME. The strongest remaining clue is that the sign of the error
FLIPS between the two sources. G2 is 50 % LOW against chart 3, where lambda' is
negative, and 45 % HIGH against the case 9 measurement, where the shaft sits at
+5 deg and the flow is upward through the disc: backing C_H/sigma out of the
measured C_XR/sigma = -0.0058 gives -0.0038 against G2's -0.0055. No constant
scale factor does that; something that changes sign with alpha_TPP can. It also
means the two comparisons may not be of the same quantity — exactly the trap
chart 5 turned out to be. The first thing to re-examine is the assumption that
a_1s = 0 makes the rigid rotor's C_H/sigma equal to the chart's, which wants a
proper articulated-rotor b_1s from Chapter 4.

**The sin/cos question, settled by the charts.** Section 1 records the p. 209
misprint on the ground that a skewed wake puts its extra downwash aft.
The charts settle it better, because they were produced by Prouty's own
program: had that program used the printed sin, the charts would carry its
signature. Measured at mu = 0.10:

| theta_0 | | C_H/sigma | B_1 | A_1 |
|---|---|---|---|---|
| 12 | cos | -0.00002 | 2.33 | -2.57 |
| | sin | +0.00168 | 0.49 | -0.73 |
| | chart | -0.0007 | 1.79 | -0.84 |
| 20 | cos | -0.00225 | 3.96 | -2.42 |
| | sin | -0.00083 | 2.32 | -0.74 |
| | chart | -0.0042 | 2.95 | -2.37 |

sin makes C_H change sign and cuts B_1 by a factor of four. The charts carry
the cos signature. `gradient_phase` keeps both forms available; the default is
cos and the question is closed.

## 4b. The isolated rotor charts, chart by chart

Verified at mu = 0.10 with the chart section's own parameters (p. 229):
theta_1 = -5 deg, M_1,90 = 0.7 -- which fixes Omega R = 710 ft/s at this mu,
NOT a constant tip speed across the plates -- c/R = 0.079, B = 0.97, x_0 = 0.
gamma = 8 so that the (gamma/8) factor of chart 5 is unity.

Each plate carries five charts, all functions of theta_0 and C_T/sigma:

    1  f/A_b + (1/mu^4) sigma (C_T/sigma)^2
    2  C_Q/sigma, with the alpha_1,270 and delta C_Q/sigma_0 limit lines
    3  C_H/sigma - (C_T/sigma) tan(a_1s)
    4  lambda', with alpha_TPP = atan(lambda'/mu + sigma (C_T/sigma)/(2 mu^2))
       printed on it
    5  (B_1 + a_1s) above the axis, (gamma/8)(A_1 - b_1s) below

**Chart 1 -- reproduced.** The printed coefficient is 1/mu^4 each time: 10,000
at mu = 0.10, 123 at 0.30, 24.4 at 0.45, 16 at 0.50. Automatic tracking fails
on this plate -- thirteen crossing curves, labels in the field -- but the
column C_T/sigma = 0.11 yields exactly nine clusters, which is also the number
of curves still inside the frame there, so the assignment is forced. G2 gives
X = 13.09 against 13.8 at theta_0 = 16 and 22.22 against 22.9 at 20, both at
matching C_T/sigma. Within reading accuracy.

**Chart 2 -- limit lines at the right torque, displaced in thrust.** Too dense
to digitise (37 clusters at C_T/sigma = 0.10, which falls on a gridline), so
the check was made on the stall limit lines, which G2 computes and no other
chart tests. The C_Q/sigma at which delta C_Q/sigma_0 reaches 0.004 is right to
4 % -- 0.0161 against 0.0155 at theta_0 = 24, 0.0142 against 0.0135 at 20,
0.0100 against 0.0105 at 16 -- but it is reached at a lower C_T/sigma, by 0.005
at theta_0 = 16 growing to 0.015 at 24. The alpha_1,270 = 12 deg line is
displaced the same way. Both indicators move together, which is what they
should do, and the displacement is exactly the drag excess of section 4.

**Chart 3 -- factor of 1.9, unexplained.** See above.

**Chart 4 -- reproduced.** The cleanest plate to digitise. One assignment trap:
the column C_T/sigma = 0.02 gives fourteen clusters for thirteen curves, and
the top two are label artefacts. Fixed by physics rather than by guessing --
at theta_0 = 0 with -5 deg of twist the blade pitch is negative outboard, so
lambda' must be POSITIVE to make C_T/sigma = 0.02, and blade element theory
puts it at +0.061.

| theta_0 | lambda' G2 | chart |
|---|---|---|
| 0 | +0.0568 | +0.0599 |
| 8 | -0.0353 | -0.0326 |
| 12 | -0.0818 | -0.0822 |
| 16 | -0.1296 | -0.1351 |
| 20 | -0.1789 | -0.1894 |

Within 0.005 up to theta_0 = 16 on a scale running from +0.20 to -0.30. The
residual changes sign at theta_0 = 12: below it G2 wants slightly more inflow
than the chart, above it slightly less, which is the same lean as charts 1
and 2. Note that charts 1 and 4 carry the same information, since
X = -2 lambda' (C_T/sigma)/mu^3, so they corroborate each other rather than
testing independently -- but their joint agreement does establish that the
thrust-inflow relation of G2 is right.

**Chart 5 -- the lower half is NOT comparable.** The upper curves, B_1 + a_1s,
carry no gamma factor and are comparable: G2 reads 25 to 35 % high (2.22
against 1.79 at theta_0 = 12, 3.94 against 2.95 at 20). A real discrepancy,
not investigated.

The lower curves, (gamma/8)(A_1 - b_1s), are a different quantity from G2's
A_1 and the gamma/8 factor is the clue. G2's A_1 trims the pitching moment of
a rigid rotor with b_1s = 0, so it absorbs alone the moment due to coning --
and coning is proportional to gamma. Measured at theta_0 = 12:

| gamma | a_0 | A_1 | (gamma/8) A_1 |
|---|---|---|---|
| 4 | 2.53 | -2.20 | -1.10 |
| 8 | 5.22 | -2.57 | -2.57 |
| 12 | 7.91 | -2.94 | -4.40 |
| 16 | 10.60 | -3.30 | -6.61 |

The product varies by a factor of six where the chart's is invariant, and A_1
grows with gamma where the chart wants 1/gamma. For an articulated rotor the
combination A_1 - b_1s is dominated by the lateral flapping b_1s, which does
vary as 1/gamma; that is what makes the chart universal. The flapping-feathering
equivalence of p. 211 therefore transfers to the FORCES and to B_1 + a_1s, but
not to A_1 - b_1s taken with its normalisation. Comparing them is a category
error, and G2's A_1 being flat in theta_0 at fixed C_T/sigma -- -2.42, -2.41,
-2.40 for theta_0 = 12, 16, 20 -- is not evidence against it.

Summary: charts 1, 2 and 4 pass; chart 3 fails by a factor of 1.9 with no
cause found in seven attempts; chart 5 upper is 30 % high, chart 5 lower is not
a like-for-like comparison.

---

## 5. Numerical traps worth remembering

**`solve_subsystems=False` on a long explicit chain.** Newton then treats every
intermediate field as a state; its first step drove U_B negative, the stall
delay's square root returned NaN, and the solver "converged" in one iteration
on zero residuals and absurd coefficients. G1b tolerated it, G2 did not.
*(NumericalRotorGroup)*

**solve_subsystems=False, the second time.** TrimConditionsGroup ran with
subsystem solves off, which is correct for the closed-form rotor -- an
explicit chain -- and catastrophic for G2, which carries its own Newton for
A_1 and B_1. With them off the outer iteration never runs the inner solver,
so every residual is evaluated on an untrimmed disc. lambda' ran to -7.9 and
theta_0 to 220 degrees before the residuals went to NaN. One word, and the
same word as the first time.

**Bounds that make a residual infeasible.** C_T/sigma was bounded below at
zero; a rotor at theta_0 = 4 deg with -10 deg of twist genuinely produces
downward thrust, the disc returns -0.10, and the problem had no solution. The
symptom looked like a hard nonlinearity.

**A converged answer reported as failure.** At theta_0 = 24 deg the residuals
reach zero on the fortieth iteration exactly, and maxiter was 40.

**`np.clip` and complex step.** `np.clip` compares complex numbers
lexicographically, so a perturbation of a limit sitting on a cell boundary is
clipped away and `check_partials` reports a spurious zero derivative — blaming
correct analytic partials. *(RotorDiscGridComp._clip_real)*

**check_totals does not behave the same in every group.** On G2, whose Newton
solves A_1, B_1 and C_T/sigma together, `check_totals(method='fd',
form='central')` DOES reconverge the solver at each perturbation: its numbers
match manually retrimmed central differences to five decimals on all nine
pairs tested. The G1b trim groups do not behave that way and needed the manual
route. Verify which case you are in before trusting the built-in check --
running the manual comparison once costs a few minutes and settles it.

Analytic totals through G2's Newton are correct, checked against manual
retrimming on twelve pairs:

| | analytic | retrimmed | rel |
|---|---|---|---|
| dC_T/d(theta_0) | 0.73206 | 0.73207 | 6e-6 |
| dC_T/d(lambda') | 1.14951 | 1.14948 | 2e-5 |
| dC_H/d(lambda') | -0.01363 | -0.01365 | 1e-3 |

The residual is the truncation of the finite difference, not a derivative
error; C_H derivatives are two orders of magnitude smaller than the thrust
ones, so they carry proportionally less signal.

**Analytic totals degrade where the airfoil model is not smooth.** G2's totals
were checked at two operating points against central differences on fresh
Problem instances, all solves converged to 1e-13:

| condition | worst pair | relative |
|---|---|---|
| example helicopter, mu = 0.3, theta_1 = -10 | dC_H/d(lambda') | 1e-3 |
| H-34 tunnel case, mu = 0.3, theta_1 = -8 | dC_T/d(lambda') | 5e-2 |

The error is independent of the finite difference step from 1e-6 to 1e-3, so
it is neither noise nor truncation, and it is present in G2 alone -- not
caused by nesting G2 inside WindTunnelRotorGroup, whose closed-form path is
exact to 3e-9 on the same twenty pairs. What is left is the Chapter 6 airfoil
model, which carries several clips and np.where branches: the Mach ceiling,
the stall excess ceiling, the segment blends. An element sitting on one of
those boundaries has a one-sided analytic derivative while a central
difference straddles it. With 180 elements a handful on a boundary is enough
for a few per cent on an integrated quantity.

Practical consequence: G2's gradients are good to about 1e-3 in the well-
behaved interior and can degrade to a few per cent near the clips. Adequate
for a gradient-based optimiser, which needs a descent direction rather than an
exact Jacobian, but a check_totals on a new operating point is worth running
before trusting it there.

**An orphan input in radians is the worst kind.** Swapping the rotor in
WindTunnelRotorGroup left a_0 unconnected, because ClosedFormRotorGroup makes
its own coning and G2 takes it as an input. The default of an unconnected
OpenMDAO input is 1.0, and 1 radian is 57.3 degrees of coning -- absurd, but
inside the range where every component still returns a number. The Newton
converged to 1e-13, no warning was raised, and A_1 came out at -20.9 deg
instead of -2.6. The bug was found by a parameter sweep, not by the solver:
dA_1/da_0 had been measured at -0.34 deg/deg just before, which predicted the
error to within half a degree and pointed straight at the coning.

**Undeclared partials hide behind a None filter.** StallDelayComp did not
declare d/d(psi), and every validation script that collected
`abs error.forward` skipped it, because an undeclared pair reports None. Only
`check_partials` on the assembled chain caught it, at 0.149.

**Smoothing where nothing was discontinuous.** Windowing the quadrature weights
with a smoothstep cost 2.5 % on the zeroth moment; exact partial-cell
trapezoidal weights are both exact and differentiable. The blend is the right
reflex for a real discontinuity, not for one introduced by the formulation.

**A saturation that reshapes the operating range.** `cap * tanh(M/cap)` turned
a perfectly physical M = 0.797 into 0.651. A limiter needs a knee: identity
below 0.85, tanh above.

**Reading a dense chart by eye.** Three hypotheses were raised and tested to
explain a 150 % gap against the p. 254 chart — grid resolution, an unstated
solidity, an unstated Lock number — and all three were refuted, sigma and gamma
having almost no effect on C_T at fixed lambda' because they enter U_P only
through cos(psi) terms that average to zero. The gap was the reading: the
theta_0 = 12 label sits at (0.075, 6.5), not at 3.5. Enlarge before comparing,
and prefer a curve label to an interpolated grid crossing.

**Unstated parameters.** Three of them turned up in Chapter 3: sigma and gamma
for the charts of p. 229, which give twist, airfoil, M_1,90, c/R and tip loss
but not solidity or Lock number; and the root cutout everywhere. Each time,
the fix was to establish that the result did not depend on the guess — or, for
the root cutout, to record that it does.


## Induced velocity option of the trim (added for Chapter 5 G5, Sept 2026)

`TrimConditionsGroup(induced='exact')` uses the exact momentum induced
velocity of p. 123 instead of C_T/2mu (default, unchanged, every published
check). With it the fuselage angle bound of level flight is widened from
0.45 to 0.80 rad, because at low speed the fuselage sits in the downwash
(alpha_F = -30 deg at 30 kt, -40 deg at 25 kt). The level trim then converges
down to 25-30 kt (20,000 lb) and 30 kt (28,000 lb), not below: the downwash
angle v1/V of p. 192 is singular in hover, so this remains a forward flight
model. At mu = 0.3 the two forms agree within 1 %.
