# Chapter 3 trim — diagnostics requested by Chapter 4

Chapter 4 exposed three symptoms in the Chapter 3 trim (C4-29, C4-30, C4-35).
This file records the diagnostics run before any component is changed.

---

## D — cold start of the trim (C4-35)

**Script.** `tools/diag_trim_cold_start.py`: the book's example helicopter,
closed-form rotor, fresh problem for every point, initial guess alpha_F = −6°,
C_T/σ = 0.085, both settings of `solve_subsystems`.

**Result.**

| case | solve_subsystems = False (current) | solve_subsystems = True |
|---|---|---|
| level 80 kt | converges, 13 iterations | converges, 3 iterations |
| level 140 kt | converges, 10 | converges, 5 |
| level 175 kt | converges, 10 | converges, 15 |
| climb 80 kt, 0 ft/min | **lands on alpha_F = +25.8°, T = 5.3 G.W., hp_M = −12,760** | converges, 3; identical to level (931 hp, −6.67°) |
| climb 80 kt, 500 ft/min | same false point | converges, 10 (1,253 hp) |
| climb 80 kt, 1,500 ft/min | same false point | converges, 13 (1,927 hp) |
| climb 80 kt, 3,000 ft/min | same false point | stops at alpha_F = −25.5°, the lower bound |

Every cold start tried, from alpha_F = −20° to +10° and C_T/σ of 0.085 and
0.10, fell on the same false point with the current setting.

**Cause.** With the closed-form rotor the trim Newton runs with
`solve_subsystems=False` (the code turns it on only for the numerical rotor).
Its unknown vector then holds every explicit output of the trim as well as the
two trim states, and those outputs are unscaled: at the first iterate the
largest residuals are hp (2.8e5), H_M (2.4e5), T (2.0e4) and H_T (1.9e4),
against O(0.1) for alpha_F and C_T/σ. The Newton step and the Armijo line
search are governed by the horsepower and pound entries, the convergence is
linear instead of quadratic even in level flight, and in climb the iteration
is walked onto the upper bound of alpha_F (0.45 rad), where the 'scalar'
bound enforcement pins it and the solver stalls on a point with five times the
weight in thrust and negative rotor power. Climb at zero rate is the same set
of equations as level flight, so the difference is entirely in the path.

**Second finding.** With subsystem solves on, the steep climb at 80 kt reaches
the lower bound of alpha_F: 3,000 ft/min needs a fuselage more than 25.8°
nose down. The ±0.45 rad bound is too tight for steep climbs at low speed.

**Proposed changes (not made yet).**
1. `solve_subsystems=True` for the closed-form rotor as well, which is what
   the numerical rotor already uses.
2. Wider bounds on alpha_F in climb, or bounds tied to the flight path angle.
3. A regression test: climb at zero rate must equal level flight.
4. Re-run the maximum speed of C4-30 after (1), since its failure at
   mu = 0.46 may share this cause: level flight at 175 kt trims cold with both
   settings here, but only in 15 iterations with subsystem solves.

---

## A — where the forward flight power goes (C4-29)

**Script.** `tools/diag_power_breakdown.py`: example helicopter, 20,000 lb,
sea level, trimmed in level flight by the closed-form and the numerical
rotors (subsystem solves on). The main rotor power is split by the energy
method: induced `T v_i`, parasite `q f V`, and the remainder, which is the
blade drag power (profile torque plus the work of the profile H-force). The
book's main rotor power is taken from Figure 4.48 through the drive equation
of p. 311.

**At 140 kt (mu = 0.364).**

| hp | closed form | numerical | book |
|---|---|---|---|
| main rotor | 1,353 | 1,499 | 2,234 |
| induced | 260 | 258 | 260 |
| parasite (f = 20.0 ft²) | 571 | 568 | 571 |
| blade drag (remainder) | 521 | 673 | **1,402** |
| equivalent mean c_d | 0.0091 | 0.0117 | **0.0244** |

Induced and parasite powers agree across all three routes. **The whole gap is
blade drag power.** Put the other way, the book's number would need a parasite
area of 49 ft² with c_d = 0.010 — unlikely against the 19.3 ft² of its own
drag build-up.

**Against speed (closed form).**

| kt | mu | trim | book | gap | c_d, trim | c_d, book |
|---|---|---|---|---|---|---|
| 60 | 0.156 | 983 | 1,028 | +5 % | 0.0097 | 0.0109 |
| 80 | 0.208 | 931 | 962 | +3 % | 0.0095 | 0.0103 |
| 100 | 0.260 | 981 | 1,092 | +11 % | 0.0093 | 0.0117 |
| 120 | 0.312 | 1,120 | 1,478 | +32 % | 0.0092 | 0.0161 |
| 140 | 0.364 | 1,353 | 2,234 | +65 % | 0.0091 | 0.0244 |
| 160 | 0.415 | 1,710 | 3,080 | +80 % | 0.0091 | 0.0305 |

The book's charts carry a blade drag that rises steeply beyond mu ≈ 0.26,
three times the design value by mu = 0.42. The value at mu = 0.31, 0.0161,
matches the 0.0164 already back-solved from Table 3.5 case 6 in the Chapter 3
notes: the two findings are the same phenomenon.

**What this rules in and out.**
- The closed form is flat in c_d by construction, so it cannot follow; that
  is a modelling limit, not a bug.
- It also sits 9 % below its own energy equivalent (0.0091 against 0.0100),
  about 55 hp at 140 kt: a small shortfall in the profile H-force, consistent
  with the C_H/σ finding of the Chapter 3 notes but far too small to be the
  cause.
- The numerical rotor, whose Chapter 6 airfoil carries stall and drag rise,
  gets only to 0.0117: it does not reproduce the rise either. That is the
  lead to follow.
- One candidate to test: the isolated rotor charts are drawn for a generic
  rotor (theta_1 = −5°, M_1,90 = 0.7, c/R = 0.079), and the example helicopter
  has −10° of twist. The twist moves the retreating blade stall.

---

## C — our numerical rotor against the printed charts (in progress)

**What is compared.** The isolated rotor charts are drawn for a generic rotor,
p. 229: −5° twist, M_1,90 = 0.7 (so the tip speed changes from page to page),
c/R = 0.079, B = 0.97, and NACA 0012. The example helicopter of Chapter 4 has
−10° of twist and runs at M_1,90 = 0.79 at 140 kt. Chart 2 gives C_Q/σ against
C_T/σ at constant collective, which is exactly the quantity the power gap of
step A lives in.

**Our model on the chart rotor** (numerical rotor, momentum inflow, trimmed,
12 by 25 grid; driven as the passing chart tests drive it):

| mu | theta_0 | lambda' | C_T/σ | C_Q/σ | alpha_1,270 |
|---|---|---|---|---|---|
| 0.30 | 16° | −0.050 | 0.0906 | 0.00764 | 15.0° |
| 0.30 | 16° | −0.080 | 0.0623 | 0.00684 | 11.1° |
| 0.30 | 12° | −0.050 | 0.0474 | 0.00372 | 7.5° |
| 0.40 | 16° | −0.050 | 0.0765 | 0.00643 | 15.6° |
| 0.40 | 16° | −0.080 | 0.0485 | 0.00580 | 11.4° |
| 0.40 | 12° | −0.050 | 0.0378 | 0.00333 | 7.8° |

**Trap met on the way.** `RotorChartGenerator.point` was driven with the
inflow sign of the Chapter 3 notes and returned negative thrust with negative
torque; the passing chart tests drive `NumericalRotorGroup` directly through
their own `chart_point` helper, and that is the path used above. The
generator's convention needs checking before it is used for a whole page.

**Still to do.** Digitise chart 2 of the mu = 0.30 and 0.40 plates at the same
collectives and compare C_Q/σ point by point. Tick detection on these two
scans fails on the gridlines, so the axes have to be calibrated on the frame
and the labelled curves followed by hand.

**Why it matters.** If our model reproduces chart 2 on the chart rotor, then
the blade drag of step A is right and the Chapter 4 gap comes from the book
applying generic charts (−5° twist, M_1,90 = 0.7) to a rotor with −10° of
twist at M_1,90 = 0.79. If it does not, the drag model of Chapter 6 is short
near stall and that is what has to be fixed.

**Chart 2 digitised (mu = 0.30 and 0.40).** Axes calibrated on the dotted
gridlines, 0.01 in C_T/sigma and 0.002 in C_Q/sigma; curves traced from
C_T/sigma = 0.015 by nearest-neighbour continuity. The fan is too dense to
assign every collective safely, but the topmost curve, theta_0 = 24°, is
isolated and unambiguous:

| mu | C_T/sigma | chart | our G2 | ratio |
|---|---|---|---|---|
| 0.30 | 0.05 | 0.0104 | 0.0135 | +30 % |
| 0.30 | 0.07 | 0.0132 | 0.0179 | +36 % |
| 0.30 | 0.09 | 0.0168 | 0.0242 | +44 % |
| 0.40 | 0.05 | 0.0101 | 0.0130 | +29 % |
| 0.40 | 0.07 | 0.0130 | 0.0181 | +39 % |

**So our rotor is not short of blade drag — it is generous**, by the same 30 %
the Chapter 3 notes measured at mu = 0.10. The missing power of Chapter 4
cannot be blamed on a weak drag model.

**Where it then comes from.** At the example helicopter's own operating point
(140 kt: C_T/sigma = 0.089, f/A_b = 0.083, so X = 0.166 and lambda' = −0.025),
our G2 on the chart rotor needs theta_0 = 13.5° and returns C_Q/sigma = 0.0045,
which is what our trim gives (0.00475). The book's curves imply 0.0078. On the
same plate, at the same lambda', C_Q/sigma runs 0.0034 at theta_0 = 12°,
0.0051 at 14°, 0.0103 at 16°: **C_Q/sigma triples over four degrees of
collective**. A small error in the thrust-inflow relation — the X against
C_T/sigma of chart 1, where the Chapter 3 notes already measured G2 reading
13.09 against the chart's 13.8 — moves the collective enough to explain the
whole gap.

**Next step.** Digitise chart 1 (X against C_T/sigma) of the mu = 0.30 and
0.40 plates and compare the collective our G2 needs for the book's own
(X, C_T/sigma) pairs. If our collective is low by one or two degrees there,
the fix belongs to the thrust-inflow side of G2, not to the drag polar.

**Chart 4 digitised (mu = 0.30 and 0.40).** Identified by the alpha_TPP
formula printed on each plate: `tan^-1(3.33 lambda' + 5.56 sigma C_T/sigma)`
is 1/mu = 3.33, so mu = 0.30, and 2.5 / 3.12 is mu = 0.40. Axes calibrated on
the gridlines, 0.01 in C_T/sigma and 0.05 in lambda'. At C_T/sigma = 0.060,
where the thirteen curves are evenly spaced and unambiguous:

| mu | theta_0 | chart lambda' | our G2 | difference | in collective |
|---|---|---|---|---|---|
| 0.30 | 12° | −0.0432 | −0.0394 | +0.0038 | 0.34° |
| 0.30 | 14° | −0.0632 | −0.0606 | +0.0026 | 0.23° |
| 0.30 | 16° | −0.0877 | −0.0819 | +0.0058 | 0.52° |
| 0.30 | 20° | −0.1327 | −0.1242 | +0.0085 | 0.77° |
| 0.40 | 12° | −0.0347 | −0.0293 | +0.0054 | 0.49° |
| 0.40 | 14° | −0.0528 | −0.0490 | +0.0038 | 0.34° |
| 0.40 | 16° | −0.0758 | −0.0686 | +0.0072 | 0.65° |
| 0.40 | 20° | −0.1185 | −0.1071 | +0.0114 | 1.03° |

The sign is the same everywhere and the error grows with collective: our rotor
needs **less inflow than the chart for the same thrust and collective**, or
equivalently the chart needs half a degree to one degree more collective than
we do for the same working point. That is the same lean the Chapter 3 notes
measured on chart 1 at mu = 0.10 (X = 13.09 against 13.8, also about half a
degree).

**How much of the Chapter 4 gap that buys.** On the same plate C_Q/sigma
changes by about 0.0017 per degree of collective near the example's working
point, so 0.5 to 0.8° is 0.0009 to 0.0014 of C_Q/sigma against a deficit of
0.0031 (0.0047 ours against 0.0078 implied by the book). **The thrust-inflow
error explains about a third of the missing power, not all of it.**

**Where the rest must be.** Our C_Q/sigma reads 30 to 44 % high at theta_0 =
24° and yet the trim comes out low, so the response of C_Q/sigma to collective
differs in shape, not only in offset. The next comparison is therefore chart 2
at matched (theta_0, C_T/sigma) pairs rather than at matched trim points, over
the whole collective range, to separate a bias from a slope error.

**Matched (theta_0, C_T/sigma) on chart 2: automatic reading refused.** Two
independent schemes were tried on the mu = 0.30 and 0.40 plates: tracking each
curve from the left by continuity, and classifying the crossings at a given
C_T/sigma by their local slope, the collective curves rising and the limit
lines falling. Both break down above C_T/sigma = 0.05, where thirteen
collective curves, two alpha_1,270 lines and two delta C_Q/sigma_0 lines cross
in a band 0.006 wide: the nearest-neighbour match jumps between neighbours and
the slope test mislabels flat collective curves as limit lines. A wrong
assignment here would be worse than no reading at all, so the automatic route
is abandoned, as it already was for chart 1 in the Chapter 3 notes.

**What our model gives at those points**, for a reading by hand
(chart rotor, mu = 0.30, C_Q/sigma):

| theta_0 | C_T/sigma = 0.04 | 0.06 | 0.08 |
|---|---|---|---|
| 12° | 0.00357 | 0.00370 | 0.00329 |
| 14° | 0.00460 | 0.00516 | 0.00522 |
| 16° | 0.00576 | 0.00674 | 0.00731 |

Nine readings off the mu = 0.30 plate would settle the last question: a
constant offset against the chart means a bias, a widening one means the
response of torque to collective is wrong, and that is what decides whether
the fix belongs to the inflow side or to the drag side of G2.

**Summary of the diagnosis so far.**

| finding | measured | status |
|---|---|---|
| cold start of the climb trim | false root at the alpha_F bound; cured by solve_subsystems | cause found (D) |
| where the missing power sits | entirely in blade drag power; induced and parasite agree | located (A) |
| our drag at high collective | 30 to 44 % above the chart at theta_0 = 24° | not a shortfall |
| our inflow at matched thrust | 0.5 to 1.0° of collective short | explains a third of the gap |
| torque against collective | shape differs, not only offset | open, needs the nine readings |

**Chart 2 read on the zoom (mu = 0.30, C_T/sigma = 0.045).** One pixel column
free of gridlines and labels gives fifteen crossings: thirteen collective
curves and two limit lines. The assignment is fixed by the top curve, whose
value (0.00965) matches the theta_0 = 24° track measured on the whole plate
(0.0104 at C_T/sigma = 0.05), and the thirteen then fall in order down to
theta_0 = 0 at 0.00014.

| theta_0 | chart | our G2 | ratio |
|---|---|---|---|
| 12° | 0.00498 | 0.0036 | **−28 %** |
| 14° | 0.00589 | 0.0047 | **−20 %** |
| 16° | 0.00675 | 0.0060 | **−11 %** |
| 24° | 0.00965 | 0.0122 | **+26 %** |

**This closes the diagnosis.** Our torque is low by 10 to 30 % at the
collectives a helicopter actually cruises at, and high by a quarter at the
extreme collective: the error is in the **shape** of C_Q/sigma against
theta_0, which rises too steeply, not in the level of the drag polar. The
crossover sits near theta_0 = 18 to 20°.

**And the two errors compound in the same direction at the operating point.**
The example helicopter trims near theta_0 = 13 to 14°, where the torque is 20
to 25 % low; and the inflow lean of chart 4 pushes the trimmed collective
another 0.5 to 1.0° down, worth a further 15 to 25 % at 0.0017 of C_Q/sigma
per degree. Together they account for the 38 % of C4-29 without needing any
missing stall drag.

| finding | measured | status |
|---|---|---|
| cold start of the climb trim | false root at the alpha_F bound | cause found (D) |
| where the missing power sits | entirely in blade drag power | located (A) |
| torque against collective | too steep: −28 % at 12°, +26 % at 24° | **cause found (C)** |
| inflow at matched thrust | 0.5 to 1.0° of collective, solid above theta_0 = 18° | contributing cause |
| the two compounded | 35 to 50 % at the cruise point | matches C4-29 |

---

## Fix 1 applied — the trim solver

**Changes to `TrimConditionsGroup`.**
1. `solve_subsystems=True` for the closed-form rotor as well as the numerical
   one, with the reason recorded in the code.
2. Newton tolerances tightened to `atol=1e-12`, `rtol=1e-14`: with subsystem
   solves on, the residual norm is dominated by the unscaled explicit outputs,
   and the old 1e-10 left the trim residuals themselves at 3.3e-11 — two
   Chapter 3 tests that ask for 1e-12 caught it at once.
3. The alpha_F bound is widened to ±0.80 rad for climb and autorotation, level
   flight keeping ±0.45 rad, which was put there to tame the tail rotor.

**Effect.**

| case | before | after |
|---|---|---|
| level 80 kt | 13 iterations | 3 |
| level 140 kt | 10 | 5 |
| climb 0 ft/min | false root | identical to level, 3 iterations |
| climb 1,500 ft/min | false root | 1,927 hp |
| climb 3,000 ft/min | false root | converges, alpha_F = −29.1°, 3,048 hp |

The 3,000 ft/min case needed both changes: the solver setting alone still hit
the old bound.

**Regression tests.** `test_climb_at_zero_rate_is_level_flight` and
`test_steep_climb_trims_past_the_old_fuselage_bound` in
`tests/forward_flight/test_trim_conditions.py`.

**Chapter 4 consequences.**
- **C4-35 is resolved.** `ForwardClimbGroup` now converges from a cold start in
  under a second: 2,738 ft/min at 60 kt and 2,935 at 80 kt on maximum
  continuous, 3,992 at 80 kt on takeoff power. Two group tests were added.
  The rates sit above Figure 4.48 (2,650 and 3,270 at the peak) in the same
  direction as C4-29, as expected.
- **C4-30 is unchanged.** The maximum speed still walks to 177 kt, mu = 0.46,
  where the trim stops converging, against the book's 162 kt. That was always
  a level problem, not a solver problem: it will move only when the torque of
  G2 does.
- Nothing physical moved: Tables 3.3, 3.4 and 3.5 and every Chapter 4 anchor
  read exactly as before. The defect was in the iteration path, not in the
  equations. 375 tests pass across Chapters 1, 3 and 4.

---

## C revisited — the chart 2 assignment was wrong, and the conclusion flips

**How the error was caught.** Splitting our torque into its lift-driven and
profile parts at C_T/sigma = 0.045 gave, at theta_0 = 12°, C_Q/sigma = 0.00369
= 0.00249 (lift) + 0.00120 (profile), and the lift part is just −lambda' C_T
(0.00234). Reading chart 2 through that identity turned the earlier assignment
into an inflow of −0.082 at theta_0 = 12°, where chart 4 reads −0.035: the two
plates of the same rotor cannot disagree by that much, so one of the two
readings was wrong.

**The fix: read the labels, not the curves.** Each collective label is printed
on its own curve, interrupting it. Locating the digit glyphs on the zoom and
taking their height gives an assignment that needs no tracking at all:

| theta_0 | label sits at | chart C_Q/sigma | our G2 | ratio |
|---|---|---|---|---|
| 18° | C_T/sigma = 0.049 | 0.0064 | — | — |
| 16° | 0.055 | 0.0057 | 0.0064 | +12 % |
| 14° | 0.060 | 0.0047 | 0.0052 | +10 % |
| 12° | 0.065 | 0.0035 | 0.0036 | **+3 %** |
| 10° | 0.068 | 0.0023 | — | — |

**So our rotor is not 28 % low at cruise collective: it is 3 to 12 % high**,
and its excess grows with collective (+26 % at theta_0 = 24°, measured on the
isolated top curve). The earlier "torque too steep" finding was an artefact of
an assignment shifted by one curve, which the top-curve anchor had fixed
wrongly.

**What that does to the Chapter 4 gap.** At the example's cruise point
(C_T/sigma = 0.089, X = 0.166, so lambda' = −0.025), the chart gives
C_Q/sigma ≈ 0.004 to 0.0042, i.e. 1,200 to 1,400 hp — our trim gives 1,353 hp.
**The charts agree with us.** Figure 4.48 implies 2,234 hp at the same point.

The book's own two routes disagree by the same order elsewhere: Table 3.5
case 6 is 2,109 hp by the chart method against 1,760 by the closed form, 20 %,
where Figure 4.48 at 140 kt asks for 65 %.

**Revised conclusion.** The chain of Chapters 3 and 4 is consistent with the
isolated rotor charts within 3 to 12 % at cruise collectives. The outlier is
the power required curve of Figures 4.38 and 4.48, not our rotor. Before any
component is changed, this has to be confirmed independently — the obvious
check being to rebuild Figure 4.38 from the charts by the book's own chart
method and see which of the two book figures it lands on.

---

## Figure 4.38 rebuilt by the book's own chart method (mu = 0.30)

**Procedure, exactly as p. 229 prescribes.** Example helicopter, 20,000 lb,
sea level, mu = 0.30 (115.5 kt at 650 ft/s tip speed):

1. `C_T/sigma = G.W. / (rho A (Omega R)^2 sigma) = 0.0830`
2. `X = f/A_b + (sigma/mu^4)(C_T/sigma)^2 = 0.083 + 0.072 = 0.1555`
   with f = 20.0 ft² (the trim's own reading of Figure A.2) and A_b = 240 ft²
3. `lambda' = -X mu^3 / (2 C_T/sigma) = -0.0253`
4. Chart 4, mu = 0.30, at C_T/sigma = 0.083: the thirteen curves are ordered
   and never cross, so the reading is unambiguous — lambda' = −0.0227 at
   theta_0 = 12° and −0.0429 at 14°, so **theta_0 = 12.3°**
5. Chart 2, same plate, theta_0 = 12° curve (label-anchored) at
   C_T/sigma = 0.083: **C_Q/sigma = 0.0035**
6. `h.p._M = (C_Q/sigma) sigma rho A (Omega R)^3 / 550 = 997 hp`
7. Through the drive equation of p. 311, with the 27 hp our tail rotor needs:
   **P_req = 1,091 hp**

**The three routes at the same point.**

| route | engine hp |
|---|---|
| the book's own charts, read here | 1,091 |
| our chain (Chapter 3 trim + Chapter 4 losses) | 1,176 |
| Figure 4.48 | 1,490 |

**Conclusion.** Our chain is 8 % above the charts. Figure 4.48 is 37 % above
the charts and 27 % above us. The charts and our model agree to within the
digitising noise of the plates; the power required curves of Figures 4.38 and
4.48 do not agree with either.

This settles the direction of C4-29: **our forward flight power is not 38 %
low, the book's own power curves are about 35 % high against its own charts.**
The remaining 8 % between us and the charts is the same small excess measured
curve by curve on chart 2 (+3 % at theta_0 = 12°, +10 to +12 % at 14 and 16°).

**What this does not explain.** Why Figures 4.38 and 4.48 sit where they do.
Candidates worth one more look: a different parasite area behind those figures
(49 ft² would do it, against the 19.3 ft² of the book's own drag build-up), or
curves drawn for a heavier condition than the stated 20,000 lb. Neither is
resolved here, and neither affects our model.

**Components to change: none for now.** The rotor and the trim reproduce the
charts; the Chapter 4 figure does not, and a model is not corrected to match a
figure that contradicts its own source.
