# Validation notes — `prouty.trim`

Chapter 8, The Helicopter in Trim, p. 481-539.

Each entry records what the book prints, what an independent derivation gives,
the decision taken, and the test that enforces it.

A note on the shape of this chapter. Prouty solves the trim problem twice: once
with the full trigonometry of p. 485-515, and once with the linearised tables
he actually computes with, Table 8.4 pp. 518-521 and Table 8.11 pp. 536-537.
The two are not the same model, and the difference is not always small. Every
element and equilibrium component therefore takes `linearized=True/False`;
`True` reproduces the tables term for term, and is what the anchors below are
measured against.

---

## C8-1 — The tables drop the rotor torque coupling from `R_M` and `M_M`

**Printed.** p. 485 gives the main rotor hub moments in full, with
`X = a1s_M + i_M`:

```
R_M = (dR/db1s)_M b1s_M + Q_M sin(X)
M_M = (dM/da1s)_M a1s_M - Q_M sin(b1s_M)
N_M =  Q_M cos(X) cos(b1s_M)
```

The `R_M` row of Table 8.11 p. 537 is `(dR/db1s) b1s_M` alone, printed as
`+200940 b1s_M`. The `M_M` row of Table 8.4 p. 520 is `(dM/da1s) a1s_M` alone,
printed as `200,940 a1s_M`. Both `Q_M sin(...)` terms are gone. The `N_M` row
keeps `Q_M`, which is the small-angle limit of the third line and is correct.

**Size of the omission.** The example helicopter at 115 knots, with
`Q_M = 34,726 ft-lb`, `a1s_M = -0.019 rad` from p. 522 and `b1s_M` of the order
of the -0.89 deg read off Figure 8.31:

| | Table form | p. 485 form | difference |
|---|---|---|---|
| `R_M` | -3,114.6 | -3,774.3 | -659.7, 21 % |
| `M_M` | -3,817.9 | -3,279.6 | +538.3, 14 % |

The roll term is the one that matters. The R equilibrium equation of Table 8.11
reads `1245 + 355485 b1s_M + 341 beta + 5.43 T_T = 0`, so `Q_M sin(a1s_M)` is
half of that leading constant. Eliminating `T_T` between the R and N equations
at `beta = 0` gives `b1s_M = -0.78 deg` as printed and `-0.67 deg` with the
term restored — a 14 % move on the lateral flapping, against a figure that
reads -0.89 deg.

**Why it is not simply an approximation.** Dropping `Q_M sin(a1s_M)` is not a
small-angle step: the sine is already small, and what makes the product
non-negligible is the size of `Q_M` against the residual hub moment, not the
size of the angle. Physically it is the main rotor torque vector, tilted out of
the vertical by the shaft incidence and the longitudinal flapping, projecting
onto the roll axis. A helicopter with a stiffer hub or a lower torque would see
less of it; this one does not.

**Decision.** Neither form is corrected. `linearized=True` reproduces the
tables, which is what the printed solutions were computed from and therefore
what the anchors must be measured against. `linearized=False` keeps p. 485
intact. The gap is asserted rather than closed, so that a change which happens
to close it is noticed rather than absorbed.

**Enforced by.**
`test_trim_chapter8.py::test_main_rotor_torque_coupling_is_dropped_by_the_tables`,
`::test_main_rotor_linearized_reproduces_table_811`.

---

## C8-2 — `Z_T l_T = -148` in Table 8.4 is a rounded value squared

**Printed.** Table 8.4 p. 519 gives the tail rotor row of the Z equation as
`Z_T = -4`, and Table 8.4 p. 520 gives the corresponding row of the M equation
as `Z_T l_T = -148`.

**What the numbers are.** From Table 8.5 p. 523-524, `b1s_T = -0.0054 rad` and
`T_T = 661 lb`, so p. 487 gives

```
Z_T = b1s_T T_T = -3.569 lb
```

which rounds to the printed -4. But `-148 = -4 x 37` exactly, and `l_T = 37 ft`
is the tail rotor arm of Table 8.5. The M-equation row is therefore the
*rounded* `Z_T` multiplied by the arm, not the product of the unrounded values,
which is -132.1 ft-lb.

**Consequence.** 15.9 ft-lb, against an M-equilibrium constant of -2,871: about
0.6 %, so nothing moves. It is recorded because it is a trap, not because it
matters. Anyone anchoring the pitching moment equilibrium on -148 will spend
time chasing a 12 % gap on that row which exists only in the printing, and will
be tempted to look for it in `b1s_T`, in `l_T`, or in the sign convention of
p. 487 — none of which is wrong.

**Decision.** Compute `Z_T` from the unrounded inputs. The -148 is recorded as
a *rounded* anchor, not as a target. The `Z_T = -4` anchor is kept, with a
tolerance wide enough to admit -3.57.

**Enforced by.**
`test_trim_chapter8.py::test_tail_rotor_ZT_lT_row_of_table_84_is_a_rounding_artifact`,
`::test_tail_rotor_reproduces_table_84`.

---

## Open anchors

Not disagreements, but numbers that do not close exactly and are worth knowing
about while the chapter is built.

| where | book | here | gap |
|---|---|---|---|
| `T_T` from the antitorque relation, p. 487 and p. 510 | 661 lb, Table 8.5 | 667.1 lb | +0.9 % |
| Vertical stabiliser lift curve slope, Figure 8.6 p. 490 | 3.0 /rad, Table 8.3 | 3.29 /rad | +9.7 % |
| Fin effective aspect ratio, Figure 8.19 p. 505 | 3.2, p. 504 | 3.34 | +4.4 % |

The first is localised: `Q_M/l_T` reproduces the 939 lb of Table 8.5 to 0.05 %,
so the difference sits in Prouty's `Y_V` and not in the relation. Table 8.5
gives `L_V = 287 lb` and `D_V = 58 lb`, which with the sidewash angles of
Table 8.3 make `Y_V = 287.4`; closing the gap would need `Y_V = 293.4`.

The second is a reading of Figure 8.6, whose ordinate is the
Helmbold-Diederich relation

```
a / A.R. = 2 pi / (2 + sqrt[ A.R.^2 (1 + tan^2 Lambda_c/2) + 4 ])
```

and not an empirical curve — the printed scale tops out at `pi/2 = 1.571`,
which is the `A.R. -> 0` limit of that expression. It reproduces the horizontal
stabiliser of Table 8.2 exactly (4.020 against 4.0 at `A.R. = 4.5`,
`Lambda = 13 deg`) and misses the vertical stabiliser by 9.7 % (3.290 against
3.0 at `A.R.eff = 3.2`, `Lambda = 27 deg`). Since the same formula lands on one
surface and not the other, the likeliest cause is a chart reading rather than a
missing term. To be settled when the lift curve slope component is written.

---

## C8-3 — `vortex_axis` runs 16 % above Figure 8.11

**Not a disagreement with the book.** An alternative to reading it, and a
record of what that alternative costs.

**Why an alternative exists at all.** Figure 8.11 pp. 494-495 is a survey, not
a curve: nineteen traces of `v_H/v1` against `Y/R`, at two longitudinal
stations and ten vertical ones, from Heyson and Katzoff, NACA TR 1319, 1957.
Using it means choosing the trace nearest the stabiliser's `X'/R` and `Z'/R`
and averaging over its span — a judgement rather than an interpolation. It is
therefore left as a free input, `downwash_model='input'`, which is the default.

For a geometry sweep, where `X'/R` and `Z'/R` are design variables, that
default has nothing to offer. `downwash_model='vortex_axis'` supplies a
momentum value instead: the on-axis induced velocity of a uniform vortex
cylinder,

```
v(s)/v1 = 1 + s/sqrt(s^2 + R^2)
```

evaluated at the distance along the skewed wake axis, with `k = v1/V`,

```
s/R = (-X'/R + k Z'/R) / sqrt(1 + k^2)
```

**Size of the gap.** Example helicopter at 115 knots, `X'/R = -1.08`,
`Z'/R = +0.3`, `k = 0.0405`:

| | `v_H/v1` | `eps_MH` |
|---|---|---|
| Figure 8.11, as Prouty reads it | 1.5 | 0.0607 rad, 3.48 deg |
| `'vortex_axis'` | 1.737 | 0.0703 rad, 4.03 deg |

**Where the 16 % comes from, and why it was not tuned away.** Two known
omissions, both in the same direction. The model keeps the rotor radius as the
cylinder radius, whereas at `mu = 0.3` the wake axis is nearly horizontal and
the cross-section normal to it is an ellipse of semi-axes `R` and `kR`; the
on-axis law is therefore evaluated on a tube far wider than the real one, and
overshoots. And it ignores lateral position entirely, so the advancing-side
asymmetry that p. 495 identifies as the source of pitch-sideslip coupling is
absent by construction.

Neither is repairable inside a level-one model. Correcting the first needs the
skewed elliptic cylinder of Castles and De Leeuw, NACA TR 1184 — complete
elliptic integrals, differentiable but heavy. Correcting the second needs the
non-uniform disc loading TR 1319 was written about, which is also the only way
to reach the values above 2 that Figure 8.11 shows and no uniform-disc theory
can produce. A fitted fudge factor on `R` would close the 16 % on this one
geometry and mean nothing on the next, which is the failure mode the option
exists to avoid.

**Decision.** Ship both. `'input'` is the default and is what the anchors are
measured against; `'vortex_axis'` is documented as generous by roughly a fifth
and is for keeping a gradient-based sweep moving, not for replacing a reading.

**Enforced by.**
`test_trim_chapter8.py::test_vortex_axis_runs_high_against_figure_811`,
`::test_vortex_axis_is_unity_at_the_disc_and_two_far_downstream`,
`::test_downwash_reproduces_page_493`.

---

## C8-4 — Table 8.4's `X_H` rows carry a different `L_H` from Table 8.5

**Printed.** Table 8.4 p. 518 gives the two horizontal stabiliser rows of the
X equation as a constant plus a part linear in `Theta` and `T_M`. For the tilt
of the stabiliser lift:

```
7        -235 Theta + .0008 T_M
```

Table 8.5 p. 523 lists the trim lift of that surface as `L_H = -273 lb`.

**Rebuilding the coefficients.** p. 488 and p. 498 make the row
`L_H [Theta - gamma_c - (v_H/v1) T_M/(4 q A_M) - eps_F0
- (d eps_F/d alpha_F)_H (Theta - gamma_c - T_M/(4 q A_M))]`, so with
`eps_F0 = 0.024`, `(d eps_F/d alpha_F)_H = 0.23`, `v_H/v1 = 1.5`,
`q = 45` and `A_M = 2827` the three printed numbers each imply a value of
`L_H`:

| coefficient | expression | printed | implied `L_H` |
|---|---|---|---|
| constant | `-L_H eps_F0` | 7 | -292 |
| `Theta` | `L_H (1 - 0.23)` | -235 | -305 |
| `T_M` | `L_H (1.5 - 0.23)/(4 q A_M)` | .0008 | -320 |

Three independent routes, all landing between -292 and -320 against the -273
of Table 8.5, and mutually consistent to the two significant figures the table
prints: taking `L_H = -305` reproduces all three as 7.32, -234.9 and .000761.

**The Z rows do close.** The same exercise on the `Z_H` lift row of p. 519
rebuilds exactly from `q_H/q = 0.6`, `A_H = 18` and `a_H = 4.0`:

| coefficient | computed | printed |
|---|---|---|
| constant | 147.7 | 149 |
| `Theta` | -1496.9 | -1497 |
| `T_M` | .004852 | .0048 |

So the parameters of Table 8.2 and Table 8.5 are right, the downwash model is
right, and the discrepancy is confined to whichever `L_H` was substituted into
the X rows.

**What the lift itself says.** `L_H = -273` over `(q_H/q) q A_H = 486` gives
`C_LH = -0.5617` and, at `a_H = 4.0`, `alpha_H = -8.05 deg`. That is the
-8.03 deg this package computes from p. 489, and not the -7.9 deg Table 8.8
p. 530 lists for level flight. Prouty's own lift confirms the angle of attack
chain against the table that appears to contradict it, which leaves -273 as
the trustworthy number and the X rows as the outlier.

**Decision.** Not resolved at element level. `HorizStabForcesComp` computes
from whatever `L_H` it is handed, and the anchors above are recorded so that
the X-equilibrium component is built knowing its printed row cannot be matched
term for term with `L_H = -273`. To be settled there.

**Enforced by.**
`test_trim_chapter8.py::test_horiz_stab_lift_matches_table_85`,
`::test_horiz_stab_lift_confirms_the_alpha_H_chain`.

---

## Open anchors, continued

| where | book | here | gap |
|---|---|---|---|
| `D_H`, Table 8.5 p. 523 | 15 lb | 14.13 lb | -5.8 % |

Both parts of the polar are small. Neither `delta = 0.02` nor `C_D0 = 0.0064`
nor `A.R. = 4.5` can be moved far enough to close it without breaking
something the lift already anchors. Recorded, not chased.

The vertical stabiliser lift curve slope entry of the first open-anchor table
is now enforced by
`test_trim_chapter8.py::test_lift_curve_slope_vertical_stabiliser_runs_high`
and `::test_lift_curve_slope_horizontal_stabiliser`.

---

## C8-5 — Two different parameters, both credited to Figure 8.15

**Correction to an earlier reading of this entry.** A first pass on a poor
scan read Table 8.5's `(d eps_F/d alpha_F)_V` as `-.23` and recorded a sign
error in the book. A clean read of p. 524 shows it printed as `.23`, positive.
There is no typographical error. What follows is the finding that survives.

**Printed.** Table 8.5 pp. 524-525 credits Figure 8.15 twice, once for the
horizontal stabiliser at `.23` and once for the vertical stabiliser, also at
`.23`. p. 509 then says the fuselage sidewash may be assumed to follow
`d eta_F/d beta = d eps_F/d alpha = 0.06`, "taken from Figure 8.15 for the
downwash effect of the body alone".

**They are not the same parameter.** One is longitudinal, one is lateral, and
the fin needs both:

| | enters | through | value used by the tables |
|---|---|---|---|
| `(d eps_F/d alpha_F)_V` | `Z_V`, Table 8.4 p. 519 | tilt of the fin X-force in side view | +0.23 |
| `d eta_F/d beta` | `Y_V`, Table 8.11 p. 536 | fin angle of attack in plan view | 0.06 |

Prouty credits Figure 8.15 for both because it is the only fuselage
interference chart in the chapter, and reads the whole-configuration value for
one and the body-alone value for the other. The trap is that Table 8.5 lists
only the first, so an implementation that takes its `.23` for the sidewash as
well will be 16 % out on the fin side force.

**Rebuilding the printed coefficients confirms both.**

The `Y_V` lift row of Table 8.11 is `-q (q_V/q) A_V a_V (1 + d eta_F/d beta)`
against `beta`. With `q = 45`, `q_V/q = 0.6`, `A_V = 33` and `a_V = 3.0` the
leading factor is 2673:

| `d eta_F/d beta` | coefficient | printed |
|---|---|---|
| 0.06 | -2833.4 | **-2833** |
| 0.23 | -3288 | — |

The `Z_V` row of Table 8.4 is
`-[D_V + L_V(eta_MV + eta_TV)] (1 - (d eps_F/d alpha_F)_V)` against `Theta`,
and the bracket is `58 + 287(-0.052 + 0.046) = 55.99`:

| `(d eps_F/d alpha_F)_V` | coefficient | printed |
|---|---|---|
| +0.23 | -43.4 | **-43** |
| -0.23 | -69.3 | — |

The other two coefficients of the same row agree at +0.23 as well: the
constant is `55.99 x 0.024 = 1.34` against a printed 1, and the `T_M`
coefficient is `.00013975` against a printed .0001.

**Decision.** `FuselageSidewashComp` defaults to p. 509's 0.06 for the lateral
slope, and the longitudinal slope for the fin is +0.23, the same magnitude and
sign as the horizontal stabiliser's. Nothing in the code carries -0.23.

**Enforced by.**
`test_trim_chapter8.py::test_fuselage_sidewash_slope_is_the_006_of_page_509`,
`::test_vert_stab_Z_is_the_tilt_of_its_own_X_force`.

---

## C8-6 — Three quarters of the fin drag is biplane interference

**Not a disagreement.** A result worth flagging, because it is easy to build
the fin drag from its polar alone and be 74 % low.

Table 8.5 p. 524 gives `D_V = 58 lb`. The fin's own polar at `C_LV = 0.324`,
`A.R.eff = 3.2`, `delta = 0.01` and `C_D0 = 0.0064` produces 14.9 lb. The
remaining 42.8 lb is the tail rotor interference of p. 509:

```
dD_int = (8/pi) |(T_T / 2 R_T)(Y_V / b_V)| K_int / q
```

with `T_T = 661`, `R_T = 6.5`, `Y_V = 287`, `b_V = 7.7`, `K_int = 0.4` and
`q = 45`. Total 57.7 against the printed 58.

The term is large because it scales with the *product* of two forces, and the
tail rotor thrust is an order of magnitude larger than anything the fin's own
drag involves. It also means fin drag is not a fin property: it moves with
tail rotor thrust, hence with main rotor torque, hence with the whole trim
state.

**Two consequences for the model.** The absolute value is not differentiable
where either force crosses zero, which a sideslip sweep through zero fin load
can find; the value is exact as printed and the kink is documented rather than
smoothed. And `Y_V` inside it is itself a function of the drag it produces;
p. 510 breaks the loop by taking `Y_V` at its no-interference value, which is
what `tail_rotor_feedback=False` does at group level.

**Enforced by.**
`test_trim_chapter8.py::test_vert_stab_drag_is_mostly_interference`,
`::test_interference_drag_kink_at_zero_side_force`.

---

## C8-7 — Figure A.2 is empennage on, so Chapter 3 cannot feed Chapter 8's fuselage

**The trap.** `prouty.forward_flight.FuselageAeroComp` reads Figure A.2 p. 679
and returns `L_F` and `D_F` for the example helicopter. Chapter 8 also needs
`L_F` and `D_F` for the example helicopter. Wiring one into the other is the
obvious move and it is wrong: Figure A.2's curves are labelled *empennage on*,
while Chapter 8 models the horizontal stabiliser as its own component. The
stabiliser would be counted twice.

**The numbers say so.** Table 8.5 p. 525 gives the bare fuselage lift slope as
75 ft^2/rad, credited to Appendix A. Chapter 3's calibrated Figure A.2 slope is
1.953 ft^2/deg. The horizontal stabiliser contributes
`(q_H/q) A_H a_H = 0.6 x 18 x 4.0 = 43.2 ft^2/rad`:

| | ft^2/deg |
|---|---|
| Figure A.2, empennage on | 1.953 |
| horizontal stabiliser | 0.754 |
| difference | 1.199 |
| Table 8.5, bare fuselage | 1.309 |

Within 9 %, which is as close as two readings of two different figures by the
same author get. The direct check fails as badly as expected: Figure A.2 at
`alpha_F = -3.26 deg` gives `L/q = -10.9 ft^2` where Table 8.5's derivatives
give -5.8.

**Decision.** Chapter 8 gets its own `FuselageDerivativesComp`, a constant plus
a slope per axis from Table 8.5, and does not import the Chapter 3 component.
The two coexist and describe different bodies. The same caution applies to
drag: Figure A.2's `f` is the whole airframe, so the `f` fed to Chapter 8 must
exclude both stabilisers.

**Enforced by.** `test_trim_chapter8.py::test_chapter_3_figure_a2_is_empennage_on`.

---

## Open anchors, continued

| where | book | here | gap |
|---|---|---|---|
| `L_F`, Table 8.5 p. 523 | -281 lb | -259.5 lb | -7.6 % |

`M_F` from the same two derivatives lands at -11,757 against a printed -11,722,
0.3 %, so the pitching moment pair is right and the lift pair is the odd one.
Inverting the printed lift gives `alpha_F = -3.62 deg`, where the same table's
inputs give -3.26 deg and Table 8.8 p. 530 lists -3.3 deg. The gap is in the
book's own iteration between its "initial trim" and "resultant" columns —
which also disagree by 34 lb on `L_H + L_F` and by 19 lb on `D_H + D_V + D_F` —
and not in the derivative model.

---

## C8-8 — Table 8.4's M equation has no `Z_V l_V` row

**Printed.** Table 8.4 pp. 520-521 builds the pitching moment from a `-X h`
term and a `+Z l` term for each component:

```
M_M  -X_M h_M  +Z_M l_M
M_T  -X_T h_T  +Z_T l_T
     -X_H h_H  +Z_H l_H
     -X_V h_V
M_F  -X_F h_F  +Z_F l_F
```

The vertical stabiliser contributes only the first of its pair. There is no
`Z_V l_V` row.

**Size of the omission.** p. 502 gives the fin a Z-force, `Z_V = X_V sin(phi_V)`,
worth 4.93 lb at 115 knots, and Table 8.5 puts the fin 35 ft aft of the c.g.
The missing term is 173 ft-lb, against an M equation the book itself closes to
about 129 ft-lb at its converged unknowns. It is above the noise of its own
solution.

**Decision.** The same rule as C8-1. `linearized=True` reproduces Table 8.4 as
printed and leaves the term out; `linearized=False` includes it, because the
fin does produce that moment. The option has meant "the equations Prouty
solved" rather than only "small angles" since C8-1, and this is the second
place the distinction bites.

**What it buys, measured.** Running the element group at Prouty's converged
unknowns and closing the three equations on its output:

| | `res_X` | `res_Z` | `res_M` |
|---|---|---|---|
| `linearized=True` | 0.97 | 18.6 | **103** |
| `linearized=False` | 1.26 | 23.0 | 785 |

Prouty's own linearised M equation, `-2871 + 355485 a1s + 32253 Theta
+ .5004 T_M`, evaluated at `a1s = -0.019`, `Theta = -0.0165` and
`T_M = 20,556`, leaves 129. So the package closes the book's own equations as
well as the book does, to 103 ft-lb out of terms reaching 12,000.

The 682 ft-lb between the two rows is not a mystery: 538 of it is the
`Q_M sin(b1s_M)` term of C8-1 and 173 is this one, which together account for
711 of it. That is the strongest evidence so far that both omissions are real
and that nothing else is wired wrong.

**Enforced by.**
`test_trim_chapter8.py::test_M_equilibrium_ZV_lV_row_is_missing_from_table_84`,
`::test_table_84_mode_reproduces_proutys_own_residual`.

---

## C8-9 — The hover lateral flapping of p. 532 contradicts the roll angle beside it

**Printed.** p. 532 solves the hover lateral-directional equations for the
example helicopter and gives

```
b1s = -.027 rad = -1.5 deg
Phi = -0.050 rad = -2.9 deg
```

with the parameters of Table 8.10 p. 533: `G.W. = 20,000`, `T_M = 20,840`,
`T_T = 1,540`, `y_M = 0`, `h_M = 7.5`, `h_T = 6`, `dR_M/db1s = 200,940`.

**What those parameters give.** The R equation of p. 531 is triangular in
`b1s_M`:

```
b1s_M = (T_M y_M - T_T h_T) / (dR_M/db1s + T_M h_M)
      = (0 - 1540 x 6) / (200,940 + 20,840 x 7.5)
      = -9,240 / 357,240 = -0.025865 rad = -1.482 deg
```

which rounds to the printed -1.5 deg but not to the printed -0.027 rad.
Nothing in Table 8.10 can be moved to reach -0.027: it would need
`T_T = 1,608` where the table says 1,540, or `T_M = 18,838` where it says
20,840.

**The roll angle settles it.** `Phi = -(T_T + T_M b1s_M)/G.W.`, so:

| `b1s_M` used | `Phi` | would print as |
|---|---|---|
| -0.025865, computed here | -0.050049 | **-0.050 rad, -2.9 deg** |
| -0.027, as printed | -0.048866 | -0.049 rad, -2.8 deg |

p. 532 prints -0.050 and -2.9. Prouty's own roll angle was therefore computed
from -0.0259, and the -0.027 beside it is a slip in transcription, not a
different calculation.

This is the second time a printed result corroborates the package against
another printed result on the same page: the first was `L_H = -273`
confirming `alpha_H = -8.05 deg` against Table 8.8's -7.9, in C8-4.

**Decision.** Compute from Table 8.10 and anchor on `Phi`, which both routes
agree about. The degree figure for the flapping, -1.5, is also kept as an
anchor; only the radian figure is excluded.

**Enforced by.**
`test_trim_chapter8.py::test_hover_lateral_flapping_disagrees_with_its_own_roll_angle`,
`::test_hover_lateral_roll_angle_matches_page_532`.

---

## C8-10 — Table 8.11's weight row has the wrong sign

**Printed.** Table 8.11 p. 536 gives the last row of the Y equation as

```
Tilt of Gross Weight    -G.W. Phi        -20,000 Phi
Equilibrium  = 0        415 + 20606 b1s - 13068 beta + .809 T_T - 20,000 Phi = 0
```

**p. 531 says the opposite.** The hover form of the same equation is

```
Y    T_M b1s_M + T_T = -G.W. Phi
```

which rearranged as a sum equal to zero puts a *plus* in front of the weight
term. The physics agrees: with `Phi` positive right-side-down, gravity leans
toward positive body Y, so its body-Y component is `+G.W. cos(Theta) sin(Phi)`.
Table 8.4's X row, `-G.W. Theta`, is correctly signed by the same reasoning.

**Figure 8.31 decides it.** The R and N rows of Table 8.11 can be solved for
`b1s_M` and `T_T` without touching the Y row, and Figure 8.31 p. 538 annotates
both at two points. They agree:

| | computed from Table 8.11 | Figure 8.31 |
|---|---|---|
| `beta = 0`: `b1s_M` | -0.779 deg | -0.78 deg |
| `beta = 0`: `T_T` | 660.7 lb | 661 lb |
| `Phi = 0`: `b1s_M` | -0.894 deg | -0.89 deg |
| `Phi = 0`: `T_T` | 789.1 lb | 789 lb |
| `Phi = 0`: `beta` | 3.21 deg | 3.2 deg |

Those two rows are therefore right, which makes the Y row a clean test. At
`beta = 0` it returns

| weight term | `Phi` |
|---|---|
| `+G.W. Phi`, p. 531 | **-1.92 deg, left** |
| `-G.W. Phi`, as printed | +1.92 deg, right |

Figure 8.31 shows -1.9 deg, left, and labels the axis so there is no reading
ambiguity. The hover solution of p. 532 is also negative, -2.9 deg, and a
helicopter does not change which way it hangs between hover and 115 knots.

**Decision.** `LatYEquilibriumComp` uses `+G.W. cos(Theta) sin(Phi)`, and
`linearized=True` gives `+G.W. Phi` — Table 8.11's row with its sign
corrected, which is the one place in this package where the linearised option
does *not* reproduce the printed table. Reproducing it would put the example
helicopter in a right bank in hover and in forward flight, contradicting both
Figure 8.28 and Figure 8.31.

**Enforced by.**
`test_trim_chapter8.py::test_Y_equilibrium_weight_sign_follows_page_531_not_table_811`,
`::test_table_811_rows_reproduce_figure_831_at_zero_sideslip`,
`::test_lateral_equilibrium_at_the_zero_bank_point`.

---

## C8-11 — p. 535 names a combination Chapter 3 does not produce

**Printed.** p. 535 gives the lateral cyclic at trim as

```
A_1 = (A_1 + b1s_M)_beta=0 - b1s_M + B_1 sin(beta)
```

and says the first term comes from the performance charts of Chapter 3, in
exact parallel with the longitudinal `B_1 = (B_1 + a1s_M) - a1s_M` of p. 522.

**Chapter 3 sets the difference, not the sum.** p. 169 obtains the lateral
flapping by setting the rotor pitching moment to zero, and what falls out is

```
A_1 - b1s_M = -[(4/3) mu a_0 + v1/(Omega R)] / (1 + mu^2/2)
```

The asymmetry with the longitudinal case is deliberate and Prouty's own: it is
`B_1 + a1s_M` but `A_1 - b1s_M`, a consequence of the pitch convention
`theta = theta_0 + (r/R) theta_1 - A_1 cos(psi) - B_1 sin(psi)` of p. 165. The
`prouty.forward_flight` package implements it that way, validated against
Table 3.3, and names the output `A1_b1s`.

The p. 535 formula is internally consistent — `(A_1 + b1s) - b1s` does return
`A_1` — but the quantity it asks for is not the one Chapter 3 hands over.
Mirroring the p. 522 formula without flipping the sign with it produces
exactly this.

**Size of it.** The two readings differ by `2 b1s_M`. With
`A_1 - b1s_M = -2.3 deg` from Chapter 3 and `b1s_M = -0.78 deg` from the
lateral trim at zero sideslip:

| reading | `A_1` |
|---|---|
| Chapter 3 convention, used here | -3.08 deg |
| p. 535 as printed, fed the Chapter 3 number | -1.52 deg |

1.56 degrees against a 2.3 degree aerodynamic requirement. Not cosmetic.

**Decision.** `LatCyclicPitchComp` takes Chapter 3's combination under
Chapter 3's name, `A1_b1s`, and adds `b1s_M`. The alternative would be to
accept p. 535 literally and require the caller to supply a sum that nothing
in the package computes.

**Settled by the rigging curves.** When this entry was written the argument
rested on p. 165 and p. 169 rather than on arithmetic, because p. 535 gives no
numerical example for `A_1` and Figure 8.31 plots lateral control position in
per cent. Figure A.5 p. 682 has since been digitised and it converts:

```
pct_A1 = (A_1 + 10.333) / 0.17144        % from full left
```

| reading | `A_1` | lateral stick | Figure 8.31 at `beta = 0` |
|---|---|---|---|
| Chapter 3 convention, used here | -3.08 deg | **42.3 %** | 44 % |
| p. 535 as printed | -1.52 deg | 51.4 % | 44 % |

Two points against seven. The Chapter 3 convention is right and p. 535 has
mirrored the longitudinal formula of p. 522 without flipping the sign with it,
as this entry supposed.

**Enforced by.**
`test_trim_chapter8.py::test_lat_cyclic_pitch_uses_the_chapter_3_convention`,
`::test_lat_cyclic_pitch_at_zero_sideslip`.

---

## C8-12 — Table 8.6's climb angle contradicts its own resultant forces

**Printed.** Table 8.6 p. 527 gives the flight condition for the speed
stability calculation at 135 knots as `G.W. = 20,000`, `q = 61.5` and

```
Climb angle    gamma_c    rad    .049(a)        (a) Using method of Chapter 3
```

positive, a 2.8 degree climb. The text that sets the calculation up, p. 526,
describes the opposite: the trim is recomputed "at a slightly higher speed
assuming no change in collective pitch", using the Chapter 3 example
"Helicopter in Dive at Constant Collective Pitch", and the result is expected
to be "the angle of climb, `gamma_c` (or dive, `gamma_D`)". At fixed collective
a helicopter accelerating from 115 to 135 knots must descend.

**The same table's resultant forces give the angle back.** Three of them
depend on `gamma_c` only through an angle of attack, and each can be inverted
using the derivatives of Table 8.5, with `Theta = -2.3 deg` and
`T_M = 20,496 lb` from p. 526:

| from | implies `alpha` | implies `gamma_c` |
|---|---|---|
| `L_F = -250` | `alpha_F = -1.96 deg` | -2.03 deg |
| `M_F = -14,268` | `alpha_F = -2.32 deg` | -1.67 deg |
| `L_H = -334` | `alpha_H = -7.20 deg` | -1.49 deg |

All three land near -1.7 degrees: a shallow dive, as the text says, and not
the +2.8 degree climb the table prints. The sign of the printed value is
wrong, and its magnitude is not right either.

**What this package returns.** `LongitudinalTrimGroup` fed the Table 8.6
elements, with `T_M_bar = 20,618`, `H = -72`, `Q_M = 882 x 37`, `H_T = 34`,
`Q_T = 164`, `b1s_T = -0.012`, `f = 1049/61.5`:

| `gamma_c` | `T_M` | `Theta` | `a1s_M` | `B_1` | `L_H` |
|---|---|---|---|---|---|
| +0.049, as printed | 21,238 | -2.12 | -0.58 | 9.18 | -481 |
| -0.030, back-computed | 20,576 | -1.84 | -1.12 | 9.72 | -306 |
| -0.049 | 20,412 | -1.73 | -1.26 | 9.86 | -262 |
| **book** | **20,496** | **-2.30** | **-0.70** | **9.30** | **-334** |

The back-computed angle gives the best thrust, 0.4 %, and the best stabiliser
lift, 8 %, but leaves `Theta` and `a1s_M` about 0.4 deg out in every case.
Two things that would close that are not printed anywhere: the value of
`v_H/v1` at 135 knots, which Figure 8.11 makes a function of `V/v1_hover`, and
`q_H/q` at the new speed.

**Decision.** The 135 knot point is not used as an anchor. What is anchored is
the conclusion, and it survives the ambiguity: `B_1` rises from 8.9 deg at
115 knots to between 9.2 and 9.9 deg at 135 knots for every `gamma_c` tried,
so the example helicopter has positive speed stability whichever value is
taken. That is the whole finding of pp. 525-528, and it does not depend on
resolving the table.

**Enforced by.** `test_trim_chapter8.py::test_speed_stability_of_page_527`,
which anchors the sign and the 0.4 deg shift Prouty reports, not the 135 knot
trim itself.

---

## C8-13 — Figure 8.19's ordinate label contradicts its own legend

**Printed.** The first panel of Figure 8.19 p. 505 labels its ordinate

```
A.R._V / A.R._V+B
```

and prints, inside the plotting area, a legend defining the same symbol as
"Ratio of the aspect ratio of the vertical panel in the presence of the body
to that of the isolated panel" — which is `A.R._V+B / A.R._V`, the reciprocal.
They cannot both be right.

**The worked example settles the usage without settling the wording.** p. 504
reads 1.05 off this chart for the example helicopter and multiplies by it, and
1.05 is what the printed ordinate shows at `b_V/2r_1 = 7.7/1.5 = 5.13`. So
whatever the label ought to say, the number to take is the one on the axis and
the operation is a multiplication. The component does that.

The distinction is not academic at other geometries: the ordinate peaks at
1.63 near `b_V/2r_1 = 2`, so reading it upside down there would divide the fin
aspect ratio by 1.63 instead of multiplying, a factor of 2.7 wrong.

**Which curve is which.** The two curves are labelled `lambda_V <= .6` and
`1.0`, and the labels sit close enough together to be ambiguous at first
reading. The upper curve is `lambda_V <= .6`: the `1.0` is printed on the peak
of the *lower* curve, which is where a curve label goes, while the
`lambda_V <= .6` block sits above both as a heading. The example helicopter has
`lambda_V = .21`, so it uses the upper curve, and the upper curve is the one
that returns 1.05 at 5.13.

---

## Digitisation of Figure 8.19

The three panels were tracked with the predictor-corrector used for Figure A.2
in `prouty.forward_flight`, smoothed with a spline and sampled onto regular
grids that `VertStabEffectiveARComp` interpolates with Akima splines.

| factor | digitised | p. 504 | error |
|---|---|---|---|
| `f_B`, panel a | 1.062 | 1.05 | +1.1 % |
| `f_H`, panel b | 1.133 | 1.1 | +3.0 % |
| `K_H`, panel c | 0.660 | 0.64 | +3.2 % |
| `A.R._V,eff` | 3.34 | 3.2 | +4.4 % |

Each factor sits within chart-reading precision of Prouty's own value. They
happen to err the same way, so the product runs 4.4 % high; through the
Helmbold relation of C8-6 that is about 2 % on the fin lift curve slope, on top
of the 9.7 % already open between that relation and Table 8.3.

### Two curves that look merged and are not

A first pass concluded that the two curves of panel a become one line beyond
`b_V/2r_1 = 5.2`, and that the four curves of panel b merge beyond
`Z_H/b_V = -0.8`, and averaged them together on that basis. Both conclusions
were wrong, and each was wrong for a different reason.

**Panel a.** The curves do touch on paper, but the dark band there is 5 to 6
pixels thick where an isolated line is 2 to 3. Two lines are still present.
They are recovered by splitting the band: where the two *are* separately
resolved, the band extent minus 2.5 pixels reproduces their measured spacing
exactly, which calibrates the rule, and the same rule then carries them apart
to the right-hand frame. The separation falls from 0.152 at `b_V/2r_1 = 1` to
0.008 at 7 and is nowhere zero.

**Panel b.** These four never approach each other at all. They appear merged
past `Z_H/b_V = -0.78` only because they turn upward and become steep, so one
column crosses many rows of a single curve and the column-wise tracker reports
a thick band that is not four curves. Past that point they are tracked by
*rows*: at each ordinate the band's left edge is the `x/c_V = 0.8` curve and
its right edge the 0.5 curve, since the higher curve reaches a given value at
the lesser height ratio. The two middle curves are placed evenly between,
which the last cleanly resolved column supports — there the four are spaced
0.023, 0.023 and 0.032 apart. The ordering `f_H(0.5) < f_H(0.6) < f_H(0.7) <
f_H(0.8)` then holds at every tabulated `Z_H/b_V`, which is the check that the
row tracking worked.

**Where panel b stops.** The tables end at `Z_H/b_V = -0.95`, not -1.0,
because the curves leave through the *top* of their own frame at about -0.97
rather than through the right-hand edge. There is nothing printed to read
between -0.95 and -1.0, and extrapolating a spline into it produced values
between 1.4 and 2.7 depending on the curve, which is how the omission was
caught.

Panel c needed none of this. It is clean over its whole span except the last
3 % of the abscissa, where the legend text meets the curve; the table is
tracked to `S_H/S_V = 1.93` and the spline carries the rest.

---

## Digitisation of Figures 8.8, 8.10, 8.17 and 8.22

With Figure 8.19 done, these four close the chapter's abaques. Each needed a
different method, and the reason is worth recording: the right method is set
by how the curves fail, not by how they look.

| figure | what it gives | method | agreement |
|---|---|---|---|
| 8.8 p. 491 | end plate factor | straight line, fitted | slope 1.655 |
| 8.10 p. 493 | `(q_H - q)/D.L.` | column tracking from the peak | hits all 5 printed points |
| 8.17 p. 503 | `delta_i` | column tracking from the right frame | ordering holds everywhere |
| 8.22 p. 510 | `K_int` | complete rows only, robust quadratic | 0.400 against a read 0.4 |

**Figure 8.8 is a line, not a curve.** Tracked row by row over `h/b_H = 0.007`
to 0.56, a free fit gives `1.0141 + 1.6161 h/b_H` with a maximum residual of
0.0226. The intercept is forced to 1 — no end plate can be no effect — leaving
a slope of 1.655. Like Figure 8.6, the chart dissolves into a formula and no
table is stored.

It is **not** Hoerner's `1 + 1.9 h/b`, the result usually quoted. Figure 8.8 is
13 % below it and is credited to reference 8.3. At `h/b = 0.4` that is 1.66
against 1.76, worth about 3 % on the stabiliser lift curve slope. The chart is
what Prouty used.

**Figure 8.10 checks itself.** It plots five measured points alongside the
faired curve, and the digitised track passes through every one: 0.80 at
`V/v1 = 1.15`, 1.11 at 1.9, 0.35 at 2.7, zero at 3.7 and the low-speed zero at
0.5. The table is set to zero outside 0.45 to 3.8 rather than extrapolated,
because that is what the figure shows — the wake has missed the stabiliser by
3.7 and has not reached it below 0.5.

**Figure 8.17 needed the start point chosen carefully.** The five curves are
cleanly separated at the right frame and nearly vertical at the left, so all
five are tracked leftward from `lambda = 1.18`. The `A.R. = 20` curve passes
through the "Aspect Ratio" caption and needed a wider tolerance to cross it;
masking the caption instead removed part of the curve with it, which showed up
as a gap from `lambda = 0.59` to 1.14. The tables run from `lambda = 0.10`,
below which the curves turn nearly vertical and five near-vertical lines cannot
be told apart by columns.

Figure 8.17 starts at `A.R. = 4` and the example helicopter's fin is 3.2
effective. The component clamps rather than extrapolating, which gives 0.021 at
`lambda_V = 0.21` against the 0.01 Prouty reads — a difference worth 0.1 lb on
the fin, which is why it is clamped and not chased.

**Figure 8.22 is transposed and doubly parameterised.** Its abscissa is
`K_int` and its ordinate the separation ratio `2 y_V/(2 R_T + b_V)`, which is
the reverse of how it is used, so it is digitised by rows. Its curve family is
labelled `mu = b_V/2R_T`, a local notation with nothing to do with the tip
speed ratio that carries the same symbol everywhere else in the book.

Row tracking with a predictor failed here — the five curves are 20 to 28 pixels
apart and the trackers still swapped, because the masked gridline columns hide
a curve on about one row in seven and the tracker then locks onto its
neighbour. What worked instead was to keep only the rows where all five curves
are visible at once, so their left-to-right order identifies them with no
tracking at all, then fit each with a quadratic and reject the rows the fit
disowns. 143 to 149 of 153 complete rows survive per curve, with residuals of
0.002 to 0.006 in `K_int`, and the ordering holds at every tabulated point.

The example helicopter has `b_V/2R_T = 0.59`, essentially the leftmost printed
curve, and p. 509 reads `K_int = 0.4`. That curve gives 0.400 at a separation
ratio of 0.192, or `y_V = 2.0 ft`. Prouty prints no `y_V`, so this is a
consistency check rather than an anchor — but 2 ft between the tail rotor hub
and the fin centre is the right size for this aircraft, and since the
interference drag is three quarters of the fin drag (C8-6) the factor is worth
getting right: halving the separation raises `K_int` to 0.50 and the fin drag
from 58 to 69 lb.

---

## C8-14 — Table 8.7's stick positions disagree with Table 8.8 and with Figure A.5

**Printed.** Table 8.7 p. 529 gives, for the example helicopter at 115 knots:

```
Longitudinal cyclic pitch (corr.) B_1, deg    8.9      11.4
Stick position from full forward, %           34       29
```

Table 8.8 p. 530 gives, for the same helicopter at the same speed:

```
Cyclic pitch, B_1, deg                        8.5   8.8   9.4
Stick position, % from full forward           38    37    35
```

**They cannot both be right.** Stick position is a kinematic function of `B_1`
alone — Figure A.5 p. 682 is a straight line and nothing else enters it — so
the two tables should agree wherever they overlap. Level flight at 115 knots
appears in both: Table 8.7 puts `B_1 = 8.9 deg` at 34 %, Table 8.8 puts
`B_1 = 8.8 deg` at 37 %. Three points apart for a tenth of a degree.

The slopes disagree too. Table 8.7's two points imply -2.0 %/deg. Table 8.8's
three imply -3.33 %/deg. Figure A.5 spans `B_1 = -10` to +20.4 deg over the
full 100 %, which is -3.29 %/deg, and a fit to the printed line over 121
tracked columns gives -3.285 %/deg with a residual of 0.26 %.

| source | slope, %/deg | at `B_1 = 8.8 deg` |
|---|---|---|
| Figure A.5, fitted | -3.285 | 38.1 % |
| Table 8.8, three points | -3.33 | 37 % |
| Table 8.7, two points | -2.0 | 34 % at 8.9 |

**Decision.** Figure A.5 and Table 8.8 agree with each other to a constant
1.1 points, which is digitising precision on a chart drawn at 2.2 pixels per
per cent. `ControlPositionsComp` reproduces them, and Table 8.7's stick
positions are not used as anchors.

Nothing downstream depends on this: the finding of p. 529 is that the stick
moves *forward* in the descending turn, and it does so on any of the three
readings. But a reader calibrating a rigging model on Table 8.7 would build a
linkage 40 % too slack.

**Enforced by.**
`test_trim_chapter8.py::test_rigging_matches_table_88_stick_positions`.

---

## Digitisation of Figure A.5

All four panels are straight lines, so the figure reduces to four linear
relations and stores no table — the third chart of the chapter to dissolve
this way, after Figures 8.6 and 8.8.

```
pct_B1    = 67.03 - 3.2851 B_1              % from full forward, 10 in travel
pct_A1    = (A_1 + 10.333) / 0.17144        % from full left,     9 in
pct_pedal = (32.300 - th_75T) / 0.47790     % from full left,     5.5 in
pct_coll  = -3.735 + 5.6861 th_75M          % from full down,     12 in
```

| panel | residual | points | check |
|---|---|---|---|
| longitudinal cyclic | 0.26 % | 121 | Table 8.8, +1.1 points |
| lateral cyclic | 0.047 deg | 75 | Figure 8.31, 42.3 against 44 % |
| tail rotor collective | 0.069 deg | 51 | Figure 8.31, 55 % gives 6.0 deg |
| main rotor collective | 0.29 % | 63 | none printed |

Two details of the tracking. The longitudinal panel's abscissa is interrupted
by the vertical "% from Full Fwd" label, which makes the axis look broken and
the scale look inconsistent; measuring the tick stubs directly gives a uniform
7.73 pixels per degree across the gap, and the apparent break is the label, not
a scale change. The tail rotor panel needed the "5.5 inch" dimension arrow
masked out, at `th_75 = 2.5 deg`, before the line could be followed; without
it the tracker caught only fourteen columns at the far right.

---

## Wiring the charts into `TrimElementsGroup`

`TrimElementsGroup` takes a `charts` option. `False`, the default, leaves every
chart quantity a free input; `True` replaces seven of them with the digitised
figures and takes the geometry those figures are drawn against instead.

The default is `False` on purpose. Every anchor in this document was
established against Prouty's own readings of his own charts, and reproducing
his answers means using his numbers. `charts=True` is the honest configuration
for any *other* helicopter, where there is no printed reading to fall back on —
and its error bar is what the table below measures.

| quantity | Prouty reads | digitised | figure |
|---|---|---|---|
| `A_R_H` | 4.5 | 4.500 | 8.8, no end plates |
| `a_H` | 4.0 | 4.020 | 8.6 |
| `qH_q` | 0.6 | 0.600 | 8.9 and 8.10 |
| `K_int` | 0.4 | 0.400 | 8.22 |
| `A_R_V` | 3.2 | 3.341 | 8.19 |
| `a_V` | 3.0 | 3.359 | 8.6 at `A_R_V` |
| `delta_V` | 0.01 | 0.020 | 8.17, clamped at `A.R. = 4` |
| `delta_H` | 0.02 | 0.009 | 8.17, `lambda_H` not printed |

**Figure 8.10 is inert at 115 knots, and that is not luck.** `V/v1_hover` is
not an input: it is `2 sqrt(q / D.L.)`, in which the density cancels, so the
group computes it from the flight condition it already has. At 115 knots that
gives 4.97, past the 3.7 where Figure 8.10 returns to zero, so the wake
increment is exactly zero and `qH_q` is Figure 8.9's 0.6 unchanged. The chart
only bites in transition, where it nearly doubles the local dynamic pressure.

**The horizontal stabiliser survives; the fin does not.** `a_H` moves 0.5 % and
`L_H` moves less than 1 %. The fin compounds Figure 8.19 with Figure 8.6 —
`A_R_V` is 4.4 % high, and feeding that into the Helmbold relation on top of
its own 9.7 % open anchor makes `a_V` 12 % high — so `L_V` gains 12 %.

**What that costs the trim, measured.** Both groups solved at 115 knots:

| | readings | charts | book |
|---|---|---|---|
| `T_M` | 20,577.6 | 20,578.7 | 20,586 |
| `Theta` | -0.924 deg | -0.919 deg | -0.9 |
| `B_1` | 8.909 deg | 8.918 deg | 8.9 |
| longitudinal stick | 37.76 % | 37.74 % | 37 |
| `b1s_M` | -0.781 deg | -0.762 deg | -0.78 |
| `Phi` | -1.918 deg | -1.944 deg | -1.9 |
| `T_T` | 664.2 lb | 623.9 lb | 661 |
| lateral stick | 42.30 % | 42.41 % | 44 |

A 12 % stronger fin carries more of the antitorque, so the tail rotor thrust
falls 6 % — and that is the only observable that moves. Everything the pilot
sees stays where it was: the two stick positions shift by a tenth of a point
and the roll angle by three hundredths of a degree, less than the width of a
line on Figure 8.31.

That insensitivity is worth understanding rather than just noting. The trim
equations are a balance, not a chain: a stronger fin needs less tail rotor to
close the N equation, and the two changes very nearly cancel in the Y and R
equations that set the roll angle and the flapping. The 12 % error is real and
it lands on `T_T`, which is the quantity a designer sizes the tail rotor with.
It does not land on the handling qualities.

**Enforced by.**
`test_trim_chapter8.py::test_charts_on_reproduces_proutys_readings_to_a_few_per_cent`,
`::test_figure_810_is_inert_at_115_knots`,
`::test_charts_move_the_fin_and_leave_the_stabiliser_alone`,
`::test_charts_leave_the_longitudinal_trim_where_it_was`,
`::test_charts_cost_the_lateral_trim_six_per_cent_of_tail_rotor_thrust`.
