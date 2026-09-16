# Validation notes — `prouty.flapping`

Chapter 7, Rotor Flapping Characteristics, p. 455-479.

Each entry records what the book prints, what an independent derivation gives,
the decision taken, and the test that enforces it.

---

## C7-1 — Missing `(1 - e/R)^4` in the cross-coupling of p. 461

**Printed.** p. 460 derives the acceleration cross-coupling as

```
b1s / a1s = -(12/gamma)(e/R) / { [1 + (1/3)(e/R)] [1 - e/R]^4 }
```

p. 461 then reuses it as

```
b1s = -(12/gamma)(e/R) / [1 + (1/3)(e/R)] * a1s
```

dropping the `(1 - e/R)^4` factor. The numbers quoted on both pages
(`b1s/a1s = -0.07`, `Delta_alpha/a1s = 0.07`) follow the reduced form.

**Independent check.** Three routes agree on the complete form for the example
helicopter (gamma = 8.1, e/R = 0.05):

| Route | b1s/a1s |
|---|---|
| p. 460 closed form | -0.089453 |
| `-cot(phi)` from nu = 1.038724, zeta = 0.424829 | -0.089452 |
| Full chain from `I_b = 2870` through `HoverFlappingGroup` | -0.089967 |

The third differs only because `I_b = 2870` gives gamma = 8.054 rather than
8.1 (see C7-5). The reduced form gives -0.072860, and

```
complete / reduced = (1 - e/R)^-4 = 1.227737
```

exactly, which localises the discrepancy to that single factor and rules out
an error elsewhere in the chain.

**Decision.** Implement the complete form of p. 460. The printed -0.07 / 0.07
are recorded as *reduced-form* anchors, not as targets.

**Enforced by.** `test_accel_coupling_comp.py::test_matches_closed_form_of_page_460`,
`::test_printed_values_come_from_the_reduced_form`,
`test_hover_flapping_group.py::test_cross_coupling_uses_the_complete_form`.

---

## C7-2 — Sign slip on the first `Delta_alpha` line, p. 461

**Printed.** p. 461 gives, in two consecutive lines,

```
Delta_alpha = beta_dot r' / [Omega (r' + e)]          (line 1)
Delta_alpha = -b1s r' / (r' + e)     at psi = 180 deg (line 2)
```

**Why they conflict.** With `beta = a0 - a1s cos(psi) - b1s sin(psi)` (p. 464),
`beta_dot = Omega (a1s sin(psi) - b1s cos(psi))`, so at `psi = 180 deg`,
`beta_dot = +Omega b1s`. Line 1 then yields `+b1s r'/(r' + e)`, the opposite of
line 2.

**Which one is right.** The `U_P` of p. 464 carries the flapping term as
`-(r'/R)(a1s sin(psi) - b1s cos(psi))`, and `alpha = theta + U_P/U_T` with
`U_T = Omega(r' + e)`. The flapping contribution to the angle of attack is
therefore `-r' beta_dot / [Omega (r' + e)]`: line 1 is missing a minus sign,
line 2 is correct. Line 2 is also the one consistent with the positive 0.07
quoted immediately below it.

**Decision.** Implement line 2.

**Enforced by.** `test_accel_coupling_comp.py::test_station_factor` and the
sign of `dalpha_over_a1s` in `::test_complete_form_anchor`.

---

## C7-3 — `v1/(Omega R)` coefficient, p. 466 vs p. 468 *(affects G2)*

**Printed.** The cosine component of the aerodynamic hinge moment, p. 466,
carries

```
- (v1/Omega R) [1/4 - (1/6)(e/R) - (1/12)(e/R)^2]
```

The cosine equation assembled from it on p. 468 carries `-(1/3)(v1/Omega R)`.

**Independent check.** Setting `e/R = 0` in the p. 466 bracket and combining
with `M_CF,cosine = Omega^2 a1s e M_b/g` gives

```
A1 - b1s = -[(4/3) mu a0 + v1/Omega R] / (1 + mu^2/2)
```

which is exactly the Chapter 3 relation Prouty himself quotes on p. 474. The
p. 468 coefficient of 1/3 would instead give `(4/3) v1/Omega R`.

The same 4/3 propagates into the closed-form `b1s` of p. 469, whose
`sigma/(2 mu)` term should carry a coefficient of 1.

**Decision.** p. 466 is authoritative. Applied on both routes: the 3x3 system
is built from pp. 465-466 and never sees the 1/3, and the closed forms use a
coefficient of 1 on `v_1/(Omega R)` inside `N_2`.

**Enforced by.**
`test_flapping_matrix_comp.py::test_lateral_flapping_matches_the_chapter_3_relation`,
which solves the system at `e/R = 0` and recovers
`A_1 - b_1s = -[(4/3) mu a_0 + v_1] / (1 + mu^2/2)` to 1e-12, then checks that
the 1/3 version would be measurably off.

---

## C7-4 — `mu^4/4` sign in the simplified flapping equations, p. 469 *(affects G2)*

**Printed.** `a1s` carries `(1 + mu^4/4)` in the denominator of its
cross-coupling term; `b1s`, three lines below, carries `(1 - mu^4/4)`.

**Independent check.** Both terms arise from the product
`(1 - mu^2/2)(1 + mu^2/2) = 1 - mu^4/4`. The two should be identical.

**Second, independent confirmation.** The same 2x2 structure reappears for
flapping due to pitch and roll rates (p. 473), with the same `kappa` and the
same denominators. There Prouty prints `(1 - mu^4/4)` on *both* `a_1s` and
`b_1s`. The p. 469 `a_1s` is the odd one out.

**Decision.** Use `(1 - mu^4/4)` in both. Print inconsistency, documented not
reproduced.

**Enforced by.**
`test_closed_form_flapping_comp.py::test_matches_printed_page_469_expressions`
and `::test_printed_mu4_sign_shifts_a1s_by_a_known_amount`, which checks the
gap equals `kappa N_2 [1/(1 - mu^4/4) - 1/(1 + mu^4/4)]` at three advance
ratios. The gap grows as `mu^4` and happens to vanish at `mu = 0.3` for the
reference trim, because `N_2` is near zero there; the identity holds
regardless.

---

## C7-5 — `(1 - e/R)^4` grouping on p. 458 is *not* an inconsistency

Recorded so it is not re-opened. The damping of p. 458 is printed as

```
c = (I_b gamma Omega / 8) (1 - e/R)^4 [(1 + e/3R) / (1 - e/R)]
```

Direct integration of the hinge moment on the same page gives

```
c = (rho a c_chord Omega / 2) int_0^(R-e) r'^2 (r' + e) dr'
  = (rho a c_chord R^4 Omega / 8) (1 - e/R)^3 (1 + e/3R)
```

and `(1 - e/R)^4 / (1 - e/R) = (1 - e/R)^3`. The two are the same expression.
The code uses the collapsed form.

**Enforced by.** `test_flap_damping_comp.py::test_damping_matches_quadrature_of_hinge_moment`,
`::test_collapsed_shape_factor`.

---

## C7-6 — Which first static moment the rotor stiffness uses, p. 477

**Printed.** p. 477 presents two uniform-mass expressions as equivalent:

```
dM_M/da1s = (1/4)(e/R) b m R (Omega R)^2
          = (3/4)(e/R) A_b rho R (Omega R)^2 a / gamma
```

**Independent check.** The derivation on the same page gives, with no
assumption on the mass distribution,

```
M_M = (1/2) e b Omega^2 a1s M_b/g   ->   dM_M/da1s = (1/2) e b Omega^2 (M_b/g)
```

The two printed shortcuts follow from it, but only with `M_b/g = m R^2/2` and
`I_b = m R^3/3`, that is with both moments taken from the *centre of
rotation*. p. 456 defines them about the *flapping hinge*, which carries the
`(1 - e/R)^2` and `(1 - e/R)^3` factors of p. 457.

So the discrepancy is not about `I_b`, as first suspected, but about the
first static moment. The ratio between the two conventions is exactly
`1/(1 - e/R)`:

| dM_M/da1s | ft-lb/rad |
|---|---|
| hinge convention, from the derivation | 212,732 |
| centre convention | 202,096 |
| as printed on p. 477 | 200,940 |

The last two differ only by the `gamma = 8.1` against `I_b = 2870` rounding
of C7-7.

**Decision.** `RotorStiffnessComp` implements
`dM_M/da1s = (1/2) e b Omega^2 (M_b/g)` and inherits `e` and `M_b/g` from
`BladeInertiaComp`, so it stays exact for any mass distribution.
`convention='center'` reproduces p. 477, written as
`(3/4)(e/R) b Omega^2 I_b`, which is the printed `A_b` form rearranged since
`c rho a R^4 = gamma I_b`.

The inversion of p. 477 is consistent with the centre convention only. With
`K' = (4/3)(dM_M/da1s)/(b Omega^2 I_b)`, the centre form gives
`(e/R)_eff = K'` as printed, and the hinge form gives `K'/(1 + K')`.

**Enforced by.** `test_rotor_stiffness_comp.py::test_hinge_convention_follows_the_derivation`,
`::test_center_convention_equals_the_printed_A_b_form`,
`::test_center_convention_equals_the_printed_linear_mass_form`,
`::test_the_gap_between_conventions_is_the_static_moment`, which checks the
ratio equals `1/(1 - e/R)` exactly, and `::test_round_trip`.

---

## C7-7 — Appendix A internal rounding

`I_b = 2870 slug ft^2` and `gamma = 8.1` are both quoted for the example
helicopter but are not quite consistent: with `c = 2 ft`, `rho = 0.002378`,
`a = 6 /rad`, `R = 30 ft`,

```
I_b = 2870   ->  gamma = 8.054
gamma = 8.1  ->  I_b   = 2853.6
```

a 0.6 % gap. `I_b` is treated as the primary input, since it is a measured
blade property while `gamma` is derived. Book anchors are given tolerances
that absorb this plus the two-decimal printing.

**Enforced by.** `test_lock_number_comp.py::test_example_helicopter_anchor`.

---

## C7-8 — Dropping `e/R` from the brackets is not harmless for coning

**Printed.** p. 466: "For most rotors, the hinge offset ratio, e/R, will be
small enough that it can reasonably be eliminated from those terms inside the
square brackets." Everything from p. 467 onward follows that.

**What it costs.** True for `a_1s` and `b_1s`, whose brackets are dominated by
a single term. Not true for coning: the constant bracket is a *difference* of
comparable contributions, so a 3 % change in each moves the sum by far more.
For `theta_1 = -10 deg`, `alpha_s = -5 deg`, `mu = 0.3`:

| e/R | theta_0 | change in `C_const` |
|---|---|---|
| 0.05 | 10 deg | -41 % |
| 0.05 | 14 deg | +20 % |
| 0.05 | 18 deg | +10 % |
| 0.10 | 14 deg | +41 % |

End to end at `e/R = 0.05`, the two routes of `ForwardFlightFlappingGroup`
differ by up to 0.9 deg on `a_1s` at `mu = 0.45`, and by less than 0.3 deg on
`b_1s`.

**Decision.** The numerical route keeps the full brackets of pp. 465-466 and
is the default. The closed forms are kept as an audit path, not for
production.

**Enforced by.**
`test_flapping_matrix_comp.py::test_dropping_e_over_R_from_brackets_costs_a_lot_on_coning`,
`test_forward_flight_flapping_group.py::test_methods_diverge_once_the_offset_is_real`.

---

## C7-9 — Coning factor inside `N_2`, p. 469

**Printed.** The coning contribution to the numerator of the cross-coupling
term is written `[(2/3) mu gamma / a] / (1 + (3/2) e/R)`, whereas the coning
of p. 467 carries `(1 - e/R)^2 / (1 + e/2R)`.

**Independent check.** To first order,
`(1 - x)^2 / (1 + x/2) = 1 - (5/2) x + O(x^2)`, so the consistent short form
would be `1 / (1 + (5/2) e/R)`, not `1 / (1 + (3/2) e/R)`. At `e/R = 0.05`
the printed factor is 5.6 % high (0.9302 against 0.8805).

**Decision.** `ClosedFormFlappingComp` builds `N_2` from the actual `a_0` it
computes, so the inconsistency does not arise. Minor: it moves `b_1s` by well
under a tenth of a degree.

**Enforced by.**
`test_closed_form_flapping_comp.py::test_printed_coning_factor_in_N2_is_higher`.

---

## C7-10 — Hover numerator with a forward-flight denominator, p. 475

**Printed.** The longitudinal cyclic needed to suppress the flapping from a
pitch rate is written

```
Delta_B_1 = [1 / (da_1s/dB_1)] (16/gamma) (q/Omega)
da_1s/dB_1 = -(1 + (3/2) mu^2) / (1 - mu^2/2)
```

**The mismatch.** `(16/gamma)(q/Omega)` is the *hover*, zero-offset value of
`-a_1s,rate`, the special case Prouty highlights at the bottom of p. 473. The
general expression on the same page carries a denominator:

```
a_1s,rate = -(16/gamma)(q/Omega) / (1 - mu^2/2)      at e/R = 0
```

The sensitivity, by contrast, is the full forward-flight one. Combining a
hover numerator with a forward-flight denominator leaves out a factor of
`1/(1 - mu^2/2)`:

| mu | shortfall |
|---|---|
| 0.15 | 1.1 % |
| 0.30 | 4.7 % |
| 0.45 | 11.3 % |

At the worked example, `mu = 0.3`, the consistent answer is -0.644 deg rather
than the -0.615 deg the printed formula gives (quoted as -0.62).

**Decision.** `LongitudinalCyclicTurnComp` takes `a_1s_rate` as an input
rather than rebuilding the numerator, so both halves are evaluated at the
same flight condition. The printed result is still reproducible by feeding
the hover value.

**Enforced by.**
`test_longitudinal_cyclic_turn_comp.py::test_printed_numerator_is_the_hover_value`,
which checks the ratio equals `1/(1 - mu^2/2)` exactly at three advance
ratios, and `::test_printed_formula_reproduced`.

---

## C7-11 — The step to `C_T/sigma + (a/8) lambda'`, p. 479

**Printed.** Differentiating the Chapter 3 `C_H/sigma` equation gives

```
d(C_H/sigma)/da_1s = (a/8) lambda'
                     + (a/4)[(2/3 + mu^2) theta_0 + (1/2 + mu^2/2) theta_1
                             + lambda' - mu (B_1 + 2 a_1s)]
```

and p. 479 then says the second term "for all practical purposes, is the
equation for `C_T/sigma`", giving

```
d(C_H/sigma)/da_1s = C_T/sigma + (a/8) lambda'
```

**Independent check.** With `lambda = lambda' - mu (B_1 + a_1s)` from p. 166,
the exact difference between the two is `-(a/4) mu a_1s`. At 115 kt with
`a = 6`, `mu = 0.3` and `a_1s = 2.87 deg` that is -0.0225 against a retained
value of 0.0679: a third of the answer, not a rounding.

**Decision.** The short form is implemented, because the p. 479 table follows
it. Checked by working the table backwards: the long form would put
`lambda'` at +0.010 at 115 kt, positive, which contradicts the same page's
statement that `lambda'` grows more negative with speed.

The long form is not offered as an option, because the book gives the full
expression only for the H-force; adding it would mean inventing the Y-force
counterpart.

**Enforced by.**
`test_h_force_flapping_deriv_comp.py::test_the_dropped_term_is_a_third_of_the_answer`.

---

## What can and cannot be checked in the p. 479 table

The table gives the c.g. moment per radian of flapping in three flight
conditions, without stating the trim that produced them:

| Condition | Rotor force | Implied `d(C_H/sigma)/da_1s` | Implied `lambda'` | Implied `alpha_TPP` |
|---|---|---|---|---|
| Hover | 73,500 | 0.04059 | -0.05632 | — |
| 115 kt | 123,000 | 0.06792 | -0.01988 | -1.55 deg |
| 160 kt | 108,000 | 0.05964 | -0.03092 | -3.10 deg |

The implied tip path plane angles are plausible and steepen with speed, so
the table is internally consistent. Only the hover row can be reproduced
independently, since `lambda' = -sqrt(C_T/2)` there with no trim needed. It
gives 69,400 against 73,500, 5.6 % low, which corresponds to an induced
velocity about 5 % below the momentum value — the kind of correction the
Chapter 1 hover analysis applies through tip loss and profile effects.

The non-monotonic trend across the three rows is physical: the induced
velocity collapses with speed, raising `lambda'`, while the disc pitches
further nose down, lowering it. The first wins to about 115 kt.

**Enforced by.**
`test_flapping_moment_buildup_comp.py::test_hover_entry_from_first_principles`,
`::test_implied_tip_path_plane_angles_are_plausible`,
`::test_the_rotor_force_contribution_peaks_around_115_knots`.

---

## Book anchors, Chapter 7 hover section

Computed with `HoverFlappingGroup`, `blade_input='I_b'`, `I_b = 2870`,
`R = 30 ft`, `c = 2 ft`, `e/R = 0.05`, `rho = 0.002378`, `a = 6 /rad`,
`Omega R = 650 ft/s`, sea level.

| Quantity | Page | Book | Computed | Note |
|---|---|---|---|---|
| `omega_n/Omega` | 457 | 1.04 | 1.0387 | two-decimal printing |
| `zeta` | 459 | 0.42 | 0.4224 | printing + C7-7 |
| `phi` | 459 | 84.8 deg | 84.86 deg | read off Figure 7.3 |
| `psi_63` | 462 | 130 deg | 130.59 deg | C7-7 |
| `b1s/a1s` | 460 | -0.07 | -0.0900 | **C7-1**, reduced form printed |
| `Delta_alpha/a1s` | 461 | 0.07 | 0.0840 | **C7-1** |
| `(e/R)_eff`, AH-56A | 477 | ~0.13 | 0.13 at nu = 1.106 | order of magnitude only |

---

## Conventions adopted in G1

- Blade properties (`I_b`, `M_b/g`, `e`, `omega_n/Omega`) are scalars; they do
  not depend on the flight condition.
- `gamma` and everything downstream of it are `(num_nodes,)`: the Lock number
  varies with density. Freezing it at sea level would misstate the damping
  ratio by 26 % at 10,000 ft.
- `a` is declared in `1/rad` on every input that consumes it.
- Where the book gives both a general and a uniform-mass expression, only the
  general one is coded; the uniform-mass closed form becomes a test. This
  keeps a single code path and leaves the components valid for a non-uniform
  blade.
- `c_damp` / `c_crit` rather than `c`, following Prouty's own warning on
  p. 458 that italic *c* is the chord and roman c the damping.

## Conventions adopted in G2

- The induced-velocity loop of pp. 467-468 is linear, so it is closed
  algebraically rather than with a solver. It is written as
  `C_T/sigma = 2 mu K S / P`, `v_1 = sigma K S / P` with `P = 2 mu + K sigma`,
  which is the same expression as the printed substitution but has no
  division by `mu` and therefore stays finite at zero advance ratio. The
  value there is still meaningless: the p. 468 inflow is the high-speed
  momentum result.
- The whole group is feed-forward. The 3x3 flapping system is solved by an
  explicit component with analytic `dx/db = A^-1` and
  `dx_k/dA_ij = -(A^-1)_ki x_j`, rather than by `om.LinearSystemComp`, which
  is implicit and collapses the leading dimension at `vec_size=1`.
- Every equation is scaled by `Omega^2 I_b`. The system matrix is then
  dimensionless with a condition number between 1.2 and 1.8 across
  `mu` in [0, 0.45] and `e/R` in [0, 0.13].
- `v1_over_OmegaR` is an input to both flapping routes rather than being
  rebuilt from `(C_T/sigma) sigma / (2 mu)`, so either can be driven by a
  better inflow model.
- The `(1 - e/R)` prefactor of `C_T/sigma` on p. 467 was checked symbolically:
  the thrust integral over the hinge-to-tip span is
  `(R^3/3)(1 - e/R)[1 + e/R + (e/R)^2]`, and the moment integral is
  `(R^4/4)(1 - e/R)^2 [1 + (2/3)(e/R) + (1/3)(e/R)^2]`, which is exactly the
  `theta_0` bracket printed on p. 465. Both prefactors are correct; only the
  brackets are approximated, see C7-8.
