# Validation notes — `prouty.vertical`

Chapter 2, Aerodynamics of Vertical Flight, p. 93-117.

Each entry records what the book prints, what an independent derivation gives,
the decision taken, and the test that enforces it. C2-1 to C2-3 were spotted on
the first reading and settled when their groups were coded; C2-3 and C2-6 were
closed last, on evidence from the book itself (Sept 2026). No entry is open.

---

## C2-1 — Collective increment for 500 ft/min (G3, p. 101)

**Printed.** Δθ = (v_1c + V_c − v_1hov) / (0.75 ΩR) gives 0.4° at 500 ft/min.

**Check.** v_1hov = 39 ft/s, ΩR = 650 ft/s: v_1c + V_c − v_1hov = 4.39 ft/s,
Δθ = 0.0090 rad = 0.52°. Dividing by ΩR alone gives 0.39°, i.e. the printed
value: the 0.75 was dropped in the numerical evaluation, not in the equation.

**Decision.** `ClimbCollectiveComp` keeps the equation as printed, the
three-quarter station being its stated basis: 0.52° at 500 ft/min.

**Tests.** `tests/vertical/test_g2_g3_climb.py::test_collective_increment_500_fpm_p101`.

## C2-2 — Vibration onset in the vortex ring state (G5, p. 102)

**Printed.** 23 % to 125 % of v_1hov "would cover the region between 400 and
2,900 ft/min" for the example helicopter.

**Check.** v_1hov = 38.6 ft/s (20,000 lb, sea level, the value Figure 2.14
uses, C2-3): 0.23 × 38.6 × 60 = 532 ft/min, 1.25 × 38.6 × 60 = 2,893 ft/min.
The upper bound agrees and confirms 38.6 ft/s; the lower one is 25 % under
the percentage it quotes. 400 ft/min would be 17 % of v_1hov; no velocity of
the chapter (39, 40.4 ft/s) gives 400 at 23 %.

**Decision.** `VortexRingBoundariesComp` keeps the percentages (option
`rough_bounds = (0.23, 1.25)`): 532-2,893 ft/min for the example helicopter.

**Tests.** `tests/vertical/test_g5_vortex_ring.py::test_example_boundaries_p102`.

## C2-3 — Two hover induced velocities: 40.4 ft/s is the outlier (G7, pp. 101, 112, 114) — closed

**Printed.** 39 ft/s on p. 101, 40.4 ft/s on p. 112 for the example helicopter
"at normal gross weight and rotor speed", giving V_D = 1.97 × 40.4 = 4,780 ft/min.

**Check.** 20,000 lb at sea level gives 38.6 ft/s. 40.4 ft/s needs T/ρ 9.5 %
higher (21,900 lb at sea level, or 20,000 lb at ρ/ρ₀ = 0.913); the text states
neither. Every other number of the chapter uses 38.6 ft/s:

| evidence | page | implies |
|---|---|---|
| Figure 2.14 at 650 ft/s: 4,559 ft/min = 1.97 × v_1hov × 60 | 114 | 38.6 ft/s |
| vortex ring upper bound 2,900 ft/min = 1.25 × v_1hov × 60 | 102 | 38.7 ft/s |
| (V_c/2)² ≪ v_1hov², "(39)² = 1,521" | 101 | 39 ft/s |

A search of every project scan finds 40.4 only on p. 112, with no calculation
leading to it. The one candidate weight behind it, 21,900 lb, is contradicted
by Chapter 4 (C2-6).

**Decision (closed).** 40.4 ft/s is an inconsistent input of the p. 112
example. The model uses the stated conditions (20,000 lb, sea level,
v_1hov = 38.6 ft/s), which reproduce Figure 2.14. The 4,780 ft/min of the text
is kept as a test of the chart reading alone, with 40.4 ft/s supplied.

**Tests.** `test_figure_2_14_uses_20000_lb_at_sea_level`,
`test_figure_2_13_reading_p112`.

## C2-4 — Figure 2.13 does not join momentum theory at either end (G0, p. 113)

**Printed.** Figure 2.13 gives V̄_D against V̄_D − v̄₁ from wind tunnel tests,
between the momentum branches of pp. 94-95.

**Check.** Digitized on the p. 113 scan (calibration in
`examples/vertical/reproduce_figure_2_13.py`):

- at hover the curves start at V̄_D − v̄₁ ≈ −1.07 (θ₁ = 0°) and −1.085 (θ₁ = −12°),
  i.e. v̄₁ ≈ 1.08, not the momentum value 1.0;
- they stop at V̄_D = 2.6, still 0.35 (0°) to 0.6 (−12°) below windmill-brake
  momentum in V̄_D − v̄₁, with a slope (1.6-1.8) that closes the gap slowly.

**Decision.** `AxialInducedVelocityComp` blends the three branches with cubic
smoothsteps (decision: smooth blend, Sept 2026):

- `low_window = (0, 0.25)`: hover and climb are pure momentum, v̄₁(0) = 1; the
  chart takes over at a quarter of v_1hov, where p. 95 says momentum breaks down;
- `high_window = (2.6, 3.6)`: beyond its last point the chart is continued with
  its end slope (C1) and blended into windmill-brake momentum.

Both windows are options. θ₁ is interpolated linearly between the two curves.

**Anchors.** V̄_D = 1.975 at V̄_D − v̄₁ = 0.22, θ₁ = −10° (book 1.97, p. 112);
v_1hov = 38.6 ft/s (book 39, p. 101); momentum exact in climb and beyond 3.6.

**Observation.** The left "Simple Momentum Theory" dashed line of Figure 2.13
is drawn loosely, up to 0.14 off the formula near V̄_D = 2. The model uses the
formula.

**Tests.** `tests/vertical/test_g0_flow_states.py`.

## C2-5 — Tail rotor factor on the whole climb increment (G2, p. 98)

**Printed.** p. 98 writes Δh.p. = (1/550) {[climb] − [hover]} {1 + v_1hovT R_M / ((ΩR)_M l_T)}:
the braces put the tail rotor factor on the difference.

**Check.** Chapter 4 prints the same equation on p. 314 with the factor on the
hover bracket alone, which gives a non-zero Δh.p. at V_c = 0; C4-3 applied it to
the whole difference. Chapter 2 prints exactly that reading.

**Decision.** `ClimbPowerGroup` reuses `VerticalClimbPowerComp` of Chapter 4
unchanged; C4-3 is confirmed by the book itself.

**Tests.** `test_zero_increment_at_hover`, `test_figure_2_3_ah1g_calculated_line`.

## C2-6 — Figure 2.4 sits 4-10 % above the full equation: the figure is the outlier (G2, p. 100) — closed

**Printed.** Figure 2.4 gives the full-equation power increment of the example
helicopter up to 4,000 ft/min; its inputs are not listed. The text quotes
150 h.p. at 500 ft/min from the quick estimate at 20,000 lb.

**Check 1 — the equation.** Figure 2.3 (AH-1G, every input printed on the
figure, P_M = 870 hp for the tail rotor thrust): within 3 % above 600 ft/min,
within 4 hp below.

**Check 2 — Figure 2.4 with the Appendix A data** (20,000 lb, D_v/G.W. = 0.04,
l_T = 36.8 ft, P_M = 1,600 hp), Figure 2.4 over model:

| R/C, ft/min | Fig. 2.4, hp | Appendix A | D_v/G.W. = 0.06 | G.W. = T = 21,900 lb |
|---|---|---|---|---|
| 502 | 200 | 1.047 | 0.986 | 0.957 |
| 1,498 | 700 | 1.082 | 1.008 | 0.995 |
| 2,606 | 1,400 | 1.089 | 0.997 | 1.008 |
| 3,854 | 2,400 | 1.097 | 0.981 | 1.021 |

Density, tail rotor power or ΔA_z C_D close less than a third of the gap; two
single changes reproduce the curve, D_v/G.W. = 0.06 and 21,900 lb.

**Check 3 — Chapter 4, same equation, same helicopter, same author.** p. 314
repeats the equation of p. 98 and Figure 4.36 applies it at 20,000 lb.
`VerticalClimbGroup` (C4-28), with Chapter 4's own vertical drag
(D_v/G.W. = 0.043), reproduces it; at its excess power of 1,769 hp (sea level,
standard day, takeoff power):

| basis | rate of climb, ft/min | vs Figure 4.36 (3,270) |
|---|---|---|
| Appendix A data (Chapter 4 chain = G2) | 3,244 | −0.8 % |
| Figure 2.4 read backwards | 3,095 | −5.3 % |
| D_v/G.W. = 0.06 | 3,055 | −6.6 % |
| G.W. = 21,900 lb | 3,071 | −6.1 % |

Where Prouty uses the equation again, he gets what the model gets with the
Appendix A data. Figure 2.4 does not, and neither explanation of it survives
Figure 4.36.

**Decision (closed).** Figure 2.4 is inconsistent with the rest of the book;
the Appendix A data stay the reference, nothing is calibrated. The tests keep
the gap and the two rejected explanations visible.

**Tests.** `test_figure_2_4_example_gap_with_appendix_data`,
`test_figure_2_4_candidate_explanations`,
`tests/vertical/test_cross_check_chapter4.py`.

## C2-7 — Thrust damping: Chapter 9 already carries the same equation (G4, p. 102)

**Printed.** p. 102 differentiates C_T/σ = (a/4)[θ_T − V_c/(2ΩR) − √(C_T/2)] at
constant pitch: ∂(C_T/σ)/∂V_c = −1/{ΩR [8/a + √(σ/2)/√(C_T/σ)]}. No numerical
example.

**Check.** Table 9.1 (p. 564) uses the same expression for dC_T/σ/dλ, and
`BasicRotorDerivativesHoverComp` computes it: G4 in hover matches it to 1e-12
for both rotors' solidities, gives .490 for the example main rotor (book .49)
and dT/dV_c = −182 lb/(ft/s), the dZ/dż of Table 9.2 (p. 566). The small-climb
form is independent of V_c; the exact momentum form (option `'exact'`) is equal
at V_c = 0 and moves with the climb rate: .373 / .441 / .490 / .554 / .618 at
V_c = −20 / −8 / 0 / 10 / 20 ft/s for the example rotor.

**Decision.** `inflow='exact'` by default, `'small_climb'` reproduces p. 102 as
printed. The damping is evaluated at the thrust point of
`AxialThrustIdealTwistComp` with the same inflow, and a finite-difference test
checks that it is the slope of that thrust.

Chapter 9 link (decision, Sept 2026): `HoverDerivativesGroup(thrust_damping=
'external')` and `BasicRotorDerivativesHoverComp` take `dCT_sigma_dlambda_M` and
`_T` as inputs (Table 9.1 values .49 and .44 as defaults) instead of computing
them; `'table'` stays the default, so Chapter 9 is unchanged. Fed by G4 in
hover, the 44 rows of Table 9.4 are reproduced to 1e-12; in a climb only the
rows built on dCT/σ/dλ move, in the ratio of the exact g.

**Tests.** `tests/vertical/test_g4_thrust_damping.py`, including the Chapter 9 link.

## C2-8 — The autorotation parameter with the Chapter 6 drag (G7, p. 112)

**Printed.** (3/2)√3 / (√σ c̄_l^{3/2}/c_d) = 0.22 for the example helicopter;
c_d is not given.

**Check.** 20,000 lb, sea level, 650 ft/s: C_T/σ = 0.0829, c̄_l = 6 C_T/σ = 0.497.
0.22 implies c_d = 0.00865. The Chapter 6 NACA 0012 model (decision, Sept 2026)
at the 0.75 R Mach number (0.437) and α = c̄_l/a = 4.66° gives c_d = 0.0104,
hence 0.264. Figure 2.13 is flat there (dV̄_D/dy ≈ 0.3): V̄_D = 1.986 instead of
1.97, +0.8 % on the rate of descent. The same c_d = 0.0104 reproduces the "11 %"
of p. 115 (10.8 %), while c_d = 0.00865 would give 13 %.

**Decision.** Chapter 6 drag by default (`drag='airfoil'`); `drag='input'` takes
c_d directly and reproduces the printed 0.22.

**Tests.** `test_parameter_with_chapter6_drag`, `test_parameter_with_book_drag_p112`,
`test_extra_power_p115`.

## C2-9 — 40 hp raise the rate of descent by 0.4 %, not 1.5 % (G7, p. 115)

**Printed.** 40 hp increase V̄_D − v̄₁ by 11 % "but the rate of descent will
increase by only 1.5 %".

**Check.** The 11 % is reproduced. On Figure 2.13 at θ₁ = −10°, y = 0.22 → 0.244
moves V̄_D by 0.008 (slope dy/dV̄_D ≈ 3 on the digitized curves, 3.3 in the
model): +0.4 %. 1.5 % would need a slope of about 0.8, which the chart has
nowhere on its lower branch.

**Decision.** Chart as digitized; 0.4 % recorded.

**Tests.** `test_extra_power_p115`.

## C2-10 — Figure 2.14: level right, minimum at 470 instead of 550 ft/s (G7, p. 114)

**Printed.** Rate of descent and collective pitch against tip speed, minimum
rate of descent at about 550 ft/s, where c̄_l^{3/2}/c_d is largest.

**Check.** With a = 5.73 (Chapter 1) and the Chapter 6 drag, over 490-690 ft/s
the rate of descent is within 1 % and the collective within 0.25°. The Chapter 6
drag rises less steeply with c̄_l than the one behind the figure, so the
minimum moves to 470 ft/s; it is shallow (4,525 against 4,545 ft/min at
550 ft/s). A constant c_d = 0.00865 matches the figure above 590 ft/s and misses
the rise below. At 440 ft/s, α̅ exceeds the stall onset α_L = 8°: the warning
of p. 97 on the lower rotor speed limit.

**Decision.** Nothing tuned. `examples/vertical/reproduce_figure_2_14.py` plots
both drags against the digitized figure.

**Tests.** `test_figure_2_14_rate_of_descent_and_collective`,
`test_stall_warning_at_low_tip_speed`.

## C2-11 — The blade element in vertical climb (G1, pp. 95-97)

**Printed.** p. 96 extends the combined momentum and blade element method of
Chapter 1 with the climb velocity; no numerical example.

**Implementation.** `HoverRotorGroup(flight='climb')` (also on
`BladeElementGroup` and `InflowGroup`; default 'hover', Chapter 1 unchanged)
replaces step 5 by `ClimbInflowRatioComp`, whose v1_Or is the total inflow
ratio (V_c + v1)/(Ωr): the angle of attack and the torque of steps 6 and 14 then
include the climb ("inflow drag", p. 95). The empirical corrections of steps
16-20 are hover fits and are applied unchanged.

**Check.** At V_c = 0 the Figure 1.45 case is reproduced to 1e-12. Trimmed at
20,000 lb, sea level:

| V_c, ft/s | Δθ₀ model | Δθ₀ p. 101 (G3) | ΔP model, hp | G.W.(v₁c+V_c−v₁hov)/550 |
|---|---|---|---|---|
| 8.33 | 0.55° | 0.52° | 177 | 160 |
| 16.67 | 1.16° | 1.08° | 372 | 336 |
| 33.3 | 2.52° | 2.36° | 813 | 731 |

The mean angle of attack outboard of 0.5 R moves by 0.04° while the pitch goes
up 0.55° at 500 ft/min, as p. 96 states. The blade element power increment is
a steady 10-12 % above the momentum one, the induced power factor that the
combined method carries through its tip loss and non-uniform inflow. The
collective increment supports C2-1: 0.55°, not 0.4°.

**Tests.** `tests/vertical/test_g1_blade_element.py`.

## C2-12 — Tail rotor vortex ring state and drive torque (G6, G8, pp. 107-116)

**Printed.** p. 108: the curves of Figure 2.7 are unstable between velocity
ratios 0.4 and 0.8; the UH-1 tail rotor (v_1hov = 48 ft/s) should meet its
maximum vortex ring effect at 70 % of that, about 20 kt, and meets it earlier
because of the main rotor wake. p. 116: the tail rotor drive is designed for a
moderate multiple of the maximum hover torque, for example twice.

**Implementation.** `TailRotorVortexRingGroup`: the axial velocity
V_D_T = V_y_T + r l_T (sideward flight and hover turn, same state, p. 107) and
a second instance of the G5 component with `rough_bounds = (0.4, 0.8)`, outputs
suffixed `_T`. `TailRotorDriveTorqueComp`: Q_T_design = k Q_T_hov_max, k = 2.
The main rotor wake interaction (Figures 2.9-2.12) and the transient of
Figure 2.15 are test data of particular aircraft and are not modelled
(decision, Sept 2026).

**Check.** 0.707 × 48 ft/s = 20.1 kt (book: about 20 kt). v_hov_T links from
G2 (`tail_rotor='hover_power'`) by promotion.

**Tests.** `tests/vertical/test_g6_g8_tail_rotor.py`.
