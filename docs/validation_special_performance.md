# Validation notes — `prouty.special_performance`

Chapter 5, Special Performance Problems, p. 339-377.

Each entry records what the book prints, what an independent derivation gives,
the decision taken, and the test that enforces it. C5-1 to C5-3 were spotted on
the first reading; their groups are not coded yet.

---

## C5-1 — Drive system polar inertia (G2a, p. 348)

**Printed.** J = J_M + (Ω_T/Ω_M) J_T + J_trans; 11,735 slug·ft² for the example.

**Check.** Referring the tail rotor energy to main rotor speed needs the square
of the speed ratio: J = 11,600 + (100/21.67)² 25 + 20 ≈ 12,150 slug·ft².

**Decision.** Squared ratio by default; `DriveInertiaComp(inertia_ratio='book')`
for the printed form. With either, t_KE = 1.25 s at 4,000 hp (printed 1.2 s).

**Tests.** `tests/special_performance/test_g2a_g2f_rotor_energy.py::test_anchor_p348_inertia_book_and_coherent`.

## C5-2 — Time for a 180° turn (G1, p. 343)

**Printed.** t_180 = (π/2) V/(g √(n²−1)); 656 hp and 230 hp for a 1.5 g turn
from 115 to 100 kt losing 50 ft.

**Check.** The turn rate is g √(n²−1)/V, so 180° (π rad) takes π V/(g √(n²−1)):
the printed time is that of a 90° turn. Both printed powers follow from it;
the coherent form gives half, 328 hp and 115 hp.

**Decision.** `TurnEnergyPowerComp(turn_time='coherent')` by default,
`turn_time='book'` reproduces the printed numbers.

**Tests.** `tests/special_performance/test_g1_turns.py::test_anchor_p343_energy_power_book`,
`::test_energy_power_book_option_is_twice_coherent`.

## C5-3 — Low hover height, multi-engine (G2d, p. 357)

**Printed.** h_lo = V_LG J Ω₀² [1 − √((C_W/σ)/0.2)] / [1,100 (hp_IGE − hp_avail)].

**Check.** The single-engine form (p. 354) is linear in (C_W/σ)/0.2, which is
what the kinetic energy balance gives: thrust ∝ Ω², so Ω_f²/Ω₀² = (C_W/σ)/0.2.

**Decision.** Linear form by default; `LowHoverHeightComp(engines='multi', book=True)`
for the square root.

**Tests.** `tests/special_performance/test_g2d_height_velocity.py::test_c5_3_book_option`.

## C5-4 — Pitch rate in the example turn (G1, p. 342)

**Printed.** 0.08 rad/s at 115 kt and n = 1.2.

**Check.** Θ̇ = (g/V)(n²−1)/n = (32.2/194.1)(0.44/1.2) = 0.061 rad/s.

**Decision.** The relation is kept; the printed number is an example value only.

**Tests.** `::test_anchor_p342_pitch_rate_and_relief`.

## C5-5 — Cyclic relief in the example turn (G1, p. 342)

**Printed.** ΔB₁ = 0.2°.

**Check.** ΔB₁ = 16 Θ̇/(γΩ) with γ = 8.1, Ω = 21.67 rad/s (Appendix A):
0.32° with 0.061 rad/s, 0.42° with the printed 0.08 rad/s. 0.2° corresponds to
a factor 8 instead of 16. Chapter 7 p. 473 confirms 16/γ (hover, e = 0).

**Decision.** 16/γ kept; the printed number is an example value only.

**Tests.** `::test_anchor_p342_pitch_rate_and_relief`.

## C5-6 — Power required in a steady turn (G1, p. 343)

**Printed.** 3,170 hp in a 1.2 g turn at 115 kt, against 1,470 hp in level
flight, both from Figure 4.38 at the effective weight n·GW.

**Check.** Figure 4.38 read at 115 kt: the 20,000 lb curve gives about 1,470 hp,
the 24,000 lb curve about 3,200 hp, where it crosses the upper stall limit
(ΔC_Q/σ₀ = 0.008 at the critical azimuth, about 2,300 hp at the rotor). The
printed pair is therefore consistent with the figure; the rise comes from the
chart stall torque increment. The Chapter 4 chain has no such increment:

| model | 20,000 lb | 24,000 lb | ratio |
|---|---|---|---|
| closed-form trim (default) | 1,172 hp | 1,301 hp | 1.11 |
| numerical rotor, DiscAirfoilGroup, 12×15 and 24×21 | 1,206 hp | 1,433 hp | 1.19 |
| Figure 4.38 (book) | 1,470 hp | 3,170 hp | 2.16 |

C_T/σ = 0.102 at μ = 0.30 lies inside the level-flight band of Figure 5.2
(0.097-0.123): the book itself places this turn at the level-flight stall
boundary, where the steep power rise lives.

**Decision.** `SteadyTurnPowerGroup` is the Chapter 4 chain at n·GW as p. 343
prescribes; the gap is documented, not tuned. Near the Figure 5.2 boundaries
the turn power is optimistic; `ThrustCapabilityComp.n_margin` flags them.

**Tests.** `::test_turn_power_is_level_power_at_effective_weight`,
`::test_turn_power_below_print_c5_6`.

**Status: correction designed, paused (Sept 2026).** Target is the chart method
of Chapter 3, not the 3,170 hp of Figure 4.38.

- Chart method at 24,000 lb, mu = 0.30 (p. 229 uses the top chart of the first
  plate for collective): C_T/sigma = 0.0996, X = 0.187, theta_0 = 16° on the X
  chart of p. 262 (read by the user), C_Q/sigma ≈ 0.009 on the torque chart,
  against 0.0035 at 20,000 lb. The inflow chart of p. 263 gives 18° at
  lambda' = −0.0253: the two plates disagree near stall; p. 229 prescribes p. 262.
- Twist (p. 230): the charts are for −5°; stall limit lines and torque curve
  knees move by ΔC_T/sigma ≈ −0.003 (theta_1 + 5°), i.e. +0.015 for the example
  (−10°). Our numerical rotor stalls hard on the chart rotor and barely on the
  example rotor, as this shift predicts.
- Validated design: `StallTorqueIncrementComp`,
  ΔC_Q/sigma(mu, C_T/sigma_eff, X) = C_Q/sigma chart − C_Q/sigma closed form on
  the chart rotor, smooth floor at 0, Akima 3-D table from the X and C_Q/sigma
  charts of pp. 258-266 (mu = 0.20-0.40, label-anchored, user spot checks);
  C_T/sigma_eff = C_T/sigma + 0.003 (theta_1 + 5°) − shift for an airfoil
  stall angle Δalpha_stall (input, default 0, p. 230); ΔP added to main rotor
  power before the drive losses; `ForwardFlightPowerGroup(stall=False)`.
- Expected: no change at 20,000 lb (C_T/sigma_eff = 0.068); roughly +250 to
  +300 hp at the rotor at 24,000 lb.

## C5-7 — Figure reference for the zoom power (G2c, p. 352)

**Printed.** "The power required was taken from Figure 4.24."

**Check.** Figure 4.24 is the hub-pylon interference drag (p. 295). The power
required curve of the example is Figure 4.38; back-solving Figure 5.6 gives
about 3,900 hp at 160 kt, which is the 20,000 lb curve of Figure 4.38.

**Decision.** Reference only; P_0 and P_1 are inputs fed by the Chapter 4 chain.

## C5-8 — Flare angle of the example (G2e, p. 362)

**Printed.** θ̇_max = 88 deg/s, Δt = 1.25 s, α_TPP,max = 100 deg (use 45).

**Check.** θ̇_max Δt = 88 × 1.25 = 110 deg. No consequence: both exceed the
45 deg limit.

**Decision.** Relation kept; example value only.

---

## G1 — Turns and pullups (pp. 340-346)

- Load factor relations cross-checked between modes (bank, turn rate, pitch rate);
  Θ̇ = ω sin Φ and R ω = V verified.
- Figure 5.2: both edges of each shaded band digitized on a grid overlay of the
  scan. `band_fraction` (default 0.5) places the boundary in the band; for
  'level', 0 = low drag, 1 = high drag. Transient upper edge ≈ 0.17 for
  μ = 0.1-0.3 (p. 344); level edges 0.077 / 0.038 at μ = 0.5 match the side
  labels; transient ≥ steady turns ≥ level everywhere.
- Figure 5.3 (test data) is not implemented.

## G2a — Rotor speed decay (pp. 348-350)

- J: see C5-1. t_KE = ½JΩ₀²/(550 hp₀) = 1.25 s at 4,000 hp (printed 1.2 s).
- Ω/Ω₀ = 1/(1 + f t/2t_KE): 30 % lost in the first second (both engines,
  f = 1), 17 % with one engine out (f = ½), as printed p. 350. Figure 5.4 is
  this closed form, not digitized; the closed form is checked against a
  numerical integration of the decay equation.

## G2f — Autorotative indices (pp. 363-364)

- AI = (JΩ²/GW)(ρ/ρ₀)/D.L. = 39.0 ft³/lb for the example, as printed.
- t_equiv: the printed 0.8 s needs the example's hover OGE power and
  (C_T/σ)_max, which come from Chapters 1 and 4. Checked against t_KE, and
  jointly with the flare time of p. 362 (1.25 s): both printed times are met by
  (C_T/σ)_max = 0.1406 and hp_OGE = 1,643 hp with J = 11,735 slug ft² — values
  to compare with Chapters 1 and 4 when G2 is linked.
- Figure 5.13 (pilot opinion) is not implemented.

## G2b — Steady descent in autorotation (pp. 350-351)

- R/D(V) from the Chapter 3 trim in `mode='autorotation'` (closed-form rotor,
  losses of p. 197), L/D = V/(R/D). Trim residuals below 1e-10.
- Figure 5.5 (digitized): within 10 % from 70 to 140 kt; the chain sits 1-4 %
  low up to 100 kt and 7-10 % low at 120-140 kt.
- Best speeds by Newton on V around the trim, central-difference stencil
  (±2 kt): minimum rate of descent 81 kt, 1,573 ft/min (bottom of Figure 5.5
  at about 80-85 kt); minimum descent angle 125 kt, L/D = 6.59 against the
  printed 6.3 (+5 %), at the tangent speed marked on Figure 5.5.
- dV/dGW through the nested Newton checked against finite differences (2 %).

## G2c — Zoom and glide distance (pp. 351-352)

- γ_c = arccos[(C_W/σ)/(C_T/σ)_max] with the conservative 0.12 (p. 352).
- Figure 5.6 (failure at 160 kt) digitized on the scan: Δh and Δd at
  V_1 = 70-140 kt. Their ratio Δd/Δh = V_1/(R/D) matches Figure 5.5, also
  digitized, within 5 % at every speed: the glide relation is anchored.
- Δh: rotor kinetic energy excluded as p. 352 advises. Back-solving Figure 5.6
  gives hp_0 + hp_1 from 5,000 hp (V_1 = 70 kt) to 7,500 hp (150 kt), i.e.
  about 3,900 hp at 160 kt (C5-7). The printed Δh is not reproduced with the
  Chapter 4 chain: P_0, P_1 (Chapter 4) and R/D (G2b) are inputs of
  ZoomGlideGroup.
- ZoomGlideChainGroup connects them: G2b and Chapter 4 G7 run on the same
  speeds [V_1, V_0] so per-node design inputs share one shape. Failure at
  160 kt: the extra glide distance peaks at 87-90 kt, as on Figure 5.6.
  Δh is 40-60 % above the print because the chain needs 1,876 hp at 160 kt
  where Figure 5.6 implies about 3,900 hp (C4-29, C5-7): the gap is the
  Chapter 4 power curve, not the zoom relation.

## G2d — Height-velocity diagram (pp. 352-358)

- Figures 5.8 and 5.9 digitized on clean crops supplied by the user (grid and
  tick calibration). Figure 5.9 top: ten straight lines V_min = a + b V_CR
  fitted with residuals below 0.4 kt; C_L/sigma = 2(C_T/sigma)/mu_min^2 read on
  the figure's box, a and b interpolated in C_L/sigma. Figure 5.9 bottom:
  h_hi(V_CR), 13 (FAA) and 15 (military) points. Figure 5.8: both branches
  parametrized by the height fraction (smooth at the nose).
- V_min: quartic of p. 357 checked against the power expression of p. 356
  (derivative zero, true minimum). No printed V_min.
- Figure 5.10 (sea level, 20,000 lb): noses at 80 kt (FAA) and 101 kt
  (military), tops at 1,330 and 1,390 ft. Figure 5.9 bottom gives h_hi within
  3 % at those speeds. Both noses imply V_min ≈ 82-83 kt through Figure 5.9
  top; the energy method with f = 20 ft², C_d = 0.01, e = 0.8 gives 84.3 kt,
  hence V_CR = 87 / 105 kt (+9 % / +4 %) since dV_CR/dV_min ≈ 2.8.
- Multiengine: V_CR = V_sink/2 (FAA) or V_sink (military), h_CR = max(50 ft, h_lo)
  smoothed over ±1 ft. The book gives no h_hi for one engine out; Figure 5.9
  bottom is used (assumption). V_sink needs the Chapter 4 power curve:
  `MultiEngineCriticalSpeedComp.RD` is ready for a BalanceComp.
- High-speed portion of Figure 5.7: no method in the book (p. 358), not modeled.

## G2e — Minimum touchdown speed (pp. 358-363)

- θ̇_max = γΩΔB₁/16 = 87.8 deg/s for ΔB₁ = 8° (printed 88). Δt = 1.25 s with
  the values above. α_TPP = min(θ̇Δt, 45°), min smoothed over ±1°.
- μ_auto from the Chapter 3 closed-form rotor (project decision): C_T/σ =
  (C_W/σ)/cos α_TPP, C_Q/σ = 0 solved by Newton on μ, λ' = μα_TPP − v₁/ΩR
  (InflowComp convention), exact induced velocity (regular at low μ).
- Figure 5.12 digitized on the user's clean crop, validation only. C_W/σ = 0.05:
  within 10 % from 10° to 45°. C_W/σ = 0.10: within 10 % from 30°; 16-25 % low
  at 15-20°, where C_T/σ ≈ 0.105 puts the charts' −5° twist rotor in stall
  (same mechanism as C5-6). The flare itself is used at 45°, where the gap is 5 %.
- Example: V_TD = 18.9 kt against 21 kt printed (μ_auto = 0.080; Figure 5.12
  gives 0.082, i.e. 19.7 kt). Total derivatives through the autorotation
  balance checked by complex step.

## G7 — Towing (pp. 371-372)

- Tension/G.W. = −sin γ + √(sin²γ + (T_max/G.W.)² − 1), from the force balance
  at hover (checked: the rotor thrust holding the computed tension is T_max).
- p. 372: T_max = 27,800 lb (hover OGE, sea level, Figure 4.35), 17,000 lb, flat
  towline: 21,995 lb against 22,000 lb printed.
- Figure 5.18 is this closed form; curve ends for T_max/G.W. = 1.2 and 2.4
  at γ = 0° and 45° agree within 3 %.
- T_max is an input; the Chapter 4 hover analysis gives the example's
  27,800 lb (margin −40 hp, docs/validation_performance.md).
