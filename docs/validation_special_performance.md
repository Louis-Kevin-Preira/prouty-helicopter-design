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

**Decision.** Squared ratio by default; `book` option for the printed form.

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

**Decision.** Linear form by default; `book` option for the square root.

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
