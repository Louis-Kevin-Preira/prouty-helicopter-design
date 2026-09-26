# Validation notes — `prouty.performance`

Chapter 4, Performance Analysis, p. 273-338.

Each entry records what the book prints, what an independent derivation gives,
the decision taken, and the test that enforces it. C4-1 to C4-3 (tail rotor
gross thrust p. 283/286, pseudo ground effect p. 282, vertical climb p. 314)
are recorded when their components are coded.

---

## C4-4 — The "95 °F day" is isothermal

**Printed.** Figures 4.1, 4.2 and 4.33-4.35 compare a standard day with a
"95 °F day". The text never defines it.

**Check.** Figure 4.33 p. 314 prints both altitudes for each density ratio.
A standard lapse shifted by +36 °F misses them by 500 to 4,500 ft; a day at
95 °F at every altitude lands within 100 ft:

| ρ/ρ₀ | book | ΔT = +36 °F | isothermal 95 °F |
|---|---|---|---|
| 0.367 | 23,800 | 28,268 | 23,710 |
| 0.533 | 14,700 | 17,727 | 14,753 |
| 0.739 | 6,300 | 7,708 | 6,369 |
| 0.862 | 2,200 | 2,714 | 2,235 |

**Decision.** `DayTemperatureComp(day='isothermal')` feeds `AtmosphereComp`
the altitude-dependent offset `dT = T_day − T₀(1 − L h)`, with
`T_day = 95 °F` by default. `AtmosphereComp` keeps its equations; it only
gained a `num_nodes` option (default 1, so Chapter 1 is untouched) for
`EngineGroup`.

**Test.** `test_fig_4_33_altitudes_need_an_isothermal_hot_day`.

---

## C4-5 — The piston lapse law is not in the book

**Printed.** p. 274 says reciprocating ratings are limited by intake manifold
pressure and rpm, and vary with altitude and temperature. The only engine data
are the turboshaft charts of Figures 4.1-4.3.

**Decision.** `PistonPowerLapseComp` uses `P = P_SL (p/p₀) √(T₀/T)`, written
`σ √(T/T₀)` so it reads `AtmosphereComp` directly. Supercharged engines hold
`P_SL` up to a critical altitude of the standard day and are aspirated above.
The law omits internal friction (the −0.132 term of Gagg-Ferrar), so it is
slightly optimistic at altitude.

The flat-to-aspirated corner is a quadratic fillet of half width `w = 0.02`
in `φ/φ_c`: C1, monotone, never above `min(1, φ/φ_c)`, and 0.5 % below `P_SL`
exactly at `h_c`. A smoothstep blend was tried first and rejected: it
overshoots 1 just below `h_c`, so the power rose with altitude.

**Tests.** `test_piston_aspirated_lapse_is_delta_over_sqrt_theta`,
`test_piston_hot_day_loses_sqrt_of_temperature_ratio`,
`test_piston_supercharged_is_flat_then_aspirated`,
`test_piston_supercharged_blend_is_continuous`.

---

## C4-6 — Example turboshaft: digitization of Figures 4.1 and 4.2

**Figure 4.1 p. 274.** Pixel digitization of the six curves, done twice (book
scan and a cleaner copy, agreement within 3 hp). Every curve is a straight
line, rms 1-2 hp, no curvature.

| rating | standard day | 95 °F day (isothermal, C4-4) |
|---|---|---|
| takeoff | 2129 − 40.0 h | 1862 − 57.85 h |
| intermediate | 2002 − 40.2 h | 1722 − 57.8 h |
| max continuous | 1598 − 23.0 h | 1256 − 43.0 h |

*h in thousands of ft.* The torque limit plateau reads 2,082 hp; 2,080 hp is
used. It cuts the standard takeoff line at 1,225 ft, matching "about 1,200 ft"
p. 320. Standard maximum continuous joins intermediate at 23,500 ft, so
`MCP ≤ intermediate` is enforced.

The lines are not extended with warnings beyond their drawn range (30,000 ft
standard, 25,500 ft hot): they are linear formulas, so the solvers can still
evaluate them there, and the book's analyses stay inside (highest hot-day
point, Fig. 4.33: 23,800 ft). A warning is raised only when `T_air` leaves the
band between the two days.

**Figure 4.2 p. 275.** The ram gain grows as V². A law `1 + K_r (γ/2) M²`
fitted by least squares on both conditions gives

| rating | K_r | sea level std | 4,000 ft, 95 °F | max error |
|---|---|---|---|---|
| takeoff | 1.12 | torque limited | 1.121 | 0 hp |
| intermediate | 0.65 | 0.72 | 0.63 | 9 hp |
| max continuous | 0.91 | 0.95 | 0.83 | 5 hp |

With K = 0.65, sea level intermediate reaches the torque limit at 188 kt
instead of about 165 kt (17 hp low at 165 kt).

**Discrepancy.** At V = 0 the two figures agree at sea level (±4 hp) but
Figure 4.2 is 11-14 hp below Figure 4.1 at 4,000 ft, 95 °F, for all three
ratings. The component follows Figure 4.1; Figure 4.2 is matched within 20 hp.

**Tests.** `test_turboshaft_torque_limit_below_1200_ft`,
`test_turboshaft_max_continuous_joins_intermediate_above_23500_ft`,
`test_turboshaft_ram_effect_fig_4_2`,
`test_turboshaft_intermediate_reaches_torque_limit_in_forward_flight`.

---

## C4-7 — Installed power: the p. 320 anchor ignores the ram effect

**Printed.** p. 311: installed ratings are 98 % of Figures 4.1-4.2 (2 % inlet
friction, p. 277). p. 320: the speed trade-off derivatives are taken at
"20,000 lb, 160 knots, and 3,920 h.p. (intermediate installed power rating)".

**Check.** `2 × 0.98 × 2,000 = 3,920 hp` is the sea level intermediate rating
at V = 0 (Fig. 4.1). At 160 kt Figure 4.2 raises it to about 2,070 hp per
engine, i.e. about 4,060 hp installed.

**Decision.** `InstalledPowerComp` is anchored on 3,920 hp at V = 0. When the
p. 320 derivatives are reproduced (G7), the power available must be taken at
V = 0 to match the book.

**Test.** `test_installed_power_example_helicopter_sea_level`.

---

## C4-8 — Example turboshaft fuel flow: digitization of Figure 4.3

**Printed.** Figure 4.3 p. 276 draws ten lines: seven standard-day Willans
lines (sea level to 30,000 ft), one for 4,000 ft at 95 °F, and the standard
day takeoff and maximum continuous limit loci. The flows include the 5 %
deterioration allowance of p. 275. All the Willans lines except the lowest
meet near 1,045 lb/hr at 2,100 hp.

**Digitization.** Done twice (book scan and a cleaner copy, agreement within
2.5 lb/hr on `a`, 0.002 on `b`). Every line is straight (rms 3-5 lb/hr).

| line | a (lb/hr) | b (lb/hr/hp) |
|---|---|---|
| sea level std | 242.4 | 0.3853 |
| 5,000 ft std | 197.7 | 0.4010 |
| 10,000 ft std | 158.5 | 0.4234 |
| 15,000 ft std | 119.7 | 0.4421 |
| 20,000 ft std | 87.1 | 0.4571 |
| 25,000 ft std | 56.3 | 0.4708 |
| 30,000 ft std | 28.3 | 0.4793 |
| 4,000 ft, 95 °F | 218.2 | 0.3946 |

Quadratics in altitude reproduce `a` within 1 lb/hr and `b` within 0.0035.

**Assumption.** One hot line only. Its offset from the standard day at
4,000 ft (`Δa = +11.4`, `Δb = −0.00571`, i.e. +2.5 % at 400 hp, +0.3 % at
1,600 hp) is blended with the same temperature weight as the ratings and held
constant with altitude.

**Limit loci not used.** The ratings come from Figure 4.1. The drawn limit
loci run almost tangent to the altitude lines, so their crossings are
ill-conditioned; where they can be read they sit up to about 45 lb/hr from the
points given by Figure 4.1 ratings on the altitude lines.

**Tests.** `test_turboshaft_fuel_flow_intercepts_fig_4_3`,
`test_turboshaft_fuel_flow_lines_converge_fig_4_3`,
`test_turboshaft_fuel_flow_hot_day_fig_4_3`,
`test_turboshaft_fuel_flow_deterioration_and_engines_out`.

---

## C4-9 — Gearbox losses: engine power is not the sum of rotor powers

**Printed.** p. 277 writes the example losses as
`0.0025 (4,000 + Eng.) + 0.00875 (4,000 + MR) + 0.0050 (750 + TR)`, then
assumes `Eng. = MR + TR` and rounds to `49 + 0.0112 MR + 0.0075 TR`.

**Check.** The expansion is exactly `48.75 + 0.01125 MR + 0.0075 TR`. The
engine power also carries the losses themselves, so the assumption
underestimates the nose gearbox loss by `0.0025 × losses` (about 0.2 hp).

**Decision.** `GearboxLossComp` takes `P_eng` as a separate input; the loop
`P_eng = P_MR + P_TR + losses` is closed exactly in EnginePowerRequiredComp
(linear, so no solver is needed). The drive system layout is an option, so
layouts other than the example (series and parallel boxes, any stage mix)
use the same K rule.

**Test.** `test_gearbox_loss_example_helicopter_p277`.

---

## C4-10 — Density scaling of the losses: p. 278 and p. 311 differ

**Printed.** p. 278 takes "both the transmission and accessory losses" as
proportional to the density ratio: `h.p._trans+acc = (ρ/ρ₀)(56 + .0112 h.p._M
+ .0075 h.p._T)`. p. 311 writes the engine power as
`h.p._eng/(ρ/ρ₀) = 56 + 1.0112 h.p._M + 1.0075 h.p._T`, with the rotor powers
also normalized by ρ/ρ₀ (the form of Figures 4.33-4.34).

**Check.** Multiplying p. 311 by σ gives `σ·56 + 1.0112 P_M + 1.0075 P_T`:
only the load-independent part is scaled, the load terms follow the actual
rotor powers. p. 278 literally scales the load terms too. The two agree only
at σ = 1.

**Decision.** `EnginePowerRequiredComp(density_scaled_losses=True)` follows
p. 311: design-power terms and accessories scale with σ, the K·P terms do
not. It is also the more physical reading, since a gear-mesh loss follows the
power it transmits. Default is `False` (no scaling) for the physical model.
The exact loop closure (C4-9) stays within 0.2 hp of p. 311 up to 3,000 hp.

**Tests.** `test_engine_power_closes_the_loss_loop_exactly`,
`test_engine_power_example_helicopter_p311`.

---

## C4-11 — Wake dynamic pressure: digitization of Figure 4.6

**Printed.** Figure 4.6 p. 281 gives q/D.L. against r/R at z/R = 0.1, 0.3
and 0.5, measured under a rotor with −4° twist (solid) and extrapolated by
Prouty to −10° with the squared induced velocity ratio of Chapter 1, p. 279
(dashed). Table 4.1 p. 282 lists his readings for the 21 segments of the
example helicopter, at z/R between 0.1 and 0.4.

**Digitization.** Pixel tracing of cleaner copies of the three panels;
solid and dashed separated by the gaps of the dashed line; the steep
outboard branches traced row by row; the two tip peaks of the z/R = 0.1
panel read by eye (1.58 at r/R = 0.820 measured, 1.45 at 0.817 extrapolated).
Resampled at Δr/R = 0.01, 0.005 between 0.75 and 0.90, lightly smoothed
inboard of 0.74 (at most 0.04 change).

**Interpolation.** Akima in r/R on each panel, linear in z/R between panels,
nearest panel held outside 0.1-0.5 with a warning.

**Check against Table 4.1.** With the −10° curves, 17 of the 19 readings
inside the wake agree within 0.06 (most within 0.02); with the −4° curves the
agreement is clearly worse (rms 0.09 against 0.07), which confirms that the
table used the extrapolated curves. Two readings differ:

| segment | r/R | Z/R | book | Fig. 4.6, −10° |
|---|---|---|---|---|
| 13 | 0.8 | 0.24 | 1.00 | 0.89 |
| 20 | 0.20 | 0.35 | 0.42 | 0.17 |

Segment 13 lies between the tip peak of z/R = 0.1 and the plateau of 0.3,
where linear interpolation in z/R is crude. Segment 20 cannot be read as
0.42 on any panel at r/R = 0.20; it is small (C_D·A = 3 ft²) and changes
D_v/GW by 0.0005.

**Tests.** `test_wake_q_ratio_reproduces_table_4_1`,
`test_wake_q_ratio_measured_curves_fig_4_6`.

---

## C4-12 — Twist correction of the wake: Prouty's −10° curves are weaker than his text

**Printed.** p. 279: for another twist, multiply the measured −4° distribution
by "the square of the ratio of induced velocities calculated by the method of
Chapter 1 for the two values of twist"; Figure 4.6 shows the result for −10°.

**Check.** The example rotor (p. 669) trimmed to 20,000 lb in
`HoverRotorGroup` at −10° and −4° gives an inflow ratio going from 1.22 at the
cutout to 0.89 at the tip. Applied to the −4° curves, rms deviation from
Prouty's −10° curves over 0.2 < r/R < 0.8:

| mapping of the wake point to the rotor | exponent 2 (text) | exponent 1 |
|---|---|---|
| same r/R | 0.045 / 0.090 / 0.091 | 0.021 / 0.048 / 0.050 |
| r/R ÷ wake edge R_edge(z) | 0.058 / 0.057 / 0.045 | 0.032 / 0.029 / 0.026 |

*(panels z/R = 0.1 / 0.3 / 0.5; the −4° curves themselves are 0.051 / 0.051 /
0.056 away)*

Prouty's drawn curves follow the ratio to the first power better than its
square. The square is what the text says and what the physics requires
(q ∝ v²); his hand-drawn extrapolation is probably milder than the calculation.

On the quantity that matters, the vertical drag of Table 4.1:

| q/D.L. used | D_v/GW |
|---|---|
| Prouty's readings (book) | 0.0421 |
| −10° curves of Figure 4.6 | 0.0407 |
| −4° × ratio², contracted mapping (**model**) | 0.0433 |
| −4° × ratio², same r/R | 0.0455 |
| −4° measured, no correction | 0.0371 |

**Decision.** Square of the ratio, as printed. A wake point at (r/R, z/R) is
fed by the rotor station r/R ÷ R_edge(z/R), R_edge being where the measured
q/D.L. returns to zero (0.867, 0.849, 0.845): it is better than the direct
mapping on average and lands within 0.0012 of the book's D_v/GW.

**Tests.** `test_twist_correction_against_the_minus_10_curves_fig_4_6`,
`test_induced_velocity_ratio_is_one_for_the_reference_twist`.

---

## C4-2 — Pseudo ground effect: 0.038 is a misprint for 0.38

**Printed.** p. 282: `ΔC_Q/σ = −.12 [C_T/σ^{3/2} √(.085/2) (.038)] = −.0094 (C_T/σ)^{3/2}`.

**Check.** `0.12 × √(0.085/2) × 0.038 = 0.00094`, ten times less than the
printed result. p. 281 gives v_IGE/v_OGE = 0.62 at Z/D = 0.12 (Figure 1.41),
so the bracket is `1 − 0.62 = 0.38`, which gives 0.0094. The book then uses
0.0094: −0.00024 at C_T/σ = 0.086 and 68 hp (p. 283), both reproduced.

**Figure 1.41.** Implemented in Chapter 1 as `GroundEffectComp`
(`prouty.hover`), solid curve only: 396 pixels, quartic fit rms 0.002,
tabulated every 0.05 up to z/D = 1.1, then closed on 1.0 at z/D = 1.5
(assumption beyond the drawn range, which ends near 0.99 at z/D = 1.17).
It reads 0.620 at z/D = 0.12 and 0.752 at 0.3 (book: 0.62 and 0.75).

**Tests.** `test_pseudo_ground_effect_example_helicopter_p282_p283`,
`test_ground_effect_figure_1_41_book_readings`.

---

## C4-13 — Ground proximity: Figure 4.8 used as shapes on the out of ground effect results

**Printed.** p. 283 and Figure 4.8 p. 285: model tests (Fradenburgh) of a
fuselage alone and a fuselage with wing show the vertical drag ratio and the
pseudo ground effect both reduced, even reversed, below about one rotor
diameter; S-76 model: +3 % download out of ground effect, −1 % at a 1 ft wheel
height. No method is given. For its own hover in ground effect (p. 309) the
book simply uses no vertical drag and no pseudo ground effect.

**Digitization.** Pixel columns of both panels, smoothing spline (rms within
one pixel), values at z/D = 2.6: D_v/T = 0.0249 (fuselage) and 0.168 (with
wing); ΔC_Q/σ = −0.00014 and −0.00044. The fuselage alone download is read
from z/D = 0.28 and extrapolated to 0.2.

**Decision.** Each curve divided by its value at z/D = 2.6 is a factor on the
out of ground effect results of this chapter's method:
`GroundProximityDownloadComp(configuration=...)` gives `k_Dv` and `k_PGE`,
consumed by `VerticalDragComp` and `PseudoGroundEffectComp` (default 1).
`'none'` keeps the OGE values, `'removed'` reproduces p. 309 (k = 0).

This transfers the shapes of one small model to any helicopter: it keeps the
physics that matter (reversal of the download below z/D ≈ 0.5 without wing,
loss of the pseudo ground effect near the ground) but the factors are not
general data. The fuselage alone pseudo ground effect factor slightly
exceeds 1 around z/D = 1.7 (1.014), within digitization noise.

**Tests.** `test_ground_proximity_fig_4_8_shapes`,
`test_ground_proximity_s76_download_reversal`,
`test_ground_proximity_book_switches`.

---

## C4-14 — Tail rotor-fin interference: digitization of Figure 4.9

**Printed.** Figure 4.9 p. 286: F/T against x/R for S/A = 0.10, 0.15, 0.20,
0.25, tractor and pusher installations; drawn for 0.2 ≤ x/R ≤ 1.0. Example
helicopter (p. 286): pusher, S/A = 0.25, x/R = 0.3, F/T = 0.125.

**Digitization.** Cleaner copy; pixel columns, curves separated by order
where all four are distinct and by nearest provisional fit where the pusher
curves converge; smoothing spline per curve (rms ≤ 0.0008), pusher curves
closed on zero at x/R = 1; tabulated every 0.05. End values: tractor 0.082 /
0.222 at x/R = 0.2 and 0.051 / 0.167 at 1.0 (S/A 0.10 / 0.25); pusher 0.069 /
0.159 at 0.2.

**Check.** Pusher, S/A = 0.25, x/R = 0.3: 0.123 (book 0.125).

**Interpolation.** Akima along x/R, linear between S/A curves. Outside the
drawn range the nearest edge is held with a warning, with one-sided slopes on
the edges; nothing is extrapolated. The book's own example lies inside, at the
S/A = 0.25 edge.

**Test.** `test_fin_interference_example_helicopter_p286`,
`test_fin_interference_fig_4_9_end_points`.

---

## C4-1 — Tail rotor gross thrust: 1.125 is the expansion of 1/(1 − F/T)

**Printed.** p. 283: `T_T_gross = T_T_req / (1 − F/T)`. pp. 286 and 309, with
F/T = 0.125: `T_T_gross = 1.125 T_T_net`. p. 286 also compares the fin-off
curve of Figure 4.10 "at a C_T/σ 13 % higher" for F/T = 0.13.

**Check.** `1/(1 − 0.125) = 1.1429`; 1.125 = 1 + F/T is the first order
expansion, 1.6 % low in thrust (about 25 lb for the example tail rotor). The
13 % of Figure 4.10 is the same approximation (1/(1 − 0.13) = 1.149).

**Decision.** `TailRotorGrossThrustComp` implements the equation of p. 283.

**Test.** `test_tail_rotor_gross_thrust_p283_and_example`.

---

## C4-15 — Tail rotor fin power factor: one pusher test, applied to both installations

**Printed.** p. 286: comparing the Lockheed 286 tail rotor (pusher, σ = 0.16,
x/R = 0.24, S/A = 0.23, F/T = 0.13, Figure 4.10) fin on and fin off at a C_T/σ
13 % higher, the measured power is about 94 % of the fin-off prediction at
C_T/σ = 0.08; "based on this one set of test data" the book suggests
`h.p._T = (1 − (F/T)/2)(h.p. for T_gross)`. p. 285 notes that a tractor
benefits from a pseudo ground effect and a pusher from a pseudo ceiling effect.

**Decision.** The factor is applied to both installations (user decision).
It rests on a single pusher test and should be read as a first estimate for
a tractor tail rotor.

**Test.** `test_tail_rotor_fin_power_p286_and_fig_4_10`.

---

## C4-16 — Fuselage drag: cleanliness level as a factor on the minimum curve of Figure 4.17

**Printed.** Figure 4.17 p. 294: minimum skin friction and form drag against
fineness ratio (R.N. = 7×10⁷) and the drag of ten aircraft. p. 294: use it "at
a level of aerodynamic cleanliness corresponding to that for one of the known
aircraft". p. 306: l/d = 7, C_DF = 0.078, f_F = 74 × 0.078 = 5.8 ft².

**Digitization.** Clean copy. Minimum curve 0.102 at l/d = 1, 0.036 near 2.6,
0.052 at 7, 0.079 at 10.2 (quartic fit beyond 2.5, rms 0.0014). Aircraft:

| aircraft | l/d | C_D | k = C_D / C_D,min |
|---|---|---|---|
| OH-6A | 2.48 | 0.071 | 1.99 |
| UH-1 | 4.94 | 0.080 | 1.88 |
| CH-47 | 6.00 | 0.106 | 2.25 |
| L-286 | 6.46 | 0.076 | 1.53 |
| P-80 | 6.95 | 0.075 | 1.44 |
| DC-3 | 7.02 | 0.081 | 1.55 |
| P-51 | 7.13 | 0.076 | 1.44 |
| F-104 | 8.96 | 0.077 | 1.16 |
| B-29 | 9.16 | 0.093 | 1.36 |
| B-17 | 10.17 | 0.108 | 1.37 |

**Decision (user).** `C_DF = k C_D,min(l/d)`, k from a reference aircraft
(p. 291: imperfections scale the computed friction drag); `mode='input'` takes
C_DF directly. At l/d = 7 the L-286 level gives 0.080 and the P-51 level 0.075,
bracketing the book's 0.078.

A multiplicative level transfers badly between very different fineness
ratios: from the OH-6A (l/d 2.5) it gives 0.104 at l/d 7, an additive
increment would give 0.088. Prefer a reference with a similar fineness ratio.

**Tests.** `test_fuselage_drag_example_helicopter_p306`, `test_fuselage_drag_figure_4_17`.

---

## C4-17 — Nacelle drag: digitization of Figure 4.19

**Printed.** Figure 4.19 p. 296 (Keys & Wiesner): nacelle drag coefficient on
frontal area against y/D_N, four measured points and a fairing. p. 306: two
nacelles, D_N = 2.8 ft, y/D_N = 0.5, C_DN = 0.09, A_N = 2π(1.4)² = 12 ft²,
f_N = 1.1 ft².

**Digitization.** Cleaner copy (the first pass on the book scan read 0.085 at
y/D_N = 0.5 and placed the third point at 0.54). The tick spacing of the copy
is uneven, so the axes are calibrated piecewise between ticks. Measured
points, located with a ring template: (0, 0.182), (0.302, 0.106),
(0.504, 0.088), (0.771, 0.078). Fairing: one point per pixel column,
smoothing spline (rms 0.001), tabulated every 0.1 up to 0.9, where it ends.

**Check.** 0.088 at y/D_N = 0.5 (book 0.09). With A_N = 12.3 ft² (the book
rounds to 12): f_N = 1.08 ft² against 1.1.

**Tests.** `test_nacelle_drag_example_helicopter_p306`, `test_nacelle_drag_figure_4_19`.

---

## C4-18 — Rotor hub drag: digitization of Figure 4.22 and Table 4.2

**Printed.** Table 4.2 p. 298: hub C_D on frontal area at zero angle of
attack and zero rpm, unfaired and faired, 14 hubs. Figure 4.22 p. 299: drag
ratio against shaft angle of attack (100 % rpm) and against rpm (α = 0).
Example p. 306: "D.R. = 1.00/.95 = 1.05", C_D = 1.1 × 1.05 = 1.16,
f_MH = 5 × 1.16 = 5.8 ft², f_T = 0.6 × 1.16 = 0.7 ft².

**Reading of the correction.** The table is at rpm = 0 and the chart is
normalized at 100 %, so the rpm ratio is divided by its value at 0:
`DR = [f/f_α=0](α_s) · [f/f_100%](rpm) / [f/f_100%](0)`, which is the book's
1.00/0.95 at α = 0, 100 %.

**Digitization.** Clean copies. Angle of attack: symbol centres every 2°
(template matching, the unfaired circles searched below the faired squares
where they overlap); both curves read 1.017 at α = 0 on the copy, so both are
divided by that value. rpm: the tick labels of the copy were first assigned
one level too high (the fairings then fell outside their guides), corrected;
pixel columns of each fairing, smoothing spline (rms ≤ 0.002), divided by the
value at 100 %. At 0 rpm: 0.949 unfaired (book 0.95), 0.839 faired.

**Tests.** `test_rotor_hub_drag_example_helicopter_p306`,
`test_rotor_hub_drag_figure_4_22_trends`.

---

## C4-19 — Rotor shaft drag: digitization of Figure 4.23

**Printed.** Figure 4.23 p. 300 (Hoerner): drag coefficient of a circular
cylinder against Reynolds number, with a second scale giving the cylinder
diameter at 150 kt. p. 288: R.N. = 6,400 V L at sea level. p. 306: D_s = 0.5 ft
at 115 kt gives R.N. = 0.6×10⁶, C_D = 0.3, f_MS = 0.3 ft².

**Digitization.** Clean copy, log axis calibrated decade by decade; the flat
branches traced by pixel columns, the drag crisis by pixel rows. Subcritical
1.20 up to 1×10⁵, minimum 0.27 at 6×10⁵, 0.31 near 3×10⁶, 0.30 at 10⁸.

**Reynolds number.** Computed as ρVD/µ with µ from Sutherland's law, so the
component follows altitude and temperature; at sea level ρ/µ = 6,360 s/ft²
against the book's 6,400 (0.6 % apart). The second scale of the figure checks
the calibration: 2.3 in at 150 kt falls on 3×10⁵ and 23 in on 3×10⁶, both
within 5 %.

**Check.** At R.N. = 0.6×10⁶ the curve reads 0.27, at the bottom of the
post-critical dip, where the book reads 0.3 — a 10 % difference on 0.3 ft²,
i.e. 0.03 ft² of the 19.3 ft² total.

**Tests.** `test_rotor_shaft_drag_example_helicopter_p306`,
`test_cylinder_drag_figure_4_23`.

---

## C4-20 — Hub-pylon interference: the example reads below the figure

**Printed.** Figure 4.24 p. 301 (Keys & Wiesner): interference factor k_i
against hub gap over pylon width, for fuselage angles of attack −3, 0, 3, 6
and 9°. p. 306: Z/W_p = 2.8/9 = 0.3, K_i = 0.15 at α_F = −5°,
f_M = 1.15 (5.8 + 0.3) = 7.0 ft².

**Digitization.** Cleaner copy, read row by row, keeping only the rows where
all five curves are separated (Z/W_p from 0.02 to 0.55; the text labels and
the inset sketch spoil the rest). Each curve fitted by
`a exp(−b Z/W_p) + c` (rms 0.011 to 0.024), tabulated every 0.05 up to 0.7:

| Z/W_p | −3° | 0° | 3° | 6° | 9° |
|---|---|---|---|---|---|
| 0.0 | 0.78 | 0.91 | 1.09 | 1.32 | 1.56 |
| 0.3 | 0.224 | 0.281 | 0.314 | 0.368 | 0.442 |
| 0.7 | 0.076 | 0.101 | 0.124 | 0.157 | 0.178 |

**Discrepancy.** At Z/W_p = 0.3 the −3° curve reads 0.224, and the curves are
about 0.019 per degree apart, so α_F = −5° extrapolates to 0.19, not the 0.15
of the example. f_M becomes 7.2 ft² instead of 7.0, i.e. +0.2 ft² of the
19.3 ft² total.

**Decision.** The figure is followed. α_F is extrapolated linearly below −3°
(helicopters cruise nose down, and the example itself is at −5°), with a
warning; Z/W_p is held at the ends of the table.

**Tests.** `test_hub_pylon_interference_example_helicopter_p306`,
`test_hub_pylon_interference_figure_4_24`.

---

## C4-21 — Landing gear drag: Figure 4.26

**Printed.** Figure 4.26 p. 303: drag coefficients of wheels alone (on b × d:
0.12, 0.25, 0.15), wheels with struts (on frontal area: 0.55, 0.36, 0.25),
nose wheel installations (0.58 unfaired, 0.27 faired), skids (1.01 tubular,
0.40 faired), and a curve of C_D against e/d for a nose or tail wheel with a
round or faired strut. p. 307: main gear 4 ft² at C_D = 0.3 gives 1.2 ft²;
nose gear 1.5 ft², e/d = 3.4/2 = 1.7, C_D = 0.56, f = 0.8 ft².

**Digitization.** Clean copy, pixel columns. The two struts coincide below
e/d = 1.25 (the wheel shields the strut): 0 at e/d = 0, 0.12 at 0.5, 0.46 at
1.0. Above, the round strut rises to 0.81 at e/d = 3 while the faired strut
dips to 0.42 near 1.75 before reaching 0.57.

**Check.** At e/d = 1.7 the round strut reads 0.545 against the book's 0.56;
f_NLG = 0.82 ft² against 0.8.

**Tests.** `test_landing_gear_drag_example_helicopter_p307`,
`test_landing_gear_figure_4_26`.

---

## C4-22 — Surface profile drag: Figure 4.12 rather than Figure 4.15

**Printed.** p. 307: for the stabilizers, "Estimate C_D0 from Figure 4.15",
giving 0.010 at R.N. = 2×10⁶ and 4×10⁶ with t/c = 0.12.

**Check.** Figure 4.15 (two-dimensional surfaces) digitized from a cleaner
copy puts the t/c = 0.06 to 0.20 curves in a bundle at C_D ≈ 0.018 to 0.024
above 10⁶, twice the value the book reads there; the t/c = 0 curve gives
0.012. The user reports that the page of Figure 4.15 is buckled in the scan,
which distorts the vertical scale; the figure is therefore not used.

**Decision (user).** C_d0 is built from the skin friction coefficient of
Figure 4.12 p. 289 with the usual form factor:
`C_d0 = 2 C_F(R.N.) [1 + 2(t/c) + 60(t/c)^4]`. With the forced turbulent curve
(`transition='turbulent'`, the default, and the realistic one for a surface
with imperfections, p. 291) C_F = 0.0040 at 2×10⁶, so C_d0 = 0.0100, the
book's value. The natural transition curve is available as an option.

**Figure 4.12 digitization.** The page is also buckled: its x axis undulates
by up to 10 pixels, i.e. 0.00025 in C_F, a tenth of the plotted values. The
baseline is therefore traced column by column and fitted, and each reading is
taken from the local baseline rather than from a single horizontal. The
resulting curve matches 0.455/(log R.N.)^2.58 within 0.0004 over 10⁵-10⁷.

**Figure 4.21 (junction drag).** Digitized on a clean copy: zero below
t/c = 0.065, 0.076 at 0.12 (book 0.072), 0.41 at 0.60. Akima undershoots below
the threshold, so the coefficient is clamped at zero there.

**Check of the example.** Horizontal stabilizer: C_D = 0.0181 against the
book's 0.019, f_H = 0.24 ft² against 0.2565 (rounded to 0.2 in the table).
Vertical: f_V = 0.16 ft² against 0.18.

**Tests.** `test_stabilizer_drag_example_helicopter_p307`,
`test_skin_friction_and_junction_charts`.

---

## C4-23 — Rotor-fuselage interference: Figure 4.25

**Printed.** Figure 4.25 p. 302: measured drag increment on fuselage frontal
area against fuselage angle of attack, for a wind tunnel configuration.
p. 308: "From Figure 4.25 estimate ΔC_D at α_F = 0" gives 0.018 and
f_int = 0.018 (74) = 1.3 ft².

**Digitization.** First on the book scan, then checked on a clean copy: pixel
columns of the fairing, smoothing spline, rms 0.0002. The two passes agree
within 0.0001 everywhere. Reads 0.0058 at −10°, 0.0180 at 0° (the book's
value) and 0.0229 at +8°, with the slope flattening at positive angles.

**Note.** The procedure reads this term at zero angle of attack, unlike the
hub-pylon factor of Figure 4.24 which the same example reads at −5°. The
component keeps α_F as an input, defaulting to 0.

**Tests.** `test_rotor_fuselage_interference_example_helicopter_p308`,
`test_rotor_fuselage_interference_figure_4_25`.

---

## C4-24 — Exhaust drag

**Printed.** p. 304: a helicopter exhaust is sized for hover, so it gives
residual thrust only up to some speed and residual drag beyond; a canted stack
loses the rearward momentum of the engine flow,
`D_ex = ṁ (V − V_ex cos χ)`. p. 308: the manufacturer's net residual thrust
for the example is 2(−11) = −22 lb, so `f_ex = −T_net/q = 22/45 = 0.5 ft²`.

**Implementation.** Both forms: `mode='thrust'` takes the residual thrust per
engine and the number of engines, `mode='momentum'` the mass flow, exhaust
velocity and cant angle. q = ρV²/2 comes from the atmosphere and the reference
speed, and reads 44.8 psf at 115 kt against the book's 45.

**Test.** `test_exhaust_drag_example_helicopter_p308`,
`test_exhaust_drag_momentum_form_p304`.

---

## C4-25 — Total parasite drag

**Printed.** p. 308 adds the eleven items of the example to 19.3 ft². p. 304:
"The estimating methods outlined in this chapter must be considered to produce
minimum estimates… I recommend that at least another 20 % be added to the
total to include those items that were not initially included or that will
grow during the normal development of the helicopter."

**Implementation.** The item names are an option, so any breakdown sums; the
default is the example's list. `f_misc` covers antennas, handles, lights,
steps, skin gaps and cooling leakage (0.5 ft² in the example). `margin` is 0
by default, giving `f_total`; setting it to 0.20 gives the recommended
`f_design` = 23.2 ft² for the example.

**Test.** `test_total_parasite_drag_example_helicopter_p308`.

---

## C4-26 — Ground effect power: the printed ΔC_Q/σ is ten times low

**Printed.** Chapter 1 p. 67 gives, at constant weight,
`ΔC_Q/σ = −(C_T/σ_OGE)^{3/2} √(σ/2) {1 − (1 − D_v/G.W.)^{3/2} (v_IGE/v_OGE)}`,
then p. 68 evaluates it for C_T/σ = 0.085, D_v/GW = 0.04 and v_IGE/v_OGE = 0.75
(Z/D = 0.30, Figure 1.41) as `ΔC_Q/σ = 0.000143`, "which corresponds to a
decrease of 407 h.p. out of a total of approximately 2,000 h.p."

**Check.** The equation gives 0.001505, and 0.001505 × 0.085 × ρ A (ΩR)³/550
= 429 hp with ρ = 0.002377, A = 2,827 ft², ΩR = 650 ft/s. The printed 0.000143
would give 41 hp, ten times less than the book's own 407 hp; 0.00143 is the
value consistent with it. The remaining 5 % between 429 and 407 hp is the
reading of Figure 1.41 (0.75 printed, 0.752 digitized, and 0.766 would give
exactly 407).

**Decision.** `GroundEffectPowerComp` (in `prouty.hover`, since the equation
is Chapter 1's) implements the printed equation, with
`vertical_drag='none'|'included'` for the isolated rotor and the constant
weight forms. The example of Chapter 4 p. 309 uses no vertical drag in ground
effect, i.e. D_v/GW = 0. The constant power thrust ratio of p. 68,
`(v_IGE/v_OGE)^(−2/3)`, is returned as well.

**Tests.** `test_ground_effect_power_example_p67_p68`,
`test_ground_effect_power_vertical_drag_and_thrust_ratio`.

---

## C4-27 — Hover performance: the chain reproduces the example

**Printed.** p. 311: `h.p._eng/(ρ/ρ₀) = 56 + 1.0112 h.p._M + 1.0075 h.p._T`.
p. 312: the example "can hover OGE at sea level, standard day, at 27,800 lb";
Figure 4.35, out of ground effect with takeoff power at 20,000 lb: about
11,000 ft on a standard day and 7,000 ft on a 95 °F day. p. 310: the tail
rotor is the more loaded rotor against its own maximum and limits the high
gross weight and altitude performance.

**Check (HoverPerformanceGroup, example helicopter, 20,000 lb, sea level).**

| quantity | group | book |
|---|---|---|
| main rotor power | 2,026 hp | "approximately 2,000 h.p." (p. 68) |
| tail rotor power | 202 hp | — |
| engine power required | 2,307 hp | 2,308 hp by the p. 311 equation |
| installed takeoff power | 4,077 hp | 2 × 0.98 × 2,080 |
| hover ceiling OGE, standard | 11,200 ft | about 11,000 ft (Fig. 4.35) |
| hover ceiling OGE, 95 °F | 7,075 ft | about 7,000 ft (Fig. 4.35) |
| sea level OGE capability | margin −40 hp at 27,800 lb | hovers at 27,800 lb |

**Solver.** The chain is acyclic (the yaw balance uses the main rotor power
only), so `mode='performance'` needs no solver beyond the loop inside G2. In
`mode='ceiling'` the altitude is the implicit output of HoverCeilingBalance
and the group installs a Newton solver with an Armijo-Goldstein line search:
without it the first steps push the altitude thousands of feet above the
ceiling, where the tail rotor no longer trims and the model returns NaN.

**Tail rotor collective.** At 11,000 ft the tail rotor needs 26° of collective
and reaches C_T/σ = 0.139 against its 0.155 maximum, while the main rotor is
at 0.12 against 0.167: Prouty's mismatch, reproduced.

**Tests.** `test_hover_performance_group_engine_power_p311`,
`test_hover_performance_group_sea_level_capability_p312`,
`test_hover_performance_group_ceilings_fig_4_35`,
`test_hover_performance_group_tail_rotor_limits_at_altitude`.

---

## C4-28 — Vertical climb: the chain reproduces Figure 4.36

**Printed.** p. 314 gives the power above hover, p. 315 the induced velocity in
climb, p. 316 says the equation is "evaluated… by equating it to the excess
power available". Figure 4.36 p. 317, 20,000 lb with takeoff power: about
3,270 ft/min at sea level on a standard day, zero near 12,200 ft; about
2,430 ft/min at sea level on a 95 °F day, zero near 7,300 ft.

**Check (VerticalClimbGroup on the example helicopter).**

| condition | group | Figure 4.36 |
|---|---|---|
| standard, sea level | 3,244 ft/min | 3,270 |
| standard, 5,000 ft | 2,586 | about 2,600 |
| standard, 11,000 ft | 479 | about 400 |
| 95 °F, sea level | 2,550 | 2,430 |
| 95 °F, 5,000 ft | 1,056 | about 1,100 |

**Above the hover ceiling.** The chain stops there for a physical reason: the
tail rotor runs out of collective (p. 310), so the hover trim itself fails
before the climb balance can. The balance keeps a small negative bound on V_c
so it stays solvable and differentiable just past zero, but a negative value
only measures the power deficit; the descent branch of momentum theory and the
vortex ring state are not modelled.

**Tests.** `test_vertical_climb_group_fig_4_36`,
`test_vertical_climb_group_falls_to_zero_at_the_ceiling`.

---

## C4-29 — Forward flight power: the book's figures disagree with the book's charts

**Printed.** Figure 4.38 p. 319 and Figure 4.48 p. 333 give the power required
of the example helicopter, both labelled "chart method of Chapter 3". The
table of p. 319 reads 2,330 hp at 20,000 lb, 140 kt, sea level; Figure 4.48
digitised here gives the same curve, 1,131 hp at 60 kt, a minimum of 1,059
near 80 kt, 1,577 at 120 and 2,337 at 140.

**First reading, and it was wrong.** Against those curves our chain looked 4 %
low at 60 kt and 38 % low at 140, and the gap was put down to blade drag near
stall that a constant mean c_d cannot carry.

**What the charts themselves say.** Rebuilding one point by the book's own
procedure (Chapter 3 diagnostics, section "Figure 4.38 rebuilt"), at
mu = 0.30, i.e. 115.5 kt, 20,000 lb, sea level:

| route | engine hp |
|---|---|
| the isolated rotor charts, read on the plates | 1,091 |
| this chain | 1,176 |
| Figure 4.48 | 1,490 |

The charts and this chain agree to 8 %. Figure 4.48 stands 37 % above the
charts. The same split appears inside Chapter 3: Table 3.5 case 6 is 2,109 hp
by the chart method against 1,760 by the closed form, 20 %, where Figure 4.48
at 140 kt would ask for 65 %.

**Decision.** The reference for forward flight power is the isolated rotor
chart, not Figures 4.38 and 4.48. This chain is 3 to 12 % above the charts,
measured curve by curve on chart 2 at mu = 0.30 (+3 % at theta_0 = 12°, +10 %
at 14°, +12 % at 16°), and that is the accuracy claimed here. No calibration
and no correction to the rotor: a model is not bent to match a figure that
contradicts its own source.

**What is still unexplained.** Why Figures 4.38 and 4.48 sit where they do. A
parasite area of 49 ft² would put them there, against the 19.3 ft² of the
book's own drag build-up (G4); so would a gross weight well above the stated
20,000 lb. Neither is resolved.

**Tests.** `test_forward_flight_power_matches_the_chart_method`,
`test_forward_flight_power_below_figure_4_48_c4_29` (a documented fact about
the book's figure, not a tolerance on the model).

**Re-examined with the Chapter 5 stall increment (Sept 2026).** The C5-6 work
digitized the five chart plates of pp. 258-266 and built the chart stall torque
above the closed form (`StallTorqueIncrementComp`). Adding it to this chain,
with the charts read as they are (`stall=True, stall_options={'twist_shift':
'book'}`: no p. 230 displacement of the stall limits), 20,000 lb, sea level:

| V, kt | 60 | 80 | 100 | 120 | 140 | 160 |
|---|---|---|---|---|---|---|
| chain (default) | 1,085 | 1,027 | 1,076 | 1,214 | 1,454 | 1,876 |
| chain + stall, p. 230 shift | 1,106 | 1,048 | 1,092 | 1,256 | 1,519 | 2,277 |
| chain + stall, book reading | 1,120 | 1,059 | 1,152 | 1,522 | 2,244 | 2,772 |
| Figure 4.48 | 1,131 | 1,059 | 1,189 | 1,577 | 2,337 | 3,182 |

The book reading meets Figure 4.48 within 5 % from 60 to 140 kt (160 kt is
mu = 0.415, past the last plate). The trimmed rotor sits at C_T/sigma = 0.086,
X = 0.20 at mu = 0.30, on the stall knee of the charts (theta_0 ~ 14 deg); the
rebuild above used C_T/sigma = 0.083 and X = 0.1555 (f alone) and read the
inflow plate, which lands at 12.3 deg, below the knee. So Figure 4.48 is the
chart method at the loads the full trim gives, stall torque included, read
without the twist displacement; our default chain has no stall torque.

**Status.** Explained, not changed: the default stays `stall=False` (project
decision), and the anchors above keep their meaning. Test:
`test_c4_29_book_reading_reproduces_figure_4_48`.

---

## C4-30 — Maximum speed: the balance works, the trim runs out first

**Printed.** p. 320 matches the power required of Figure 4.38 against the
engine curves of Figures 4.1 and 4.2, iterating because the compressibility
loss depends on the speed being sought; Figure 4.39 gives about 162 kt at sea
level on intermediate power at 20,000 lb.

**Implementation.** `MaxSpeedBalance` makes `P_req(V) − P_avail(V) = 0` a
residual, so the compressibility iteration of p. 320 disappears into the
Newton solve and the ram effect on the ratings is taken automatically. The
transmission torque limit already sits in the ratings, which is what kinks the
takeoff lines of Figure 4.39 at low altitude.

**Check.** On a bucket-shaped toy power curve the balance closes to 1e-8,
V_max grows with the power available, and the total derivative dV_max/dP
matches retrimmed central differences to 1e-5 — the derivative the book
estimates graphically as 0.009 kn/hp (p. 320).

**On the example helicopter it reaches 177 kt, mu = 0.46, where the Chapter 3
trim stops converging.** Since C4-29, that is no longer read as a shortfall of
our power: the book's 162 kt comes from Figure 4.39, which is built on the
power curves that stand 37 % above the charts. A maximum speed consistent with
the charts belongs beyond mu = 0.45, which is where the closed-form trim ends.
Raising that ceiling means extending the trim's domain, not its power level.

**Tests.** `test_max_speed_balance_closes_on_the_power_available`,
`test_max_speed_balance_trade_off_derivative_p320`.

---

## C4-31 — Equivalent rotor lift-to-drag ratio

**Printed.** p. 322: the rotor is compared with a wing by charging it with its
own power only — equivalent lift is the vertical component of the thrust,
equivalent drag is `550 h.p._M/V` less the parasite drag of everything but the
main rotor hub and mast. Figure 4.40 p. 323 (20,000 lb, sea level) peaks near
6.0 at about 100 kt, from 2.7 at 60 kt down to 2.9 at 160 kt.

**Check.** This chain gives 3.9 at 60 kt, 5.8 at 80 and 7.4 at 100. The
definition itself is verified term by term.

**Reading of the difference.** Figure 4.40 is built on the power required of
Figure 4.38, which C4-29 shows standing 37 % above the book's own charts; a
power that high makes the rotor look worse, so the figure understates the
ratio by about as much as our ratio exceeds it. Charged with a chart-consistent
power the peak would sit near 6.5 to 7, close to what this chain gives.

**Note.** f_rest is an input: the trim of Chapter 3 reads a single f from
Figure A.2, so the hub and mast share (7.0 ft² for the example, G4) has to be
taken out explicitly.

**Tests.** `test_equivalent_rotor_ld_definition_p322`,
`test_equivalent_rotor_ld_above_figure_4_40_c4_29`.

---

## C4-32 — Ferry mission: the stated takeoff weight is 96 lb light

**Printed.** p. 329 lists the minimum operating weight of the example as
11,261 lb (including a 481 lb cabin auxiliary tank), 15,961 lb of usable fuel
in five tanks and 409 lb for each of the two external tanks. Reserve no. 1 is
420 lb, the first three hours burn 4,400 lb and WUTO 56 lb, so
reserve no. 2 = (0.1/1.1)(15,961 − 4,876) = 1,008 lb and the landing weight is
12,689 lb. The text then gives a takeoff weight of 27,944 lb.

**Check.** All the reserve numbers are reproduced exactly. The takeoff weight,
however, is `11,261 + 2(409) + 15,961 = 28,040 lb`, 96 lb above the printed
27,944. The difference is not the WUTO fuel (56 lb), which the book removes
separately to get the mission start weight, nor the unusable fuel (30 lb),
which is already inside the minimum operating weight.

**Decision.** `FerryReservesComp` builds the takeoff weight from the parts, so
it returns 28,040 lb, and also returns the start weight, takeoff less WUTO, as
the integration limit of Figure 4.46.

**Tests.** `test_ferry_reserves_example_p329`,
`test_ferry_start_weight_and_the_book_takeoff_weight_c4_32`.

---

## C4-33 — Cruise chain assembled: specific range and best range speed

**Assembly.** `CruisePerformanceGroup` chains the atmosphere, the Chapter 3
level trim with its compressibility penalty, the drive losses, the engine fuel
flow and the specific range and endurance. One flight point costs one trim
solve. The speeds are found on top of it: `FuelFlowSlopeComp` runs the same
group as a sub-problem and returns dFF/dV by `compute_totals`, and
`BestRangeSpeedBalance` closes the tangency; the whole best range solve takes
about 4 s.

**Check (20,000 lb, sea level, no wind).**

| quantity | chain | book |
|---|---|---|
| specific range at 114 kt | 0.123 n.mi./lb | 0.114 (Fig. 4.42) |
| best range speed | 148 kt | 114 kt |
| specific range there | 0.135 | 0.114 |

**Reading of the difference.** Figures 4.42 to 4.44 are built on the fuel flow
that follows the power required of Figure 4.38, which C4-29 shows standing
37 % above the book's own charts. A power curve that rises too steeply with
speed pulls the tangency from the origin back towards low speed, which is why
the book's best range speed is low and its specific range with it. Our 148 kt
is what a chart-consistent power curve gives; it is not a failure of the
tangency balance, which lands on the maximum of the specific range to 0.2 kt
on an analytic curve.

**Two traps found while assembling, both now handled in
FuelFlowSlopeComp.** A sub-problem that has already been set up keeps its
values, so the component no longer calls `setup()` on it — doing so silently
reset every input and the trim then failed to converge. And the sub-problem's
own solvers start from wherever the last call left them, so a `reset` option
forces the trim states back to a sane guess before each run.

**Tests.** `test_cruise_group_specific_range_p323`,
`test_cruise_group_wind_moves_the_specific_range`,
`test_cruise_group_best_range_speed_with_the_real_chain_c4_29`.

---

## C4-34 — Climb angle: the book reads an arc tangent

**Printed.** p. 333: `f_climb = f + G.W. sin(gamma)/q`, and for the one engine
case "it can climb at 1,100 ft/min at 48 knots, which gives a climb angle of
12.7 degrees".

**Check.** 1,100 ft/min is 18.33 ft/s and 48 kt is 81.0 ft/s.
`arctan(18.33/81.0) = 12.75°` while `arcsin(18.33/81.0) = 13.08°`: the book
takes the speed as horizontal, not along the flight path.

**Decision.** `ClimbFlatPlateComp` has both, `angle_convention='horizontal'`
by default to follow the book, `'path'` for the strict flight path angle. The
two differ by 0.4° at this point and diverge as the angle grows; the drag
increment itself always uses sin(gamma).

**Test.** `test_climb_angle_convention_c4_34`.

---

## C4-35 — Forward climb assembled: the Chapter 3 climb trim is the weak link

**Assembly.** `ForwardClimbGroup` closes the power loop of p. 333 on top of
the cruise chain, with the climb handled where Chapter 3 already handles it:
`TrimConditionsGroup(mode='climb')` takes the rate of climb and folds the
weight component along the flight path into the drag area
(`ClimbDragAreaComp`, tail rotor H-force included), so nothing of p. 333 is
reimplemented. `ClimbFlatPlateComp` rides alongside to report the flight path
angle in the book's convention (C4-34) and the drag area it implies.

**State.** Each piece is tested on its own: the flat plate area and the angle
(C4-34), the rate of climb balance, the time and distance integrals, and the
two ceilings. The assembled group, however, inherits a cold start weakness of
the Chapter 3 climb trim: at 20,000 lb and 80 kt it trims at 1,000 and
2,000 ft/min when started from a converged neighbour and fails from scratch at
1,500 ft/min, regardless of the initial fuselage angle or blade loading. No
group-level test is asserted for that reason; the fix belongs to the Chapter 3
rework already planned, together with C4-29 and C4-30.

---

## C4-36 — Military mission: Table 4.4 reproduced

**Printed.** pp. 336-337: a ten-segment mission analysed backwards, because
"the landing weight can be closely estimated since it is simply the minimum
operating weight plus a small fuel reserve" while the takeoff weight is not
known. Table 4.4 sweeps from a landing weight of 10,430 + 300 lb up to an
estimated takeoff weight of 19,705 lb, then closes with mission fuel 2,375 lb,
total fuel 2,375/0.9 = 2,639 lb, reserve 264 lb and
T.O.G.W. = 10,430 + 6,600 + 2,639 = 19,669 lb, the sweep being "36 lb too high
due to initial reserve estimate".

**Check (MilitaryMissionGroup).** With the table's own fuel flows, times,
distances, specific range and specific endurance: every segment fuel within
4 lb, every gross weight within 15 lb, estimated takeoff weight 19,707 lb,
mission fuel 2,377 lb, total 2,641 lb, reserve 264 lb, T.O.G.W. 19,671 lb. The
gap between the swept and the closed takeoff weight comes out as the reserve
estimate error itself, 300 − 264 = 36 lb, which the test asserts as an
identity rather than as a number.

**Payload drop.** Segment 6 ends with 6,600 lb discharged, so the end weight
of that segment still carries it and the sweep steps by 6,600 lb between
segments 6 and 7 — placing the drop one segment too early was the first bug,
and the table's own column caught it.

**Tests.** `test_military_mission_segment_fuel_table_4_4`,
`test_military_mission_backward_sweep_table_4_4`,
`test_military_mission_summary_p337`.
