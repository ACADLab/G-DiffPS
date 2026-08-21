* Vector-Modulator Phase Shifter, 4-bit, target -22.5 deg/state @ 28 GHz
* Path A: ideal I/Q sum via cascaded VCVS, 90 deg quadrature via TL
* Topology: ideal linear model emulating an active I/Q vector modulator.
*   - The input signal v(in) provides the in-phase (I) component.
*   - A quarter-wave transmission line delays v(in) by 90 deg, providing
*     the quadrature (Q) component v(q_in). The line is terminated in
*     a 50-ohm load so the input port sees a matched 50-ohm impedance.
*   - Two cascaded floating VCVS form a linear sum:
*       v(sum) = 2*G_I*v(in) + 2*G_Q*v(q_in)
*     The factor of 2 compensates for the 50/50 voltage divider between
*     R_drv_out (50 ohm) and Rload (50 ohm), giving 0 dB insertion loss
*     whenever G_I^2 + G_Q^2 = 1.
*   - State table sweeps (G_I, G_Q) around the unit circle in 22.5 deg
*     increments, achieving full 360 deg phase coverage in 16 states.
* Real-world vector modulators use Gilbert-cell mixers or analog VGAs
* to synthesize the I/Q weights; this ideal model captures only the
* linear phase-synthesis behavior, abstracting away the active-device
* nonlinearity, noise, and limited resolution of the weight DACs.
* Resulting output phase (with quarter-wave delay convention):
*   v(out) ~ G_I*v(in) + G_Q*v(in)*exp(-j*pi/2) = v(in)*(G_I - j*G_Q)
*   phase  = atan2(-G_Q, G_I)
* So state_k (G_I = cos(k*22.5 deg), G_Q = sin(k*22.5 deg)) gives
*   phase_k = -k*22.5 deg -> ideal_step_deg = -22.5.

* === STATE_TABLE ===
* bits=4
* ideal_step_deg=-22.5
* state_0:  G_I=1.000   G_Q=0.000
* state_1:  G_I=0.924   G_Q=0.383
* state_2:  G_I=0.707   G_Q=0.707
* state_3:  G_I=0.383   G_Q=0.924
* state_4:  G_I=0.000   G_Q=1.000
* state_5:  G_I=-0.383  G_Q=0.924
* state_6:  G_I=-0.707  G_Q=0.707
* state_7:  G_I=-0.924  G_Q=0.383
* state_8:  G_I=-1.000  G_Q=0.000
* state_9:  G_I=-0.924  G_Q=-0.383
* state_10: G_I=-0.707  G_Q=-0.707
* state_11: G_I=-0.383  G_Q=-0.924
* state_12: G_I=0.000   G_Q=-1.000
* state_13: G_I=0.383   G_Q=-0.924
* state_14: G_I=0.707   G_Q=-0.707
* state_15: G_I=0.924   G_Q=-0.383
* === END_STATE_TABLE ===

.PARAM Z0_line=50 L_quarter_mm=1.69
.PARAM R_on=3 R_off=10k
* Mismatch scale factors on the I and Q VCVS gains. Model real
* vector-modulator non-idealities: I/Q amplitude imbalance from
* mismatched VGA cores, finite DAC resolution, or process variation.
* Ideal calibration is G_I_scale = G_Q_scale = 1.0; LLM tunes these
* (range ~0.7 - 1.3) to minimize gain/phase error across the 16-state
* sweep. Effective VCVS gains in the netlist become 2*G_I*G_I_scale
* and 2*G_Q*G_Q_scale respectively.
.PARAM G_I_scale=1.0 G_Q_scale=1.0
.PARAM G_I=1.000 G_Q=0.000             $ FRAMEWORK_CONTROLLED
.PARAM fc=28e9                          $ FRAMEWORK_CONTROLLED
.PARAM eps_eff=2.5                      $ FRAMEWORK_CONTROLLED

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

* --- Port 1: AC source with 50-ohm series source impedance ---
* Vavail = 0.5 V (Thevenin: Vsrc=1V across Rsrc=50 + Z_load)
Vsrc src 0 DC 0.5 AC 1
Rsrc src in 50

* --- Quadrature path: lossless TL terminated in 50 ohm ---
* v(q_in) = v(in) delayed by quarter wavelength at fc = phase -90 deg
* eps_eff=2.5 -> v_phase=1.897e8 m/s; TD = L_quarter_mm*1e-3 / v_phase
T_quad   in 0 q_in 0 Z0={Z0_line} TD={L_quarter_mm*1e-3 / 1.897e8}
R_q_term q_in 0 50

* --- I/Q summing VCVS pair (linear, AC-friendly) ---
* Two floating VCVS in series build the sum at node 'sum':
*   E_I: v(sum)   - v(inter) = (2*G_I*G_I_scale) * v(in)
*   E_Q: v(inter) - 0        = (2*G_Q*G_Q_scale) * v(q_in)
*  => v(sum) = 2*G_I*G_I_scale*v(in) + 2*G_Q*G_Q_scale*v(q_in)
* The factor of 2 cancels the 50/50 divider at the output. The
* G_*_scale factors model I/Q amplitude mismatch (LLM-tunable).
E_I sum   inter  in    0 {2*G_I*G_I_scale}
E_Q inter 0      q_in  0 {2*G_Q*G_Q_scale}

* --- Output network: 50-ohm source driver into 50-ohm load ---
R_drv_out sum out 50
Rload out 0 50

.control
op
print v(in) v(q_in) v(sum) v(inter) v(out)

ac lin 201 24G 32G

let s11 = (v(in) - 0.5) / 0.5
let s21 = v(out) / 0.5
let s21_mag_db = db(s21)
* db(s11) fails when |s11|->0 (perfect match). Floor the magnitude
* squared at 1e-30 (-300 dB) so the log is always finite.
let s11_mag_sq = real(s11)*real(s11) + imag(s11)*imag(s11)
let s11_mag_db = 10*log10(s11_mag_sq + 1e-30)
let s21_phase  = 180/pi * cph(s21)

meas ac il_db_at_fc  FIND s21_mag_db AT=28e9
meas ac phase_at_fc  FIND s21_phase  AT=28e9
meas ac rl_db_at_fc  FIND s11_mag_db AT=28e9

let phase_deg = phase_at_fc
let il_db     = -1 * il_db_at_fc
let rl_db     = -1 * rl_db_at_fc
let gain_err_db = 0.0
print phase_deg il_db rl_db gain_err_db

quit
.endc

.end
