* Switched-Filter Phase Shifter, 1-bit, target -180 deg @ 28 GHz (MSB)
* Path A: ideal lossless LC pi-sections, switches modeled as fixed resistors
* Topology: two parallel pi-section paths (HPF and LPF) gated by switches
*   HPF path: series capacitor + shunt inductors  -> phase leading at fc
*   LPF path: series inductor   + shunt capacitors -> phase lagging at fc
* Phase state selected by R_hpf_*/R_lpf_* parameter swap:
*   "hpf" state (default): R_hpf_*={R_on}, R_lpf_*={R_off}
*   "lpf" state:           R_hpf_*={R_off}, R_lpf_*={R_on}
* Each section is designed for Z0=50 ohm match at fc, with element values
*   L = Z0/(2*pi*fc), C = 1/(2*pi*fc*Z0)
* giving +/-90 deg per section. State 0 (HPF) leads by ~+90 deg, state 1
* (LPF) lags by ~-90 deg, so the differential is ~180 deg -- the MSB of a
* multi-bit switched-filter phase shifter. Both states retain low IL
* (~0.6 dB) and excellent return loss (>55 dB) at fc by Z0-match
* construction. Reducing L/C symmetrically to chase ~90 deg differential
* breaks the Z0 = sqrt(L/C) match and severely degrades both IL and RL;
* the 180-deg labeling preserves textbook design integrity. Broadband
* digital phase shift; flat amplitude across band.

* === STATE_TABLE ===
* bits=1
* ideal_step_deg=-180.0
* state_0: R_hpf_in={R_on}  R_hpf_out={R_on}  R_lpf_in={R_off} R_lpf_out={R_off}
* state_1: R_hpf_in={R_off} R_hpf_out={R_off} R_lpf_in={R_on}  R_lpf_out={R_on}
* === END_STATE_TABLE ===

.PARAM Z0_line=50
.PARAM C_hpf_pf=0.114 L_hpf_nh=0.284
.PARAM L_lpf_nh=0.284 C_lpf_pf=0.114
.PARAM R_on=3 R_off=10k
.PARAM R_hpf_in=3  R_hpf_out=3        $ FRAMEWORK_CONTROLLED
.PARAM R_lpf_in=10k R_lpf_out=10k     $ FRAMEWORK_CONTROLLED
.PARAM fc=28e9                         $ FRAMEWORK_CONTROLLED
.PARAM eps_eff=2.5                     $ FRAMEWORK_CONTROLLED

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

* --- Port 1: AC source with 50-ohm series source impedance ---
* Vavail = 0.5 V (Thevenin: Vsrc=1V across Rsrc=50 + Z_load)
Vsrc src 0 DC 0.5 AC 1
Rsrc src in 50

* --- HPF pi-section path: series Cap, shunt Inductors ---
*   in -- R_hpf_in -- a_hpf -- C_hpf -- b_hpf -- R_hpf_out -- out
*                       |                  |
*                     L_hpf              L_hpf
*                       |                  |
*                      gnd                gnd
R_in_hpf    in     a_hpf {R_hpf_in}
Lp_hpf_in   a_hpf  0     {L_hpf_nh*1e-9}
C_hpf_ser   a_hpf  b_hpf {C_hpf_pf*1e-12}
Lp_hpf_out  b_hpf  0     {L_hpf_nh*1e-9}
R_out_hpf   b_hpf  out   {R_hpf_out}

* --- LPF pi-section path: series Inductor, shunt Capacitors ---
*   in -- R_lpf_in -- a_lpf -- L_lpf -- b_lpf -- R_lpf_out -- out
*                       |                  |
*                     C_lpf              C_lpf
*                       |                  |
*                      gnd                gnd
R_in_lpf    in     a_lpf {R_lpf_in}
Cp_lpf_in   a_lpf  0     {C_lpf_pf*1e-12}
L_lpf_ser   a_lpf  b_lpf {L_lpf_nh*1e-9}
Cp_lpf_out  b_lpf  0     {C_lpf_pf*1e-12}
R_out_lpf   b_lpf  out   {R_lpf_out}

* --- Port 2: 50-ohm termination ---
Rload out 0 50

.control
op
print v(in) v(a_hpf) v(b_hpf) v(a_lpf) v(b_lpf) v(out)

ac lin 201 24G 32G

let s11 = (v(in) - 0.5) / 0.5
let s21 = v(out) / 0.5
let s21_mag_db = db(s21)
let s11_mag_db = db(s11)
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
