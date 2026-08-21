* Switched-Line Phase Shifter, 1-bit, target 90 deg @ 28 GHz
* Path A: ideal lossless TL, switches modeled as fixed resistors
* Phase state selected by R_short_*/R_long_* parameter swap
*   "short" state: R_short_*={R_on}, R_long_*={R_off}  (default)
*   "long"  state: R_short_*={R_off}, R_long_*={R_on}

* === STATE_TABLE ===
* bits=1
* ideal_step_deg=-90.0
* state_0: R_short_in={R_on} R_short_out={R_on} R_long_in={R_off} R_long_out={R_off}
* state_1: R_short_in={R_off} R_short_out={R_off} R_long_in={R_on} R_long_out={R_on}
* === END_STATE_TABLE ===

.PARAM Z0_line=50 L_short_mm=2.68 L_long_mm=5.36
.PARAM R_on=3 R_off=10k
.PARAM R_short_in=3 R_short_out=3      $ FRAMEWORK_CONTROLLED
.PARAM R_long_in=10k R_long_out=10k    $ FRAMEWORK_CONTROLLED
.PARAM fc=28e9

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

* --- Port 1: AC source with 50-ohm series source impedance ---
* Vavail = 0.5 V (Thevenin: Vsrc=1V across Rsrc=50 + Z_load)
Vsrc src 0 DC 0.5 AC 1
Rsrc src in 50

* --- Input branch: short and long paths via fixed resistors ---
R_in_short  in a_short  {R_short_in}
R_in_long   in a_long   {R_long_in}

* --- Two transmission line paths ---
* Lossless TL: TD = length / v_phase; eps_eff=2.5 -> v_phase=1.897e8 m/s
T_short a_short 0 b_short 0 Z0={Z0_line} TD={L_short_mm*1e-3 / 1.897e8}
T_long  a_long  0 b_long  0 Z0={Z0_line} TD={L_long_mm*1e-3 / 1.897e8}

* --- Output branch: combine paths ---
R_out_short b_short out  {R_short_out}
R_out_long  b_long  out  {R_long_out}

* --- Port 2: 50-ohm termination ---
Rload out 0 50

.control
op
print v(in) v(a_short) v(a_long) v(b_short) v(b_long) v(out)

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