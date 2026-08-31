* Gen_T17 — composed phase shifter, 1-bit

* === STATE_TABLE ===
* bits=1
* ideal_step_deg=-90.0
* state_0: R_sw={R_on}
* state_1: R_sw={R_off}
* === END_STATE_TABLE ===

.PARAM R_on=3 R_off=10k
.PARAM fc=28e9
.PARAM L_cs0_nh=0.3
.PARAM C_cs0_br_pf=0.1
.PARAM C_cs0_c_pf=0.1
.PARAM L_cs1_nh=0.3
.PARAM R_sw=3      $ FRAMEWORK_CONTROLLED

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

Vsrc src 0 DC 0.5 AC 1
Rsrc src in 50

Ccs0_br in n0 {C_cs0_br_pf*1e-12}
Ccs0_c cs0_m 0 {C_cs0_c_pf*1e-12}
Lcs0_1 in cs0_m {L_cs0_nh*1e-9}
Lcs0_2 cs0_m n0 {L_cs0_nh*1e-9}
Lcs1_ n0 out {L_cs1_nh*1e-9}
Rcs1_bp n0 out {R_sw}

Rload out 0 50

.control
op
ac lin 201 24G 32G

let s11 = (v(in) - 0.5) / 0.5
let s21 = (v(out) / 0.5)
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
