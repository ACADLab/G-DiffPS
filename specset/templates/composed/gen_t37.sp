* Gen_T37 — composed phase shifter, 1-bit

* === STATE_TABLE ===
* bits=1
* ideal_step_deg=-90.0
* state_0: R_sw={R_on}
* state_1: R_sw={R_on}
* === END_STATE_TABLE ===

.PARAM R_on=3 R_off=10k
.PARAM fc=28e9
.PARAM L_c0_As0_nh=0.3
.PARAM L_c0_Bs0_mm=1.69
.PARAM Z0_c0_Bs0=50
.PARAM L_cs1_nh=0.3
.PARAM R_sw=3      $ FRAMEWORK_CONTROLLED

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

Vsrc src 0 DC 0.5 AC 1
Rsrc src in 50

Lc0_As0_ c0_a_in c0_a_out {L_c0_As0_nh*1e-9}
Lcs1_ n0 out {L_cs1_nh*1e-9}
Rc0_inA in c0_a_in {R_sw}
Rc0_inB in c0_b_in {R_sw}
Rc0_outA c0_a_out n0 {R_sw}
Rc0_outB c0_b_out n0 {R_sw}
Rcs1_bp n0 out {R_sw}
Tc0_Bs0_ c0_b_in 0 c0_b_out 0 Z0={Z0_c0_Bs0} TD={L_c0_Bs0_mm*1e-3 / 1.897e8}

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
