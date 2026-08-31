* Gen_T36 — composed phase shifter, 1-bit

* === STATE_TABLE ===
* bits=1
* ideal_step_deg=-90.0
* state_0: R_sw={R_on}
* state_1: R_sw={R_on}
* === END_STATE_TABLE ===

.PARAM R_on=3 R_off=10k
.PARAM fc=28e9
.PARAM L_cs0_nh=0.3
.PARAM L_c1_As0_mm=1.69
.PARAM Z0_c1_As0=50
.PARAM C_c1_Bs0_pf=0.1
.PARAM R_sw=3      $ FRAMEWORK_CONTROLLED

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

Vsrc src 0 DC 0.5 AC 1
Rsrc src in 50

Cc1_Bs0_sh c1_b_in 0 {C_c1_Bs0_pf*1e-12}
Lcs0_ in n0 {L_cs0_nh*1e-9}
Rc1_Bs0_th c1_b_in c1_b_out 50
Rc1_inA n0 c1_a_in {R_sw}
Rc1_inB n0 c1_b_in {R_sw}
Rc1_outA c1_a_out out {R_sw}
Rc1_outB c1_b_out out {R_sw}
Tc1_As0_ c1_a_in 0 c1_a_out 0 Z0={Z0_c1_As0} TD={L_c1_As0_mm*1e-3 / 1.897e8}

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
