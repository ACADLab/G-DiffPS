* Gen_T18 — composed phase shifter, 1-bit

* === STATE_TABLE ===
* bits=1
* ideal_step_deg=-90.0
* state_0: R_sw={R_on}
* state_1: R_sw={R_on}
* === END_STATE_TABLE ===

.PARAM R_on=3 R_off=10k
.PARAM fc=28e9
.PARAM C_c0_As0_pf=0.1
.PARAM L_c0_As1_nh=0.3
.PARAM L_c0_Bs0_nh=0.3
.PARAM C_c0_Bs0_br_pf=0.1
.PARAM C_c0_Bs0_c_pf=0.1
.PARAM R_sw=3      $ FRAMEWORK_CONTROLLED

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

Vsrc src 0 DC 0.5 AC 1
Rsrc src in 50

Cc0_As0_sh c0_a_in 0 {C_c0_As0_pf*1e-12}
Cc0_Bs0_br c0_b_in c0_b_out {C_c0_Bs0_br_pf*1e-12}
Cc0_Bs0_c c0_Bs0_m 0 {C_c0_Bs0_c_pf*1e-12}
Lc0_As1_sh c0_A_n0 0 {L_c0_As1_nh*1e-9}
Lc0_Bs0_1 c0_b_in c0_Bs0_m {L_c0_Bs0_nh*1e-9}
Lc0_Bs0_2 c0_Bs0_m c0_b_out {L_c0_Bs0_nh*1e-9}
Rc0_As0_th c0_a_in c0_A_n0 50
Rc0_As1_th c0_A_n0 c0_a_out 50
Rc0_inA in c0_a_in {R_sw}
Rc0_inB in c0_b_in {R_sw}
Rc0_outA c0_a_out out {R_sw}
Rc0_outB c0_b_out out {R_sw}

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
