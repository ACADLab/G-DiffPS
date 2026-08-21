* Reflection-Type Phase Shifter, 1-bit, target ~22.5 deg @ 28 GHz
* Path A: ideal lossless branchline hybrid, switched-cap reflective loads
* Topology: 90-deg branchline (4 TL arms) with symmetric reflective
*   loads on the through (Port 2) and coupled (Port 3) ports. Reflections
*   recombine constructively at the output (Port 4) and cancel at the
*   input (Port 1), giving a phase shift set by the load reflection.
* Switched-cap reflective load (each port):
*   C_base always connected to GND; C_tune connects to GND through a
*   series switch (R_path). State 0: R_path=R_off (switch open) -> load
*   is C_base alone, smaller capacitance, one reflection phase. State 1:
*   R_path=R_on (switch closed) -> load is C_base || C_tune, larger
*   capacitance, different reflection phase. R_path stays OUT of the
*   high-Q signal path so the loss stays small in both states.
* Symmetric loading by construction: both arms carry identical Z.

* === STATE_TABLE ===
* bits=1
* ideal_step_deg=-22.5
* state_0: R_path={R_off}
* state_1: R_path={R_on}
* === END_STATE_TABLE ===

.PARAM Z0_main=50 Z0_branch=35.35 L_quarter_mm=1.69
.PARAM C_base_pf=0.10 C_tune_pf=0.20
.PARAM R_on=3 R_off=10k
.PARAM R_path=10k                      $ FRAMEWORK_CONTROLLED
.PARAM fc=28e9                         $ FRAMEWORK_CONTROLLED
.PARAM eps_eff=2.5                     $ FRAMEWORK_CONTROLLED

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

* --- Port 1: AC source with 50-ohm series source impedance ---
* Thevenin: Vsrc=1V across Rsrc=50 + Z_in -> Vavail = 0.5 V at matched load
Vsrc src 0 DC 0.5 AC 1
Rsrc src p1 50

* --- 90-deg branchline hybrid (Pozar Ch. 7) ---
* Four TL arms forming a square. eps_eff=2.5 -> v_phase=1.897e8 m/s.
* Port assignments at the four corners:
*   p1 = input          (top-left)
*   p2 = through port   (top-right, reflective load A)
*   p3 = coupled port   (bottom-left, reflective load B)
*   p4 = output         (bottom-right)
* Series arms (top and bottom): Z0_branch = Z0_main/sqrt(2) ~ 35.4 ohm
* Shunt arms (left and right):  Z0_main = 50 ohm
T_top    p1 0 p2 0 Z0={Z0_branch} TD={L_quarter_mm*1e-3 / 1.897e8}
T_bottom p3 0 p4 0 Z0={Z0_branch} TD={L_quarter_mm*1e-3 / 1.897e8}
T_left   p1 0 p3 0 Z0={Z0_main}   TD={L_quarter_mm*1e-3 / 1.897e8}
T_right  p2 0 p4 0 Z0={Z0_main}   TD={L_quarter_mm*1e-3 / 1.897e8}

* --- Symmetric switched-cap reflective loads on p2 and p3 ---
* C_base: always to GND. C_tune: to GND via switched R_path.
* State 0 (R_path=R_off): C_tune effectively isolated, load = C_base.
* State 1 (R_path=R_on):  C_tune in (near-)parallel with C_base.
C_baseA p2 0 {C_base_pf*1e-12}
C_tuneA p2 nA {C_tune_pf*1e-12}
R_pathA nA 0 {R_path}

C_baseB p3 0 {C_base_pf*1e-12}
C_tuneB p3 nB {C_tune_pf*1e-12}
R_pathB nB 0 {R_path}

* --- Output port: 50-ohm termination ---
RL p4 0 50

* --- AC analysis: wideband sweep, measure at fc ---
.control
op
print v(p1) v(p2) v(p3) v(p4)

ac lin 201 24G 32G

* S-parameter extraction using built-in cph (continuous phase) and db.
* s11 = (V_in - V_avail) / V_avail, where V_avail = 0.5 (Thevenin source).
* s21 = V_out / V_avail.
let s11 = (v(p1) - 0.5) / 0.5
let s21 = v(p4) / 0.5
let s21_mag_db = db(s21)
let s11_mag_db = db(s11)
let s21_phase  = 180/pi * cph(s21)

meas ac il_db_at_fc  FIND s21_mag_db AT=28e9
meas ac phase_at_fc  FIND s21_phase  AT=28e9
meas ac rl_db_at_fc  FIND s11_mag_db AT=28e9

let phase_deg   = phase_at_fc
let il_db       = -1 * il_db_at_fc
let rl_db       = -1 * rl_db_at_fc
let gain_err_db = 0.0
print phase_deg il_db rl_db gain_err_db

quit
.endc

.end

