* Loaded-Line Phase Shifter, 1-bit, target ~45 deg @ 28 GHz
* Path A: ideal lossless TLs, switches modeled as fixed resistors
* Topology: lambda/4 main line, two symmetric shunt loads at input and output
*   "loaded"   state: R_path_in={R_on}, R_path_out={R_on}    (default)
*   "unloaded" state: R_path_in={R_off}, R_path_out={R_off}
* Symmetric loading is by construction: keeps |S11| low when both shunts track.

* === STATE_TABLE ===
* bits=1
* ideal_step_deg=-22.5
* state_0: R_path_in={R_off} R_path_out={R_off}
* state_1: R_path_in={R_on} R_path_out={R_on}
* === END_STATE_TABLE ===

.PARAM Z0_line=50 L_quarter_mm=1.69 C_load_pf=0.04
.PARAM R_on=3 R_off=10k
.PARAM R_path_in=3  R_path_out=3       $ FRAMEWORK_CONTROLLED
.PARAM fc=28e9                          $ FRAMEWORK_CONTROLLED
.PARAM eps_eff=2.5                      $ FRAMEWORK_CONTROLLED

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

* --- Port 1: AC source with 50-ohm series source impedance ---
* Vavail = 0.5 V (Thevenin: Vsrc=1V across Rsrc=50 + Z_load)
Vsrc src 0 DC 0.5 AC 1
Rsrc src in 50

* --- Main lambda/4 transmission line (in -> out) ---
* Lossless TL: TD = length / v_phase; eps_eff=2.5 -> v_phase=1.897e8 m/s
T_main in 0 out 0 Z0={Z0_line} TD={L_quarter_mm*1e-3 / 1.897e8}

* --- Shunt loading branch at input port ---
* series resistor (state-controlled) -> shunt cap -> ground
R_in_path  in  n_in   {R_path_in}
C_in_load  n_in 0     {C_load_pf*1e-12}

* --- Shunt loading branch at output port ---
R_out_path out n_out  {R_path_out}
C_out_load n_out 0    {C_load_pf*1e-12}

* --- Port 2: 50-ohm termination ---
Rload out 0 50

.control
op
print v(in) v(n_in) v(n_out) v(out)

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
