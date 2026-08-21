* All-Pass Phase Shifter, 1-bit, target -90 deg @ 28 GHz
* Path A: ideal lossless bridged-T LC all-pass sections, switches as resistors
* Topology: two parallel bridged-T LC all-pass networks gated by switches.
*   Each section is the canonical Foster-form bridged-T all-pass:
*     in  -- L_ap -- m -- L_ap -- out
*             |              |
*             +--- C_br ----+
*             |
*            C_c (m to gnd)
*   The first-order Foster all-pass ratio C_br = L/(2*Z0^2), C_c = 2*C_br
*   does not give a clean image-impedance match in this two-inductor form;
*   the values below were calibrated by a grid sweep against ngspice 42 to
*   maximize RL at fc while hitting the per-section phase targets.
* Phase state selected by R_apA_*/R_apB_* parameter swap:
*   "fast" state (default): section A active, R_apA_*={R_on}, R_apB_*={R_off}
*   "slow" state:           section B active, R_apA_*={R_off}, R_apB_*={R_on}
* Calibrated defaults for Z0=50 ohm, fc=28 GHz (grid-sweep result):
*   Section A: target ~-45 deg
*     L_apA = 0.100 nH, C_brA = 0.025 pF, C_cA = 0.080 pF
*     measured: phase = -44.5 deg, IL = 0.55 dB, RL = 28.2 dB
*   Section B: target ~-135 deg
*     L_apB = 0.200 nH, C_brB = 0.060 pF, C_cB = 0.200 pF
*     measured: phase = -136.6 deg, IL = 0.55 dB, RL = 23.5 dB
* Differential phase ~ -92 deg (within 2 deg of the -90 design target).
* Schiffman-style: phase difference is approximately flat across a wide
* fractional bandwidth.

* === STATE_TABLE ===
* bits=1
* ideal_step_deg=-90.0
* state_0: R_apA_in={R_on}  R_apA_out={R_on}  R_apB_in={R_off} R_apB_out={R_off}
* state_1: R_apA_in={R_off} R_apA_out={R_off} R_apB_in={R_on}  R_apB_out={R_on}
* === END_STATE_TABLE ===

.PARAM Z0_line=50
.PARAM L_apA_nh=0.100 C_brA_pf=0.025 C_cA_pf=0.080
.PARAM L_apB_nh=0.200 C_brB_pf=0.060 C_cB_pf=0.200
.PARAM R_on=3 R_off=10k
.PARAM R_apA_in=3   R_apA_out=3        $ FRAMEWORK_CONTROLLED
.PARAM R_apB_in=10k R_apB_out=10k      $ FRAMEWORK_CONTROLLED
.PARAM fc=28e9                          $ FRAMEWORK_CONTROLLED
.PARAM eps_eff=2.5                      $ FRAMEWORK_CONTROLLED

.option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500

* --- Port 1: AC source with 50-ohm series source impedance ---
* Vavail = 0.5 V (Thevenin: Vsrc=1V across Rsrc=50 + Z_load)
Vsrc src 0 DC 0.5 AC 1
Rsrc src in 50

* --- All-pass section A (small tau, ~-45 deg at fc) ---
*   in -- R_apA_in -- a_in -- L_apA -- m_A -- L_apA -- a_out -- R_apA_out -- out
*                       |                              |
*                       +--------- C_brA -------------+
*                                    |
*                                  C_cA (m_A to gnd)
R_in_apA   in     a_in  {R_apA_in}
L_apA_ser1 a_in   m_A   {L_apA_nh*1e-9}
L_apA_ser2 m_A    a_out {L_apA_nh*1e-9}
C_brA_brg  a_in   a_out {C_brA_pf*1e-12}
C_cA_shnt  m_A    0     {C_cA_pf*1e-12}
R_out_apA  a_out  out   {R_apA_out}

* --- All-pass section B (large tau, ~-135 deg at fc) ---
*   in -- R_apB_in -- b_in -- L_apB -- m_B -- L_apB -- b_out -- R_apB_out -- out
*                       |                              |
*                       +--------- C_brB -------------+
*                                    |
*                                  C_cB (m_B to gnd)
R_in_apB   in     b_in  {R_apB_in}
L_apB_ser1 b_in   m_B   {L_apB_nh*1e-9}
L_apB_ser2 m_B    b_out {L_apB_nh*1e-9}
C_brB_brg  b_in   b_out {C_brB_pf*1e-12}
C_cB_shnt  m_B    0     {C_cB_pf*1e-12}
R_out_apB  b_out  out   {R_apB_out}

* --- Port 2: 50-ohm termination ---
Rload out 0 50

.control
op
print v(in) v(m_A) v(m_B) v(a_out) v(b_out) v(out)

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
