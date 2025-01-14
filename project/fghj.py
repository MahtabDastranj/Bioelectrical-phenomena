import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --------------------------------------------------
# 1) MODEL PARAMETERS
# --------------------------------------------------

L = 10e-6  # 10 micrometers side
surface_area = 6.0 * (L ** 2)  # 6 sides * L^2
volume = L ** 3
k = 3
e = 8.85e-12
d = 1e-9
Cm = k * e * surface_area / d  # ~0.01 pF, for a small cell

# Single-channel conductance, number of channels, etc.
g_single_channel = 20e-12  # 20 pS
channel_density = 10.0  # channels / µm^2 (example guess - user can vary)
N_channels = channel_density * (surface_area * 1e12)  # area in µm^2
g_max = N_channels * g_single_channel  # total

Erev = 135.0  # mV, reversal potential
R_ext = 1e12
# Pulsed external field parameters
pulse_frequency = 5.0  # Hz
pulse_period = 1.0 / pulse_frequency  # 0.2 s
pulse_width = 0.1  # s "on" time within each 0.2 s period
E_amplitude = 500.0  # V/m
Scale = 5e3
# We'll model the induced external current in the ODE:
I0 = 0  # 1 nA peak when field is ON

# Simulation time
t_max = 2.0  # seconds, enough to see multiple pulses
dt = 1e-5  # time step for ODE solver output
t_eval = np.arange(0, t_max, dt)


# --------------------------------------------------
# 2) HELPER FUNCTIONS FOR THE PULSED FIELD
# --------------------------------------------------

def pulse_wave(t, period, width):
    """
    Returns 1 if (t mod period) < width, else 0.
    E.g., for period=0.2 s, width=0.05 s,
    we have a 25% duty cycle.
    """
    mod_t = t % period
    return 1.0 if (mod_t < width) else 0.0


def alpha_m(v_mV):
    """Alpha rate for gating variable m (units: 1/ms)."""
    return 8.5 / (1.0 + np.exp((v_mV - 8.0) / -12.5))


def beta_m(v_mV):
    """Beta rate for gating variable m (units: 1/ms)."""
    return 35.0 / (1.0 + np.exp((v_mV + 74.0) / 14.5))


def dALLdt(t, Y):
    """
    Y = [V, m]
    We return [dV/dt (mV/s), dm/dt].
    """
    V = Y[0]  # (mV)
    m = Y[1]  # gating var

    # Gating variable ODE
    a_m = alpha_m(V)
    b_m = beta_m(V)
    dm_dt = a_m * (1.0 - m) - b_m * m

    # Ionic current
    I_Ca = g_max * m * ((V - Erev) * 1e-3)  # convert mV->V in the factor

    # Determine if the field is ON or OFF
    # If ON, E(t) = E_amplitude, else 0
    E_t = E_amplitude * pulse_wave(t, pulse_period, pulse_width)

    # Convert that E-field to an external current: I_ext = I0 * wave
    # (Here, we simply use wave(t) to multiply I0)
    wave_t = pulse_wave(t, pulse_period, pulse_width)
    I_ext = E_t * L / R_ext * wave_t  # (A)

    # Membrane equation in SI: Cm * dV/dt (V/s) = -I_Ca + I_ext
    dV_dt_SI = (-I_Ca + I_ext) / Cm  # in V/s
    dV_dt_mV = dV_dt_SI * 1e3  # convert to mV/s

    return [dV_dt_mV, dm_dt]


# --------------------------------------------------
# 3) SOLVE THE ODE
# --------------------------------------------------
V0 = -70.0
m0_inf = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
Y0 = [V0, m0_inf]

sol = solve_ivp(dALLdt, [0, t_max], Y0, t_eval=t_eval, method='RK45')
t_sol = sol.t
V_sol = sol.y[0]
m_sol = sol.y[1]

# --------------------------------------------------
# 4) BUILD ARRAYS FOR PLOTTING
# --------------------------------------------------
# (A) E-field vs time (pulsed)
E_array = np.array([
    E_amplitude * pulse_wave(tt, pulse_period, pulse_width)
    for tt in t_sol
])
E_t = E_array
I_ext = E_t * L / R_ext
# (B) Ionic current
I_Ca_array = g_max * m_sol * ((V_sol - Erev) * 1e-3)

# --------------------------------------------------
# 5) FOURIER ANALYSIS (STEADY-STATE)
# --------------------------------------------------
t_min_fft = 1  # skip first 0.2 s to ignore transients
idx_min_fft = np.where(t_sol >= t_min_fft)[0][0]

# (C) DIPOLAR APPROXIMATION & VOLTAGE RECORDING
# ---------------------------------------------
# We treat the cell as a dipole whose moment is proportional to I_Ca_array.
# Two electrodes are at distance ~1 cm from the cell. We assume an effective
# dipole length for the cell, then compute recorded voltage at distance r.
epsilon_0 = 8.854e-12  # permittivity of free space (F/m)
r = 0.01               # 1 cm
dipole_length = 1e-5   # arbitrary effective distance inside the cell (m)
# Dipole potential V_dipole_array(t) ~ p(t)/(4*pi*eps_0*r^2)
# where p(t) ~ I_Ca_array(t)*dipole_length (simplified).

V_dipole_array = (np.abs(I_Ca_array) * dipole_length) / (4.0 * np.pi * epsilon_0 * r**2)

# (D) HARMONIC ANALYSIS FOR CELL'S MEMBRANE POTENTIAL
V_for_fft = V_sol[idx_min_fft:]
dt = t_sol[1] - t_sol[0]
fft_data = np.fft.fft(V_for_fft)
freqs = np.fft.fftfreq(len(V_for_fft), dt)
# (E) HARMONIC ANALYSIS FOR RECORDED DIPOLAR VOLTAGE
Vdipole_for_fft = V_dipole_array[idx_min_fft:]/Scale
fft_dipole = np.fft.fft(Vdipole_for_fft)
freqs_dipole = np.fft.fftfreq(len(Vdipole_for_fft), dt)
# (F) PLOTTING THE FREQUENCY SPECTRA
plt.figure(figsize=(10, 5))
plt.plot(freqs_dipole, np.abs(fft_dipole), label="Dipole V_recorded(f)")
plt.xlim(2, 20)  # adjust as needed
plt.ylim(0, 0.0001)  # adjust as needed
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Fourier Spectra: Membrane vs. Dipole Recording")
plt.legend()
plt.show()
fund_freq = 1.0 / pulse_period  # or define as needed
harmonics = [2*fund_freq, 3*fund_freq]

for hfreq in harmonics:
    idx_mem = np.argmin(np.abs(freqs - hfreq))
    idx_dip = np.argmin(np.abs(freqs_dipole - hfreq))
    amp_mem = np.abs(fft_data[idx_mem])
    amp_dip = np.abs(fft_dipole[idx_dip])
    print(f"Harmonic {hfreq:.2f} Hz -> Membrane amplitude: {amp_mem:.3e}, Dipole amplitude: {amp_dip:.3e}")


# --------------------------------------------------
# 6) PLOTTING: FOUR PANELS
# --------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# Panel (1): Pulsed Electric Field
axes[0,0].plot(t_sol, E_array, 'b-')
axes[0,0].set_title("Applied Electric Field (Pulsed)")
axes[0,0].set_xlabel("Time (s)")
axes[0,0].set_ylabel("Field (V/m)")

# Panel (2): Membrane Potential
axes[0,1].plot(t_sol, V_sol, 'r-')
axes[0,1].set_title("Transmembrane Voltage")
axes[0,1].set_xlabel("Time (s)")
axes[0,1].set_ylabel("Voltage (mV)")

# Panel (3): Ionic Current
axes[1,0].plot(t_sol, I_Ca_array*1e9, 'g-')  # in nA
axes[1,0].set_title("Ionic Current (Cav2.1)")
axes[1,0].set_xlabel("Time (s)")
axes[1,0].set_ylabel("Current (nA)")

# Panel (4): Dipole Voltage
axes[1,1].plot(t_sol, V_dipole_array, 'k-')
axes[1,1].set_title("Dipole Voltage")
axes[1,1].set_xlabel("Time (s)")
axes[1,1].set_ylabel("Voltage (V)")

plt.tight_layout()
plt.show()