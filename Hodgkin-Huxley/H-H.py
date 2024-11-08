import numpy as np
import matplotlib.pyplot as plt

'Implementing H-H model'
# Constants
delta_time = 25e-6  # (50 microseconds)
stim_durations = 150e-6  # (150 microseconds  duration that is minimally sufficient to produce an action potential)
total_time = 2000e-6  # (2000 microseconds)
time_steps = int(total_time / delta_time)

# Membrane and Hodgkin-Huxley parameters
Cm = 1.0
EK = -72.1
ENa = 52.4
EL = -49.187
gK = 36.0
gNa = 120.0
gL = 0.3
Vm_init = -60.0

# Initial gating variables
n = 0.31768
m = 0.05293
h = 0.59612

# Stimulus configurations
stim_amplitudes = [200, 100, 400]  # Original, half, and double
results = {}


def alpha_n(Vm):
    return 0.01 * (Vm + 55) / (1 - np.exp(-(Vm + 55) / 10))


def beta_n(Vm):
    return 0.125 * np.exp(-(Vm + 65) / 80)


def alpha_m(Vm):
    return 0.1 * (Vm + 40) / (1 - np.exp(-(Vm + 40) / 10))


def beta_m(Vm):
    return 4 * np.exp(-(Vm + 65) / 18)


def alpha_h(Vm):
    return 0.07 * np.exp(-(Vm + 65) / 20)


def beta_h(Vm):
    return 1 / (1 + np.exp(-(Vm + 35) / 10))


def HH(Vm, n, m, h, I_stim):
    IK = gK * (n ** 4) * (Vm - EK)
    INa = gNa * (m ** 3) * h * (Vm - ENa)
    IL = gL * (Vm - EL)
    Iion = IK + INa + IL
    dVm = (I_stim - Iion) / Cm
    dn = alpha_n(Vm) * (1 - n) - beta_n(Vm) * n
    dm = alpha_m(Vm) * (1 - m) - beta_m(Vm) * m
    dh = alpha_h(Vm) * (1 - h) - beta_h(Vm) * h
    return dVm, dn, dm, dh

'Q NO.22: Linearity'
for stim_amplitude in stim_amplitudes:
    Vm = Vm_init
    n, m, h = 0.31768, 0.05293, 0.59612  # Reset gating variables
    Vm_values = []

    for t in range(time_steps):
        current_time = t * delta_time
        I_stim = stim_amplitude if current_time <= stim_durations else 0
        dVm, dn, dm, dh = HH(Vm, n, m, h, I_stim)

        Vm += dVm * delta_time
        n += dn * delta_time
        m += dm * delta_time
        h += dh * delta_time
        Vm_values.append(Vm)

    results[stim_amplitude] = Vm_values

# Plotting results
time_axis = np.arange(0, total_time, delta_time) * 1e6  # in microseconds
plt.figure(figsize=(10, 6))

for stim_amplitude, Vm_values in results.items():
    plt.plot(time_axis, Vm_values, label=f'Stimulus = {stim_amplitude} μA')

plt.xlabel("Time (μsec)")
plt.ylabel("Membrane Potential (Vm)")
plt.legend()
plt.title("Hodgkin-Huxley Model Response to Different Stimulus Amplitudes")
plt.show()

# Check linearity by comparing Vm values at specific times
t_100_index = int(100e-6 / delta_time)
t_200_index = int(200e-6 / delta_time)

# Analysis: Compare Vm values for linearity
for stim_amplitude in stim_amplitudes:
    Vm_100 = results[stim_amplitude][t_100_index]
    Vm_2000 = results[stim_amplitude][-1]  # at the end of the 2000 μsec period
    print(f"Stimulus: {stim_amplitude} μA, Vm at t ≤ 100 μsec: {Vm_100:.2f}, Vm at t > 200 μsec: {Vm_2000:.2f}")

'Q NO.23: Threshold'
stim_amplitude_range = np.linspace(100, 800, 1400)
amplitude_threshold = None
Vm_threshold = None
for stim_amplitude in stim_amplitude_range:
    Vm = Vm_init
    n, m, h = 0.31768, 0.05293, 0.59612  # Reset gating variables
    Vm_values = []

    for t in range(time_steps):
        current_time = t * delta_time
        I_stim = stim_amplitude if current_time <= stim_durations else 0
        dVm, dn, dm, dh = HH(Vm, n, m, h, I_stim)

        Vm += dVm * delta_time
        n += dn * delta_time
        m += dm * delta_time
        h += dh * delta_time
        Vm_values.append(Vm)

        # Threshold for action potential(mV)
        if Vm >= -55:
            Vm_check = Vm + dVm * delta_time
            dVm, dn, dm, dh = HH(Vm, n, m, h, I_stim - 1)
            Vm_check += dVm * delta_time
            if Vm_check < -55:
                amplitude_threshold = I_stim
                # Membrane voltage at the end of stimulus
                Vm_threshold = Vm_values[-1]
                break

# Output results
print(f"Just-above-threshold stimulus amplitude: {amplitude_threshold} μA/cm²")
print(f"Membrane voltage when just-above-threshold stimulus ends: {Vm_threshold} mV")

'Q NO.24: Time to peak'
'Q NO.25: Time to return to initial conditions:'
'Q NO.26: Leakage gL'
'Q NO.27 AP from 2nd stimulus:'
