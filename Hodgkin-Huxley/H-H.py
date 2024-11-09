import numpy as np
import matplotlib.pyplot as plt

'Implementing H-H model'
# Constants
delta_time = 20e-6  # (50 microseconds)
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
n, m, h = 0.31768, 0.05293, 0.59612

# Stimulus configurations
stim_amplitudes = [200, 100, 400]  # Original, half, and double
results = {}


def alpha_n(Vm): return 0.01 * (Vm + 55) / (1 - np.exp(-(Vm + 55) / 10))


def beta_n(Vm): return 0.125 * np.exp(-(Vm + 65) / 80)


def alpha_m(Vm): return 0.1 * (Vm + 40) / (1 - np.exp(-(Vm + 40) / 10))


def beta_m(Vm): return 4 * np.exp(-(Vm + 65) / 18)


def alpha_h(Vm): return 0.07 * np.exp(-(Vm + 65) / 20)


def beta_h(Vm): return 1 / (1 + np.exp(-(Vm + 35) / 10))


def HH(Vm, n, m, h, I_stim):
    IK = gK * (n ** 4) * (Vm - EK)
    INa = gNa * (m ** 3) * h * (Vm - ENa)
    IL = gL * (Vm - EL)
    Iion = IK + INa + IL
    dVm = (I_stim - Iion) / Cm
    dn = alpha_n(Vm) * (1 - n) - beta_n(Vm) * n
    dm = alpha_m(Vm) * (1 - m) - beta_m(Vm) * m
    dh = alpha_h(Vm) * (1 - h) - beta_h(Vm) * h
    return dVm, dn, dm, dh, IK, INa

'Q NO.22: Linearity'
for stim_amplitude in stim_amplitudes:
    Vm = Vm_init
    n, m, h = 0.31768, 0.05293, 0.59612  # Reset gating variables
    Vm_values = []

    for t in range(time_steps):
        current_time = t * delta_time
        I_stim = stim_amplitude if current_time <= stim_durations else 0
        dVm, dn, dm, dh, IK, INa = HH(Vm, n, m, h, I_stim)

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
amplitude_threshold = None
Vm_threshold = None
action_potential, action_potential_check = None, None
amp_range = np.linspace(0.1, 4, 100)
for stim_amplitude in amp_range:
    Vm = Vm_init
    n, m, h = 0.31768, 0.05293, 0.59612  # Reset gating variables
    Vm_values = []

    for t in range(time_steps):
        current_time = t * delta_time
        I_stim = (stim_amplitude) if current_time <= stim_durations else 0
        dVm, dn, dm, dh, IK, INa = HH(Vm, n, m, h, I_stim)

        Vm += dVm * delta_time
        n += dn * delta_time
        m += dm * delta_time
        h += dh * delta_time
        Vm_values.append(Vm)

        if Vm > -55:  # Threshold for action potential(mV)
            action_potential = True

    if action_potential:
        # Check by lowering amplitude by 1 μA/cm² to see if it fails to produce action potential
        Vm_check = Vm_init
        n_check, m_check, h_check = 0.31768, 0.05293, 0.59612  # Reset gating variables
        action_potential_test = False

        for t in range(time_steps):
            current_time = t * delta_time
            I_stim = (stim_amplitude - 1) if current_time <= stim_durations else 0
            dVm, dn, dm, dh = HH(Vm_check, n_check, m_check, h_check, I_stim)

            Vm_check += dVm * delta_time
            n_check += dn * delta_time
            m_check += dm * delta_time
            h_check += dh * delta_time

            if Vm > -55:
                action_potential_check = True
                break

    if not action_potential_check:
        threshold_amplitude = stim_amplitude
        vm_threshold = Vm_values[int(stim_durations / delta_time)]
        break

print(f"Just-above-threshold stimulus amplitude: {threshold_amplitude} μA/cm²")
print(f"Membrane voltage at the end of the stimulus (150 μsec): {vm_threshold:.2f} mV")

'Q NO.24: Time to peak'
stim_amplitude = [50, 200, 500]
results = {}
for stim_amplitude in stim_amplitudes:
    Vm = Vm_init
    n, m, h = 0.31768, 0.05293, 0.59612  # Reset gating variables
    Vm_values = []
    peak_time = None
    peak_value = -np.inf

    # Run the simulation
    for t in range(time_steps):
        current_time = t * delta_time
        I_stim = stim_amplitude if current_time <= stim_durations else 0
        dVm, dn, dm, dh, IK, INa = HH(Vm, n, m, h, I_stim)

        Vm += dVm * delta_time
        n += dn * delta_time
        m += dm * delta_time
        h += dh * delta_time
        Vm_values.append(Vm)

        if Vm > peak_value:
            peak_value = Vm
            peak_time = current_time

    results[stim_amplitude] = (peak_time * 1e6, peak_value)

# Output results
for stim_amplitude, (peak_time, peak_value) in results.items():
    print(f"Stimulus Amplitude: {stim_amplitude} μA/cm²")
    print(f"Time to peak: {peak_time:.2f} μsec")
    print(f"Peak Membrane Potential (Vm): {peak_value:.2f} mV\n")

'Evaluation of m,n, h gates, Vm, K and Na behavior over time'
stim_amplitude = 300.0  # Applied current (μA/cm^2)
Vm = Vm_init
n, m, h = 0.31768, 0.05293, 0.59612
Vm_values, n_values, m_values, h_values, IK_values, INa_values = [], [], [], [], [], []

for t in range(time_steps):
    current_time = t * delta_time
    I_stim = stim_amplitude if current_time <= stim_durations else 0
    dVm, dn, dm, dh, IK, INa = HH(Vm, n, m, h, I_stim)

    Vm += dVm * delta_time
    n += dn * delta_time
    m += dm * delta_time
    h += dh * delta_time
    Vm_values.append(Vm)
    n_values.append(n)
    m_values.append(m)
    h_values.append(h)
    IK_values.append(IK)
    INa_values.append(INa)

# Plotting
time_axis = np.arange(0, total_time, delta_time) * 1e6

fig, axs = plt.subplots(2, 2, figsize=(10, 9))

ax1 = plt.subplot(212)
ax1.plot(time_axis, Vm_values, label='Vm')
plt.xlabel('Time (μs)')
plt.ylabel('Membrane potential (mV)')
plt.title('Membrane Potential Vm')
plt.xlim(0, 1000)

ax2 = plt.subplot(221)
ax2.plot(time_axis, IK_values, label='IK')
ax2.margins(2, 1)
plt.xlabel('Time (μs)')
plt.ylabel('Current density (μA/cm^2)')
plt.title('K current densities')
plt.xlim(0, 1000)
plt.legend()

ax3 = plt.subplot(222)
ax3.plot(time_axis, INa_values, label='INa')
plt.title('Na current densities')
plt.xlim(0, 1000)
plt.legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 9))

axs[0].plot(time_axis, n_values, label='n(t)')
axs[0].set_xlabel('Time (μs)')
axs[0].set_ylabel('Gate value')
axs[0].set_title('K+ Gate Dynamics for n(t)')
axs[0].legend()

axs[1].plot(time_axis, m_values, label='m(t)')
axs[1].set_xlabel('Time (μs)')
axs[1].set_ylabel('Gate values')
axs[1].set_title('Na+ Gate Dynamics for m(t)')
axs[1].legend()

axs[2].plot(time_axis, h_values, label='h(t)')
axs[2].set_xlabel('Time (μs)')
axs[2].set_ylabel('Gate values')
axs[2].set_title('Na+ Gate Dynamics for h(t)')
axs[2].legend()

plt.tight_layout()
plt.show()
