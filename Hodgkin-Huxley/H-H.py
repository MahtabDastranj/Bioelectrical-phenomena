import numpy as np
import matplotlib.pyplot as plt

'Implementing H-H model'
# Constants
delta_time = 0.05
stim_durations = 150  # (150 microseconds  duration that is minimally sufficient to produce an action potential)
total_time = 2000  # (2000 microseconds)
start_time = 100
stim_end_time = start_time + stim_durations
time_steps = int(total_time / delta_time)

# Membrane and Hodgkin-Huxley parameters
Cm = 1.0
EK, ENa, EL = -72.1, 52.4, -49.187
gK, gNa, gL = 36.0, 120.0, 0.3
Vm_init = -60.0
n_init, m_init, h_init = 0.31768, 0.05293, 0.59612


def alpha_n(Vm): return 0.01 * (Vm + 50) / (1 - np.exp(-(Vm + 50) / 10))


def beta_n(Vm): return 0.125 * np.exp(-(Vm + 60) / 80)


def alpha_m(Vm): return 0.1 * (Vm + 35) / (1 - np.exp(-(Vm + 35) / 10))


def beta_m(Vm): return 4 * np.exp(-(Vm + 60) / 18)


def alpha_h(Vm): return 0.07 * np.exp(-(Vm + 60) / 20)


def beta_h(Vm): return 1 / (1 + np.exp(-(Vm + 30) / 10))


def HH(delta_time, total_time, stim_amplitude):
    t_values = np.arange(0, total_time, delta_time)
    Vm_values = np.zeros(len(t_values))
    m_values = np.zeros(len(t_values))
    h_values = np.zeros(len(t_values))
    n_values = np.zeros(len(t_values))
    IK = np.zeros(len(t_values))
    INa = np.zeros(len(t_values))
    IL = np.zeros(len(t_values))

    Vm_values[0] = Vm_init
    m_values[0] = m_init
    h_values[0] = h_init
    n_values[0] = n_init

    stable_time = None
    time_since_stable = 0

    for t in range(len(t_values) - 1):
        I_stim = stim_amplitude if start_time <= t * delta_time <= start_time + stim_durations else 0
        m = m_values[t] + delta_time * (
                    alpha_m(Vm_values[t]) * (1 - m_values[t]) - beta_m(Vm_values[t]) * m_values[t])
        h = h_values[t] + delta_time * (
                    alpha_h(Vm_values[t]) * (1 - h_values[t]) - beta_h(Vm_values[t]) * h_values[t])
        n = n_values[t] + delta_time * (
                    alpha_n(Vm_values[t]) * (1 - n_values[t]) - beta_n(Vm_values[t]) * n_values[t])

        IK[t] = gK * (n_values[t] ** 4) * (Vm_values[t] - EK)
        INa[t] = gNa * (m_values[t] ** 3) * h_values[t] * (Vm_values[t] - ENa)
        IL[t] = gL * (Vm_values[t] - EL)
        Iion = IK[t] + INa[t] + IL[t]
        dVm = delta_time / Cm * (I_stim - Iion)
        Vm_values[t + 1] = Vm_values[t] + dVm
        m_values[t + 1] = m
        h_values[t + 1] = h
        n_values[t + 1] = n

        'Stability time'
        if t * delta_time >= stim_end_time:
            Vm_return = abs(Vm_values[t + 1] - Vm_init) <= 0.1
            n_return = abs(n_values[t + 1] - n_init) <= 0.01
            m_return = abs(m_values[t + 1] - m_init) <= 0.01
            h_return = abs(h_values[t + 1] - h_init) <= 0.01

            if Vm_return and n_return and m_return and h_return:
                if time_since_stable == 0:
                    time_since_stable = t_values[t + 1]
                if stable_time is None:
                    stable_time = time_since_stable - start_time
            else:
                time_since_stable = 0
                stable_time = None


    return t_values, Vm_values, m_values, h_values, n_values, IK, INa, IL, stable_time

'Q NO.22: Linearity'
stim_amplitudes = [100, 200, 400]
t1, Vm1, _, _, _, _, _, _, _ = HH(delta_time, total_time, stim_amplitudes[0])
t2, Vm2, _, _, _, _, _, _, _ = HH(delta_time, total_time, stim_amplitudes[1])
t3, Vm3, _, _, _, _, _, _, _ = HH(delta_time, total_time, stim_amplitudes[2])

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].plot(t1, Vm1, 'b', label='Vm')
axs[0].set_title(f'Half Amplitude Stimulus ({stim_amplitudes[0]:.1f} μA/mm^2)')
axs[0].set_ylabel('Membrane Potential (mV)')
axs[0].grid()
axs[0].set_xlim([0, 600])
axs[0].set_ylim([-80, 100])
axs[0].legend()

axs[1].plot(t2, Vm2, 'b', label='Vm')
axs[1].set_title(f'Original Amplitude Stimulus ({stim_amplitudes[1]:.1f} μA/mm^2)')
axs[1].set_ylabel('Membrane Potential (mV)')
axs[1].grid()
axs[1].set_xlim([0, 600])
axs[1].set_ylim([-80, 100])
axs[1].legend()

axs[2].plot(t3, Vm3, 'b', label='Vm')
axs[2].set_title(f'Double Amplitude Stimulus ({stim_amplitudes[2]:.1f} μA/mm^2)')
axs[2].set_xlabel('Time (ms)')
axs[2].set_ylabel('Membrane Potential (mV)')
axs[2].grid()
axs[2].set_xlim([0, 600])
axs[2].set_ylim([-80, 100])
axs[2].legend()

plt.tight_layout()
plt.show()

# Linearity
print('For t ≤ 100 μsec:')
print(f'Half stimulus ratio: {np.mean(Vm1[:int(0.1 / delta_time)])/np.mean(Vm2[:int(0.1 / delta_time)]):.3f} (Expected: 0.5)')
print(f'Double stimulus ratio: {np.mean(Vm3[:int(0.1 / delta_time)])/np.mean(Vm2[:int(0.1 / delta_time)]):.3f} (Expected: 2.0)')
print('\nFor t > 200 μsec:')
print(f'Half stimulus ratio: {np.mean(Vm1[int(0.2 / delta_time):])/np.mean(Vm2[int(0.2 / delta_time):]):.3f} (Expected: 0.5)')
print(f'Double stimulus ratio: {np.mean(Vm3[int(0.2 / delta_time):])/np.mean(Vm2[int(0.2 / delta_time):]):.3f} (Expected: 2.0)\n')

'Q NO.23: Threshold'


def AP_check(Vm_values):
    threshold = -55
    return np.any(Vm_values >= threshold) and np.max(Vm_values) > -40


# Threshold detection
threshold_found = False
amplitude_threshold, Vm_threshold = None, None
amp, max_amp = 0, 100

while not threshold_found and amp < max_amp:
    amp += 0.5
    print(f'Checking amplitude {amp} as the threshold')
    _, Vm, _, _, _, _, _, _, _ = HH(delta_time, total_time, amp)
    ap = AP_check(Vm)

    if ap:
        _, Vm_check, _, _, _, _, _, _, _ = HH(delta_time, total_time, amp - 1)
        threshold_check = AP_check(Vm_check)
        if not threshold_check:
            threshold_found = True
            amplitude_threshold = amp
            Vm_threshold = Vm[int(stim_end_time / delta_time)]
            break

if threshold_found:
    print(f"Threshold stimulus amplitude: {amplitude_threshold:.2f} μA/cm²")
    print(f"Membrane voltage at the end of the stimulus (150 μsec): {Vm_threshold:.2f} mV")
# else: print("Threshold amplitude not found within the tested range.")

'Q NO.24: Time to peak'
Vm = []
stim_amplitudes = [50, 200, 500]
for stim_amplitude in stim_amplitudes:
    t_values, Vm, _, _, _, _, _, _, _ = HH(delta_time, total_time, stim_amplitude)
    max_index = np.argmax(Vm)
    time_at_max_Vm = t_values[max_index]
    print(f'\nFor stimulation amplitude {stim_amplitude} μA/cm², '
          f'maximum membrane voltage {np.max(Vm)} will hit its maximum amount at {time_at_max_Vm}')

'Q NO.25: Time to return to initial conditions'
stim_amplitude = 200  # Example stimulus
t_values, Vm_values, _, _, _, _, _, _, stable_time = HH(delta_time, total_time, stim_amplitude)

# Output the stable return time
if stable_time is not None:
    print(f'Stability return time for stimulation amplitude of {stim_amplitude}: {stable_time} μs')
else:
    print('Variables did not return to stable conditions within the simulation time.')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, Vm_values, label='Membrane Voltage (Vm)')
if stable_time is not None:
    plt.axvline(x=stable_time, color='m', linestyle='--', label='Stability Return Time')
    plt.text(stable_time, Vm_values[np.argmin(np.abs(t_values - stable_time))], f'{stable_time:.2f} μs', color='m')
plt.title('Membrane Potential Over Time')
plt.xlabel('Time (μs)')
plt.ylabel('Membrane Voltage (mV)')
plt.xlim([0, 250])  # Adjust as needed
plt.grid()
plt.legend()
plt.show()

'Q NO.26: Leakage gL'
stim = 200
def HH_EL(delta_time, total_time, stim_amplitude):
    t_values = np.arange(0, total_time, delta_time)
    Vm_values = np.zeros(len(t_values))
    m_values = np.zeros(len(t_values))
    h_values = np.zeros(len(t_values))
    n_values = np.zeros(len(t_values))
    IK = np.zeros(len(t_values))
    INa = np.zeros(len(t_values))
    IL = np.zeros(len(t_values))

    Vm_values[0] = Vm_init
    m_values[0] = m_init
    h_values[0] = h_init
    n_values[0] = n_init
    EL = (Vm_init * (gNa * m_values[0] ** 3 * h_values[0] + gK * n_values[0] ** 4 + gL) - (
            gNa * m_values[0] ** 3 * h_values[0] * ENa + gK * n_values[0] ** 4 * EK)) / gL

    stable_time = None
    time_since_stable = 0

    for t in range(len(t_values) - 1):
        I_stim = stim_amplitude if start_time <= t * delta_time <= start_time + stim_durations else 0
        m = m_values[t] + delta_time * (
                    alpha_m(Vm_values[t]) * (1 - m_values[t]) - beta_m(Vm_values[t]) * m_values[t])
        h = h_values[t] + delta_time * (
                    alpha_h(Vm_values[t]) * (1 - h_values[t]) - beta_h(Vm_values[t]) * h_values[t])
        n = n_values[t] + delta_time * (
                    alpha_n(Vm_values[t]) * (1 - n_values[t]) - beta_n(Vm_values[t]) * n_values[t])

        IK[t] = gK * (n_values[t] ** 4) * (Vm_values[t] - EK)
        INa[t] = gNa * (m_values[t] ** 3) * h_values[t] * (Vm_values[t] - ENa)
        IL[t] = gL * (Vm_values[t] - EL)
        Iion = IK[t] + INa[t] + IL[t]
        dVm = delta_time / Cm * (I_stim - Iion)
        Vm_values[t + 1] = Vm_values[t] + dVm
        m_values[t + 1] = m
        h_values[t + 1] = h
        n_values[t + 1] = n

        'Stability time'
        if t * delta_time >= stim_end_time:
            Vm_return = abs(Vm_values[t + 1] - Vm_init) <= 0.1
            n_return = abs(n_values[t + 1] - n_init) <= 0.01
            m_return = abs(m_values[t + 1] - m_init) <= 0.01
            h_return = abs(h_values[t + 1] - h_init) <= 0.01

            if Vm_return and n_return and m_return and h_return:
                if time_since_stable == 0:
                    time_since_stable = t_values[t + 1]
                if stable_time is None:
                    stable_time = time_since_stable - start_time
            else:
                time_since_stable = 0
                stable_time = None

    return t_values, Vm_values, m_values, h_values, n_values, IK, INa, IL, stable_time, EL


t_values, Vm_values, _, _, _, _, _, _, stable_time, EL = HH_EL(delta_time, total_time, stim)
print(f'EL = {EL}')
if stable_time is not None:
    print(f'Stable return time: {stable_time} ms')

plt.figure(figsize=(10, 6))
plt.plot(t_values, Vm_values, label='Membrane Voltage (Vm)')
if stable_time is not None:
    plt.axvline(x=stable_time, color='m', linestyle='--', label='Stability Return Time')
    plt.text(stable_time, Vm_values[np.argmin(np.abs(t_values - stable_time))], f'{stable_time:.2f} μs', color='m')
plt.title('Membrane Potential Over Time for gL = 0.01')
plt.xlabel('Time (μs)')
plt.ylabel('Membrane Voltage (mV)')
plt.xlim([0, 250])  # Adjust as needed
plt.grid()
plt.legend()
plt.show()

'Q NO.27 AP from 2nd stimulus'
def HH_with_stimulus(delta_time, total_time, stim_amplitude, stim_start_time, stim_duration, interval):
    t_values = np.arange(0, total_time, delta_time)
    Vm_values = np.zeros(len(t_values))
    m_values = np.zeros(len(t_values))
    h_values = np.zeros(len(t_values))
    n_values = np.zeros(len(t_values))
    IK = np.zeros(len(t_values))
    INa = np.zeros(len(t_values))
    INaflag = np.zeros(len(t_values), dtype=int)

    Vm_values[0] = Vm_init
    m_values[0] = m_init
    h_values[0] = h_init
    n_values[0] = n_init

    for t in range(len(t_values) - 1):
        # Apply first stimulus
        if stim_start_time <= t * delta_time < stim_start_time + stim_duration:
            I_stim = stim_amplitude
        # Apply second stimulus at the interval after first start time
        elif stim_start_time + interval <= t * delta_time < stim_start_time + interval + stim_duration:
            I_stim = stim_amplitude
        else:
            I_stim = 0

        # Update gating variables
        m = m_values[t] + delta_time * (alpha_m(Vm_values[t]) * (1 - m_values[t]) - beta_m(Vm_values[t]) * m_values[t])
        h = h_values[t] + delta_time * (alpha_h(Vm_values[t]) * (1 - h_values[t]) - beta_h(Vm_values[t]) * h_values[t])
        n = n_values[t] + delta_time * (alpha_n(Vm_values[t]) * (1 - n_values[t]) - beta_n(Vm_values[t]) * n_values[t])

        # Calculate currents
        IK[t] = gK * (n ** 4) * (Vm_values[t] - EK)
        INa[t] = gNa * (m ** 3) * h * (Vm_values[t] - ENa)
        Iion = IK[t] + INa[t] + gL * (Vm_values[t] - EL)

        # Update membrane potential
        dVm = delta_time / Cm * (I_stim - Iion)
        Vm_values[t + 1] = Vm_values[t] + dVm
        m_values[t + 1], h_values[t + 1], n_values[t + 1] = m, h, n

        # Set INaflag when -INa exceeds IK
        if -INa[t] > IK[t]:
            INaflag[t] = 1
        else:
            INaflag[t] = 0

    # Check if a second action potential occurred
    AP_detected = np.any(np.diff(INaflag[stim_start_time + interval:]) == 1)
    return AP_detected


def find_min_interval(delta_time, total_time, stim_amplitude):
    for interval in range(50, int(total_time), 50):  # Check intervals in multiples of 50 μs
        if HH_with_stimulus(delta_time, total_time, stim_amplitude, start_time, stim_durations, interval):
            return interval  # Return the interval when second action potential is detected
    return None  # Return None if no interval produces a second AP


interval_50uA = find_min_interval(delta_time, total_time, 50)
interval_200uA = find_min_interval(delta_time, total_time, 200)
interval_500uA = find_min_interval(delta_time, total_time, 500)

print(f"The earliest interval for 50 μA/cm² to produce a second AP is {interval_50uA} μs.")
print(f"The earliest interval for 200 μA/cm² to produce a second AP is {interval_200uA} μs.")
print(f"The earliest interval for 500 μA/cm² to produce a second AP is {interval_500uA} μs.")

'Evaluation of m,n, h gates, Vm, K and Na behavior over time'
stim_amplitude = 200.0
t_values, Vm_values, m_values, h_values, n_values, IK_values, INa_values, IL_values, _ = HH(delta_time, total_time, stim_amplitude)

# Plotting
time_axis = np.arange(0, total_time, delta_time)
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle('Behavior of Vm, IK, INa during an action potential', fontsize=16)

ax1 = plt.subplot(212)
ax1.plot(time_axis, Vm_values, label='Vm')
plt.xlabel('Time (μs)')
plt.ylabel('Membrane potential (mV)')
plt.title('Membrane Potential Vm')
plt.xlim(0, 300)
plt.grid()

ax2 = plt.subplot(221)
ax2.plot(time_axis, IK_values, label='IK')
ax2.margins(2, 1)
plt.xlabel('Time (μs)')
plt.ylabel('Current density (μA/cm^2)')
plt.title('K current densities')
plt.xlim(0, 300)
plt.legend()
plt.grid()

ax3 = plt.subplot(222)
ax3.plot(time_axis, INa_values, label='INa')
plt.title('Na current densities')
plt.xlim(0, 300)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 9))
fig.suptitle('Behavior of m, n, h gates during an action potential', fontsize=16)

axs[0].plot(time_axis, n_values, label='n(t)')
axs[0].set_xlabel('Time (μs)')
axs[0].set_ylabel('Gate value')
axs[0].set_title('K+ Gate Dynamics for n(t)')
axs[0].set_xlim(50, 300)
axs[0].grid()
axs[0].legend()


axs[1].plot(time_axis, m_values, label='m(t)')
axs[1].set_xlabel('Time (μs)')
axs[1].set_ylabel('Gate values')
axs[1].set_title('Na+ Gate Dynamics for m(t)')
axs[1].set_xlim(50, 300)
axs[1].legend()
axs[1].grid()

axs[2].plot(time_axis, h_values, label='h(t)')
axs[2].set_xlabel('Time (μs)')
axs[2].set_ylabel('Gate values')
axs[2].set_title('Na+ Gate Dynamics for h(t)')
axs[2].set_xlim(50, 300)
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()

'Time interval T required for the cell to take before going through an action potential just after getting out of one'
# T is the time it takes to become stable again, which is what we obtained above

'Dependency of threshold amplitude for action potential to waveform'
Vm_values = []


def square_wave(amplitude, t, start_time, stim_end_time):
    return amplitude if start_time <= t <= stim_end_time else 0


def sawtooth_wave(amplitude, t, start_time, stim_end_time):
    if start_time <= t <= stim_end_time:
        return amplitude * (t - start_time) / (stim_end_time - start_time)
    return 0


def sinusoidal_wave(amplitude, t, start_time, stim_end_time):
    if start_time <= t <= stim_end_time:
        return amplitude * np.sin(np.pi * (t - start_time) / (stim_end_time - start_time))
    return 0


# Updated HH function to take a waveform function as input
def HH_waveform(delta_time, total_time, waveform_func):
    t_values = np.arange(0, total_time, delta_time)
    Vm_values = np.zeros(len(t_values))
    m_values = np.zeros(len(t_values))
    h_values = np.zeros(len(t_values))
    n_values = np.zeros(len(t_values))
    IK = np.zeros(len(t_values))
    INa = np.zeros(len(t_values))
    IL = np.zeros(len(t_values))

    Vm_values[0] = Vm_init
    m_values[0] = m_init
    h_values[0] = h_init
    n_values[0] = n_init

    for t in range(len(t_values) - 1):
        # Use the waveform function to calculate I_stim at each time step
        I_stim = waveform_func(t * delta_time)

        m = m_values[t] + delta_time * (
                alpha_m(Vm_values[t]) * (1 - m_values[t]) - beta_m(Vm_values[t]) * m_values[t])
        h = h_values[t] + delta_time * (
                alpha_h(Vm_values[t]) * (1 - h_values[t]) - beta_h(Vm_values[t]) * h_values[t])
        n = n_values[t] + delta_time * (
                alpha_n(Vm_values[t]) * (1 - n_values[t]) - beta_n(Vm_values[t]) * n_values[t])

        IK[t] = gK * (n_values[t] ** 4) * (Vm_values[t] - EK)
        INa[t] = gNa * (m_values[t] ** 3) * h_values[t] * (Vm_values[t] - ENa)
        IL[t] = gL * (Vm_values[t] - EL)
        Iion = IK[t] + INa[t] + IL[t]
        dVm = delta_time / Cm * (I_stim - Iion)
        Vm_values[t + 1] = Vm_values[t] + dVm
        m_values[t + 1] = m
        h_values[t + 1] = h
        n_values[t + 1] = n

    return t_values, Vm_values, m_values, h_values, n_values, IK, INa, IL


# Function to find the minimum amplitude for an action potential
def find_min_amplitude(waveform_func, delta_time, total_time, start_time, stim_end_time):
    min_amplitude = None
    for amplitude in np.arange(0, 100, 2.5):  # Gradually increase amplitude to find minimum
        # Create a lambda function that captures amplitude for the waveform function
        I_stim_func = lambda t: waveform_func(amplitude, t, start_time, stim_end_time)
        t_values, Vm_values, _, _, _, _, _, _ = HH_waveform(delta_time, total_time, I_stim_func)

        if AP_check(Vm_values):  # Threshold for action potential
            min_amplitude = amplitude
            break
    return min_amplitude


min_amp_square = find_min_amplitude(square_wave, delta_time, total_time, start_time, stim_end_time)
min_amp_sawtooth = find_min_amplitude(sawtooth_wave, delta_time, total_time, start_time, stim_end_time)
min_amp_sinusoidal = find_min_amplitude(sinusoidal_wave, delta_time, total_time, start_time, stim_end_time)

print(f"Minimum amplitude for square wave: {min_amp_square}")
print(f"Minimum amplitude for sawtooth wave: {min_amp_sawtooth}")
print(f"Minimum amplitude for sinusoidal wave: {min_amp_sinusoidal}")
