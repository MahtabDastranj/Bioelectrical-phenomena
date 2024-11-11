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


'Q NO.27 AP from 2nd stimulus'


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


# Find minimum amplitudes for each waveform
min_amp_square = find_min_amplitude(square_wave, delta_time, total_time, start_time, stim_end_time)
min_amp_sawtooth = find_min_amplitude(sawtooth_wave, delta_time, total_time, start_time, stim_end_time)
min_amp_sinusoidal = find_min_amplitude(sinusoidal_wave, delta_time, total_time, start_time, stim_end_time)

print(f"Minimum amplitude for square wave: {min_amp_square}")
print(f"Minimum amplitude for sawtooth wave: {min_amp_sawtooth}")
print(f"Minimum amplitude for sinusoidal wave: {min_amp_sinusoidal}")

'''

# Finding I-T curve
time_intervals = np.arange(1e-3, 10e-3, 100e-6)  # Time intervals in seconds (1 ms to 10 ms)
stim_amplitude_start = 300  # Starting amplitude
I_values = []

for T in time_intervals:
    success = False
    current_I = stim_amplitude_start

    # Adjust I until it creates a second AP at interval T
    while not success:
        Vm_first_ap = simulate_ap(stim_amplitude_start, start_time=0)
        Vm_second_ap = simulate_ap(current_I, start_time=T)

        # Check if second stimulus generated an action potential
        if max(Vm_second_ap) > vm_threshold:
            I_values.append(current_I)
            success = True
        else:
            current_I += 10  # Increase the amplitude until AP is observed

# Plotting the I-T curve
plt.plot(time_intervals * 1e3, I_values, '-o')  # Convert time to ms for readability
plt.xlabel("Interval T (ms)")
plt.ylabel("Stimulus Amplitude I (μA/cm²)")
plt.title("I-T Curve Showing Relative Refractory Period")
plt.grid(True)
plt.show()'''
'''
'Q NO.25: Time to return to initial conditions'
# Stability envelope
Vm_tolerance = 0.1 * V  # mV
gate_tolerance = 0.01  # for n, m, h
Vm = Vm_init
n, m, h = n_init, m_init, h_init
Vm_values, n_values, m_values, h_values = [], [], [], []

stable_time = None
for t in range(time_steps):
    current_time = t * delta_time
    I_stim = stim_amplitude = 200 if current_time <= stim_durations else 0
    dVm, dn, dm, dh, IK, INa = HH(Vm, n, m, h, I_stim)

    Vm += dVm * delta_time
    n += dn * delta_time
    m += dm * delta_time
    h += dh * delta_time

    Vm_values.append(Vm)
    n_values.append(n)
    m_values.append(m)
    h_values.append(h)

    # Check stability after the stimulus has ended
    if current_time > stim_durations:
        if (abs(Vm - Vm_init) <= Vm_tolerance and
                abs(n - n_init) <= gate_tolerance and
                abs(m - m_init) <= gate_tolerance and
                abs(h - h_init) <= gate_tolerance):

            if stable_time is None:  # Only set stable time the first time conditions are met
                stable_time = current_time

# Output the results
if stable_time:
    print(f"\nTime to return to stable initial conditions: {stable_time * 1e6:.2f} μsec\n")
else:
    print("Membrane did not return to stable initial conditions within the simulation period.\n")

'Q NO.26: Leakage gL'


'Q NO.27 AP from 2nd stimulus'
for stim_amplitude in stim_amplitudes:
    min_interval = None  # To store the earliest interval that produces a second action potential

    for interval in range(int(stim_durations / delta_time), time_steps):
        Vm = Vm_init
        n, m, h = n_init, m_init, h_init
        INaflag = 0  # Initial flag for INa dominance
        AP_count = 0  # Counter for action potentials
        second_stimulus_ap = False

        # Main simulation loop
        for t in range(time_steps):
            current_time = t * delta_time

            # Determine stimulus based on time and interval
            if current_time <= stim_durations:
                I_stim = stim_amplitude
            elif stim_durations < current_time <= stim_durations + interval * delta_time:
                I_stim = 0
            elif stim_durations + interval * delta_time < current_time <= 2 * stim_durations + interval * delta_time:
                I_stim = stim_amplitude
            else:
                I_stim = 0

            # Compute HH model derivatives
            dVm, dn, dm, dh, IK, INa = HH(Vm, n, m, h, I_stim)

            # Update variables
            Vm += dVm * delta_time
            n += dn * delta_time
            m += dm * delta_time
            h += dh * delta_time

            # Update INaflag: set to 1 if -INa exceeds IK, otherwise 0
            if -INa > IK:
                if INaflag == 0:
                    INaflag = 1
                    AP_count += 1  # Count action potential
                    if AP_count == 2:
                        second_stimulus_ap = True
                        min_interval = interval * delta_time * 1e6  # Convert to μsec
                        break
            else:
                INaflag = 0

        if second_stimulus_ap:
            break  # Stop searching intervals once the earliest AP is found

    # Store results for this stimulus amplitude
    results[stim_amplitude] = min_interval if min_interval is not None else "No AP from second stimulus"

# Output results
for stim_amplitude, min_interval in results.items():
    if min_interval != "No AP from second stimulus":
        print(f"Earliest interval for stimulus amplitude {stim_amplitude} μA/cm²: {min_interval:.2f} μsec")
    else:
        print(f"Stimulus amplitude {stim_amplitude} μA/cm² did not produce a second action potential.")
'''