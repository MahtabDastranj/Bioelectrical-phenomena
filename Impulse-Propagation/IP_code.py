import numpy as np
import matplotlib.pyplot as plt

'Core-Conductor model'
x = np.linspace(-4, 4, 500)  # mm
Vm = 50 * np.tanh(x)  # Transmembrane potential in mV
dVm_dx = 50 * (1 - np.tanh(x)**2)  # Derivative of Vm
r_i = 3  # Intracellular resistance per unit length (Ω/mm)
r_e = 1  # extracellular resistance per unit length (Ω/mm)
Ii = -dVm_dx / r_i  # Intracellular current
dIi_dx = -100 * np.tanh(x) * (1 - np.tanh(x)**2)
im = - dIi_dx
Ie = -dVm_dx / r_e  # Extracellular current

# Normalize the waveform
Vm_norm = Vm / np.max(np.abs(Vm))

# Plot the original and normalized waveforms
plt.figure(figsize=(10, 6))

plt.plot(x, Vm, label=r"V_m(x) (Original)", color='blue')
plt.plot(x, Vm_norm, label=r"V_m (Normalized)", linestyle='--', color='red')
plt.plot(x, Ii, label=r"I_i(x) (Intracellular Current)", color='green')
plt.plot(x, Ie, label=r"I_e(x) (Extracellular Current)", color='cyan')
plt.plot(x, im, label=r'im(x)', color='magenta')

plt.title("Transmembrane Potential V_m(x) and Normalized Wave Shape")
plt.xlabel("Position x (mm)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

'Vm(t) for x0 = 2mm, theta = 2 m/sec'
Ri = 1000  # Ω.mm
Re = 0
x0 = 2  # mm
x = 10  # mm
theta = 2000  # mm/sec
t = np.linspace(0, 4, 1000)
Vm = 50 * np.tanh(t - ((x - x0) / theta))

plt.figure(figsize=(10, 6))
plt.plot(t, Vm, label=r"V_m(x) (Original)", color='blue')
plt.title("Transmembrane Potential V_m(x) during a period of time")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
