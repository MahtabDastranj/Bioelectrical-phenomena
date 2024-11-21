import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

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
Ri, Re = 1000, 0  # Ω.mm
radius = 0.05  # mm


def Vm(t, x, x0=2, theta=2000):
    return 50 * np.tanh(t - ((x - x0) / theta))


t = np.linspace(0, 4, 1000)
Vm1 = Vm(t=t, x=10, x0=2, theta=2000)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, Vm1, label=r"V_m(x)", color='blue')
plt.title("Transmembrane Potential V_m(x) during a period of time")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

'Vm(t) for x0 = 2mm, theta = 2 m/sec'
x = np.linspace(-20, 20, 1000)
Vm2 = Vm(t=3, x=x, x0=2, theta=2000)
plt.subplot(2, 1, 2)
plt.plot(x, Vm2, label=r"V_m(x)", color='blue')
plt.title("Transmembrane Potential V_m(x) along the fiber")
plt.xlabel("x(mm)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Ii, Ie
def r(radius, R):
    A = np.pi * radius ** 2
    return R / A


r_i = r(radius, Ri)
r_e = r(radius, Re)
dVm_dx = np.gradient(Vm2, x)
# Ii = -dVm_dx / r_i, Ie = -dVm_dx / r_e
Ii = - dVm_dx / r_i
Ie = - dVm_dx / r_e
plt.plot(x, Ii, label=r"Ii", color='blue')
plt.plot(x, Ie, label=r"Ie", color='green')
plt.title("Intracellular and Extracellular Currents")
plt.xlabel("x(mm)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

'EXTRACELLULAR FIELDS'
t1, t2 = 2, 5  # ms
x0 = 0  # mm
theta = 4  # mm/ms
s1, s2 = 2, 0.5  # ms^-1
a, b = 50, -60  # mV
radius = 50e-4  # cm (50 μm)
Ri = 1500  # Ω*cm
Re = 400  # Ω*cm
sigma_e = 1 / Re  # Extracellular conductivity
h = 0.01  # cm (100 μm)
r_i = r(radius, Ri)


def Vm(x, t):
    u1 = s1 * ((t - t1) - np.abs(x - x0) / theta)
    u2 = s2 * ((t - t2) - np.abs(x - x0) / theta)
    return b + a * (np.tanh(u1) - np.tanh(u2))


x = np.linspace(-50, 50, 1000)  # mm
Vm_values = Vm(x, t=3)  # Compute Vm at t = 3 ms
dVm_dx = np.gradient(Vm_values, x)  # Spatial derivative

Ii = -dVm_dx / r_i
dIi_dx = np.gradient(Ii, x)


# Extracellular potential
def phi_e(x_eval):
    def integrand(xi):
        return dIi_dx[np.searchsorted(x, xi)] / np.sqrt((x_eval - xi) ** 2 + h ** 2)

    integral, _ = quad(integrand, x.min(), x.max())
    return integral / (4 * np.pi * sigma_e)


# Plotting
plt.figure(figsize=(10, 6))

Phi_e1 = np.array([phi_e(xi) for xi in x])
plt.plot(x, Phi_e1, label=r"Phi_e(x) for h = 0.01", color='red')

h = 0.1  # cm (1000 μm)
Phi_e2 = np.array([phi_e(xi) for xi in x])
plt.plot(x, Phi_e2, label=r"Phi_e(x) for h = 0.1", color='blue')

plt.title("Extracellular Potential Phi_e(x) Along the Fiber")
plt.xlabel("x (mm)")
plt.ylabel(r"Phi_e(x) (mV)")
plt.grid(True)
plt.legend()
plt.show()

sigma_e = 1 / 400  # Extracellular conductivity (S/cm)
x0_upstroke, x0_downstroke, x0_rest = 0, 10, 25  # Locations of sources (mm)
I_upstroke, I_downstroke, I_rest = 50, -30, -10  # Currents (arbitrary units)


# Function for potential from a monopole source
def lumped_phi_e(x, x0, I, h):
    return I / (4 * np.pi * sigma_e * np.sqrt((x - x0)**2 + h**2))


x = np.linspace(0, 50, 1000)  # x from 0 to 50 mm

# Calculate Φe for each h
h_values = [0.01, 0.1, 1.0]  # in cm
phi_e_results = {}

for h in h_values:
    phi_e_upstroke = lumped_phi_e(x, x0_upstroke, I_upstroke, h)
    phi_e_downstroke = lumped_phi_e(x, x0_downstroke, I_downstroke, h)
    phi_e_rest = lumped_phi_e(x, x0_rest, I_rest, h)
    phi_e_results[h] = phi_e_upstroke + phi_e_downstroke + phi_e_rest

# Plot lumped potentials for different h
plt.figure(figsize=(10, 8))
colors = ['red', 'cyan', 'magenta']
for h, phi_e in phi_e_results.items():
    plt.plot(x, phi_e, label=f"h = {h * 10000:.0f} µm", color=colors[h_values.index(h)])

plt.title("Lumped Monopole Sources: Extracellular Potential Phi_e(x)")
plt.xlabel("x (mm)")
plt.ylabel(r"Phi_e(x) (mV)")
plt.legend()
plt.grid(True)
plt.show()


def dipole_phi_e(x, x0, p, h):
    return (p / (4 * np.pi * sigma_e)) * ((x - x0) / ((x - x0)**2 + h**2)**(3/2))


x0_dipole1 = 10  # Midpoint of dipole 1 (mm)
p_dipole1 = 20   # Dipole moment (arbitrary units)

# Dipole 2 parameters
x0_dipole2 = 30  # Midpoint of dipole 2 (mm)
p_dipole2 = -15  # Dipole moment (arbitrary units)

# Spatial range and h values
x = np.linspace(0, 50, 1000)  # From x = 0 to 50 mm
h_values = [0.01, 0.1]  # h = 100 µm (0.01 cm), h = 1000 µm (0.1 cm)
# Compute Φe for each h
phi_e_results = {}
for h in h_values:
    phi_e_dipole1 = dipole_phi_e(x, x0_dipole1, p_dipole1, h)
    phi_e_dipole2 = dipole_phi_e(x, x0_dipole2, p_dipole2, h)
    phi_e_results[h] = phi_e_dipole1 + phi_e_dipole2

# Plot Φe for each h
plt.figure(figsize=(12, 8))
for h, phi_e in phi_e_results.items():
    plt.plot(x, phi_e, label=f"h = {h * 10000:.0f} µm")

plt.title("Extracellular Potential Phi_e(x) from Lumped Dipoles")
plt.xlabel("x (mm)")
plt.ylabel(r"Phi_e(x) (mV)")
plt.legend()
plt.grid(True)
plt.show()
