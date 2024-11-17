import numpy as np

'Core-conductor model'
Vm_rest = -60.0  # mV
Cm = 1.0  # μF/cm^2
EK, ENa, EL = -72.1, 52.4, -49.187
gK, gNa, gL = 36.0, 120.0, 0.3
radius = 30e-4  # cm
rho_e = 50      # Ωcm
rho_i = 3 * rho_e  # Ωcm


def alpha_n(Vm): return 0.01 * (Vm + 50) / (1 - np.exp(-(Vm + 50) / 10))


def beta_n(Vm): return 0.125 * np.exp(-(Vm + 60) / 80)


def alpha_m(Vm): return 0.1 * (Vm + 35) / (1 - np.exp(-(Vm + 35) / 10))


def beta_m(Vm): return 4 * np.exp(-(Vm + 60) / 18)


def alpha_h(Vm): return 0.07 * np.exp(-(Vm + 60) / 20)


def beta_h(Vm): return 1 / (1 + np.exp(-(Vm + 30) / 10))


# Steady-state values of gating variables
n_rest = alpha_n(Vm_rest) / (alpha_n(Vm_rest) + beta_n(Vm_rest))
m_rest = alpha_m(Vm_rest) / (alpha_m(Vm_rest) + beta_m(Vm_rest))
h_rest = alpha_h(Vm_rest) / (alpha_h(Vm_rest) + beta_h(Vm_rest))

# Conductance at rest
gK_rest = gK * n_rest**4
gNa_rest = gNa * m_rest**3 * h_rest

# Total membrane conductance
g_m = gK_rest + gNa_rest + gL
R_m = 1 / g_m  # Ω·cm^2

# Calculations
r_m = R_m / (2 * np.pi * radius)  # Membrane resistance per unit length
r_i = rho_i / (np.pi * radius**2)  # Intracellular resistance per unit length
r_e = rho_e / (np.pi * (2 * radius)**2)  # Extracellular resistance per unit length

# Output results
print(f"Membrane Resistivity (R_m): {R_m:.2e} Ω·cm^2")
print(f"Membrane Resistance (r_m): {r_m:.2e} Ω/cm")
print(f"Intracellular Resistance per unit length (r_i): {r_i:.2e} Ω/cm")
print(f"Extracellular Resistance per unit length (r_e): {r_e:.2e} Ω/cm")
