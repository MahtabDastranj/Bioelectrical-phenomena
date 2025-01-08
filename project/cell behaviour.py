import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants and Parameters
cell_diameter = 10e-6  # Cell diameter in meters (10 μm)
cell_volume = (4 / 3) * np.pi * (cell_diameter / 2) ** 3  # Spherical volume
cell_side = (cell_volume) ** (1 / 3)  # Cube side length equivalent
membrane_surface_area = 6 * (cell_side ** 2)  # Cube surface area
channel_density = 1e6  # Ion channels per m² (assumed)
num_channels = membrane_surface_area * channel_density  # Total channels
E_field = 500  # Electric field in V/m
distance_electrodes = 0.01  # Distance between electrodes in meters
sigma_tissue = 0.2  # Tissue conductivity in S/m (assumed)
V_rest = -70e-3  # Resting potential in volts (-70 mV)
G_channel = 1e-9  # Channel conductance in Siemens (1 nS)
f_frequency = 5  # Frequency in Hz
time = np.linspace(0, 1, 1000)  # 1-second simulation

# Compute induced membrane voltage
V_membrane = E_field * cell_side  # Simplified induced voltage

# Membrane current through channels
I_channel = G_channel * (V_membrane - V_rest)  # Current per channel
I_total = num_channels * I_channel  # Total current

# Compute dipole moment and electrode voltage
dipole_moment = I_total * cell_side
distance_to_electrode = distance_electrodes
V_electrode = dipole_moment / (4 * np.pi * sigma_tissue * distance_to_electrode ** 3)

# Generate harmonics from simulated voltage
V_signal = V_membrane * np.sin(2 * np.pi * f_frequency * time)
second_harmonic = np.sin(4 * np.pi * f_frequency * time) * V_membrane / 2
third_harmonic = np.sin(6 * np.pi * f_frequency * time) * V_membrane / 3

# Fourier analysis
fft_signal = np.fft.fft(V_signal)
frequencies = np.fft.fftfreq(len(fft_signal), d=(time[1] - time[0]))

# Visualization
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, V_signal, label="Primary Signal")
plt.title("Primary Signal (Membrane Voltage)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, second_harmonic, label="Second Harmonic")
plt.title("Second Harmonic")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, third_harmonic, label="Third Harmonic")
plt.title("Third Harmonic")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

plt.tight_layout()
plt.show()

# Fourier Transform Plot
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_signal[:len(fft_signal)//2]))
plt.title("Fourier Transform of the Membrane Voltage")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

# Outputs
results = {
    "Membrane Surface Area (m²)": membrane_surface_area,
    "Number of Channels": num_channels,
    "Induced Membrane Voltage (V)": V_membrane,
    "Total Membrane Current (A)": I_total,
    "Voltage at Electrode (V)": V_electrode,
}

results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
import ace_tools as tools; tools.display_dataframe_to_user(name="Membrane Voltage and Harmonics Analysis Results", dataframe=results_df)
