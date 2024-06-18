import numpy as np
import matplotlib.pyplot as plt
import mne

# Ruta al archivo .fif generado por el script de preprocesamiento
fif_file_path = r"C:\Users\EdgardOrdoñez\OneDrive - AltiaTek\Desktop\preprocessed\sub-002\sub-002-epo.fif"

# Cargar el archivo .fif
epochs = mne.read_epochs(fif_file_path)
num_channels = len(epochs.ch_names)
print("Número de canales:", num_channels)
# Filtrar datos
epochs.filter(l_freq=1, h_freq=50)

# Definir los valores para freqs y n_cycles
freqs = np.arange(1, 50, 1)  # Frecuencias de 1 a 50 Hz con paso de 1 Hz
n_cycles = freqs / 2.0  # Asignar un número de ciclos para cada frecuencia

# Calcular y graficar el espectrograma
power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)

# Configurar la gráfica
plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(power.data.mean(axis=0)), aspect='auto', origin='lower', extent=[power.times[0], power.times[-1], power.freqs[0], power.freqs[-1]], cmap='jet')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.title('Espectrograma de Potencia Promedio')
plt.colorbar(label='Potencia (dB)')

# Mostrar la gráfica
plt.show()
