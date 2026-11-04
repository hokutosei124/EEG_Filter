import numpy as np
import pandas as pd
from scipy.signal import bessel, butter, cheby1, firwin, filtfilt, welch
import matplotlib.pyplot as plt

# Function to generate PSD using Welch's method
def compute_normalized_psd(signal):
    freqs, psd = welch(signal, fs=fs, nperseg=int(2 * fs), return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    max_val = np.max(psd)
    psd_norm = (psd / max_val) if max_val > 0 else psd
    return freqs, psd_norm

# Parameters
band_name = "Alpha"
fmin, fmax = 1, 5
fs = 200    # Hz
filter_type = "butter"
filter_order = 3
T1 = 20.12
T2 = 40.37

idx_1 = int(T1 * fs)
idx_2 = int(T2 * fs)


# Self-implemented filter section

# Load data
df = pd.read_csv("BIOPAC.csv")
raw_signal = df["EEG"].values
ref_signal = df[band_name].values

# Partition data into eyes closed & open parts
raw_closed = np.concatenate([raw_signal[:idx_1], raw_signal[idx_2:]])
raw_opened = raw_signal[idx_1:idx_2]
ref_closed = np.concatenate([ref_signal[:idx_1], ref_signal[idx_2:]])
ref_opened = ref_signal[idx_1:idx_2]

# Apply self-implemented filter to raw partitioned data
if filter_type == "butter":
    b, a = butter(N=filter_order, Wn=[fmin, fmax], btype="bandpass", fs=fs)
    self_closed = filtfilt(b, a, raw_closed)
    self_opened = filtfilt(b, a, raw_opened)
elif filter_type == "bessel":
    b, a = bessel(N=filter_order, Wn=[fmin, fmax], btype="bandpass", fs=fs)
    self_closed = filtfilt(b, a, raw_closed)
    self_opened = filtfilt(b, a, raw_opened)
elif filter_type == "cheby":
    b, a = cheby1(N=filter_order, rp=1.0, Wn=[fmin, fmax], btype="bandpass")
    self_closed = filtfilt(b, a, raw_closed)
    self_opened = filtfilt(b, a, raw_opened)
elif filter_type == "fir":
    b = firwin(numtaps=301, cutoff=[fmin, fmax], pass_zero=False, fs=fs)
    self_closed = filtfilt(b, [1.0], raw_closed)
    self_opened = filtfilt(b, [1.0], raw_opened)




# Compute spectra for all 4 conditions
f_sc, p_sc = compute_normalized_psd(self_closed)  # Self-Implemented closed
f_rc, p_rc = compute_normalized_psd(ref_closed)   # Reference closed
f_so, p_so = compute_normalized_psd(self_opened)  # Self-Implemented opened
f_ro, p_ro = compute_normalized_psd(ref_opened)   # Reference opened

# Plotting
# Set up 2x2 Subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

# Top-Left: Self-Implemented Eyes-Closed
axes[0, 0].plot(f_sc, p_sc, color="blue", lw=1.5)
axes[0, 0].fill_between(f_sc, p_sc, color="blue", alpha=0.2)
axes[0, 0].set_title(f"Self-Implemented: {band_name} (Eyes Closed)")

# Top-Right: Reference (BIOPAC) Eyes-Closed
axes[0, 1].plot(f_rc, p_rc, color="blue", lw=1.5)
axes[0, 1].fill_between(f_rc, p_rc, color="blue", alpha=0.2)
axes[0, 1].set_title(f"Reference: {band_name} (Eyes Closed)")

# Bottom-Left: Self-Implemented Eyes-Opened
axes[1, 0].plot(f_so, p_so, color="blue", lw=1.5)
axes[1, 0].fill_between(f_so, p_so, color="blue", alpha=0.2)
axes[1, 0].set_title(f"Self-Implemented: {band_name} (Eyes Opened)")

# Bottom-Right: Reference (BIOPAC) Eyes-Opened
axes[1, 1].plot(f_ro, p_ro, color="blue", lw=1.5)
axes[1, 1].fill_between(f_ro, p_ro, color="blue", alpha=0.2)
axes[1, 1].set_title(f"Reference: {band_name} (Eyes Opened)")

# Formatting all subplots uniformly
for ax in axes.flatten():
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.grid(True, ls="--", alpha=0.5)

plt.tight_layout()
plt.show()
