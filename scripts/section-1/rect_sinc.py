import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 2.0  # Width of the rectangular pulse in seconds
A = 1.0  # Amplitude of the rectangular pulse

# Time domain grid
t = np.linspace(-3.0, 3.0, 1000)
rect = np.where(np.abs(t) < T / 2, A, 0.0)
rect[np.isclose(np.abs(t), T / 2)] = A / 2.0

# Frequency domain grid (f in Hz)
f = np.linspace(-4.0, 4.0, 1000)
X_f = A * T * np.sinc(f * T)

fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(9, 7))
fig.tight_layout(pad=4.0)

ax_time.plot(
    t,
    rect,
    color="#1f77b4",
    linewidth=2.5,
    label=r"$x(t) = \mathrm{rect}(t/T)$",
)
ax_time.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax_time.axvline(-T / 2, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
ax_time.axvline(T / 2, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)

ax_time.set_title("Time Domain: Rectangular Pulse", fontsize=14, pad=10)
ax_time.set_xlabel("Time $t$ (seconds)", fontsize=11)
ax_time.set_ylabel("Amplitude", fontsize=11)
ax_time.set_ylim(-0.2, 1.3)
ax_time.set_xticks([-T / 2, 0, T / 2])
ax_time.set_xticklabels([r"$-T/2$", "$0$", r"$T/2$"])
ax_time.grid(True, linestyle=":", alpha=0.6)
ax_time.legend(loc="upper right", fontsize=11)

ax_freq.plot(
    f,
    X_f,
    color="#d62728",
    linewidth=2.5,
    label=r"$X(f) = T \cdot \mathrm{si}(\pi fT)$",
)
ax_freq.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

ax_freq.axvline(-1 / T, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
ax_freq.axvline(1 / T, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)

ax_freq.set_title("Frequency Domain: Sinc Function", fontsize=14, pad=10)
ax_freq.set_xlabel("Frequency $f$ (Hz)", fontsize=11)
ax_freq.set_ylabel("Spectral Density", fontsize=11)
ax_freq.set_ylim(-0.6 * T, 1.2 * T)
ax_freq.set_xticks([-2 / T, -1 / T, 0, 1 / T, 2 / T])
ax_freq.set_xticklabels([r"$-2/T$", r"$-1/T$", "$0$", r"$1/T$", r"$2/T$"])
ax_freq.grid(True, linestyle=":", alpha=0.6)
ax_freq.legend(loc="upper right", fontsize=11)
fig.savefig("rect_sinc.pdf")

plt.show()
