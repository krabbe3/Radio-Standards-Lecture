import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

M = 8  # Number of antenna elements
d = 0.5  # Spacing in wavelengths (d = lambda / 2)
N_samples = 400  # Number of time snapshots
theta_scan = np.linspace(-90, 90, 1000)


def steering_vector(theta_deg):
    psi = 2 * np.pi * d * np.sin(np.deg2rad(theta_deg))
    return np.exp(-1j * np.arange(M) * psi)


# Array manifold for scanning
A_scan = np.array([steering_vector(th) for th in theta_scan]).T
# input signal
np.random.seed(42)
s1_base = (
    np.random.randn(N_samples) + 1j * np.random.randn(N_samples)
) / np.sqrt(2)
s2_base = (
    np.random.randn(N_samples) + 1j * np.random.randn(N_samples)
) / np.sqrt(2)
sigma_n = 0.05
noise = (
    sigma_n
    * (np.random.randn(M, N_samples) + 1j * np.random.randn(M, N_samples))
    / np.sqrt(2)
)

# Initial slider state
init_theta1 = 0.0  # Angle of Source 1 (deg)
init_delta_th = 6.0  # Angular separation Delta theta (deg)
init_pwr_ratio = 15.0  # Power difference P2 - P1 in dB (e.g. +15 dB jammer)

fig, ax = plt.subplots(figsize=(10, 5.5))
plt.subplots_adjust(bottom=0.32)
(line_bartlett,) = ax.plot(
    [], [], "r-", lw=2.0, label="Bartlett (Conventional)"
)
line_src1 = ax.axvline(
    init_theta1, color="g", linestyle=":", lw=1.8, label="Source 1 (0 dB)"
)
line_src2 = ax.axvline(
    init_theta1 + init_delta_th,
    color="m",
    linestyle=":",
    lw=1.8,
    label="Source 2",
)

ax.set_xlim(-60, 60)
ax.set_ylim(-40, 5)
ax.set_xlabel(r"Angle $\theta$ (deg)")
ax.set_ylabel("Normalized Power (dB)")
ax.set_title(r"Bartlett Beamformer Multiple Sources)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right", fontsize=8.5)

ax_th1 = plt.axes([0.15, 0.18, 0.7, 0.025])
ax_sep = plt.axes([0.15, 0.11, 0.7, 0.025])
ax_pwr = plt.axes([0.15, 0.04, 0.7, 0.025])

slider_th1 = Slider(
    ax_th1,
    r"Source 1 $\theta_1$ (deg)",
    -40.0,
    40.0,
    valinit=init_theta1,
    valstep=0.5,
)
slider_sep = Slider(
    ax_sep,
    r"Separation $\Delta\theta$ (deg)",
    0.5,
    30.0,
    valinit=init_delta_th,
    valstep=0.5,
)
slider_pwr = Slider(
    ax_pwr,
    r"Power $\Delta P = P_2 - P_1$ (dB)",
    -25.0,
    30.0,
    valinit=init_pwr_ratio,
    valstep=0.5,
)


def update(val=None):
    th1 = slider_th1.val
    delta_th = slider_sep.val
    th2 = th1 + delta_th
    dp_dB = slider_pwr.val

    # amplitudes
    a1_scale = 1.0
    a2_scale = np.sqrt(10 ** (dp_dB / 10))

    s1 = a1_scale * s1_base
    s2 = a2_scale * s2_base

    # received signal
    a1 = steering_vector(th1)[:, None]
    a2 = steering_vector(th2)[:, None]
    X = a1 @ s1[None, :] + a2 @ s2[None, :] + noise

    # covariance
    Rxx = (X @ X.conj().T) / N_samples

    # Bartlett spectrum
    P_bartlett = np.real(np.sum(A_scan.conj() * (Rxx @ A_scan), axis=0)) / (
        M**2
    )
    P_bartlett_dB = 10 * np.log10(P_bartlett / np.max(P_bartlett))

    line_bartlett.set_data(theta_scan, P_bartlett_dB)
    line_src1.set_xdata([th1, th1])
    line_src2.set_xdata([th2, th2])
    line_src2.set_label(f"Source 2 ({dp_dB:+.1f} dB)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.canvas.draw_idle()


slider_th1.on_changed(update)
slider_sep.on_changed(update)
slider_pwr.on_changed(update)
update()
plt.show()
