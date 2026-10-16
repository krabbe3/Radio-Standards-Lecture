import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

M = 8  # Number of elements
init_d = 0.5  # Initial element spacing (units of lambda)
N_samples = 200  # Time snapshots

np.random.seed(42)
s = (np.random.randn(N_samples) + 1j * np.random.randn(N_samples)) / np.sqrt(2)
noise_pwr = 10 ** (-15 / 10)  # 15 dB SNR
noise = np.sqrt(noise_pwr / 2) * (
    np.random.randn(M, N_samples) + 1j * np.random.randn(M, N_samples)
)

# Scanning angles
theta_scan = np.linspace(-90, 90, 721)


# Array response
def steering_vector(theta_deg, d_spacing):
    psi = 2 * np.pi * d_spacing * np.sin(np.deg2rad(theta_deg))
    return np.exp(-1j * np.arange(M) * psi)


fig, (ax_geo, ax_spec) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.32, wspace=0.28)

init_theta_sig = 25.0
init_theta_steer = 0.0

ax_geo.set_xlim(-4.0, 4.0)
ax_geo.set_ylim(-1.0, 3.5)
ax_geo.set_aspect("equal")
ax_geo.set_xlabel(r"Array Axis $x / \lambda$")
ax_geo.set_ylabel(r"Propagation Axis $y / \lambda$")
ax_geo.set_title("ULA Geometry & Incident Wavefronts")
ax_geo.grid(True, linestyle="--", alpha=0.5)

ax_geo.plot([-3.8, 3.8], [0, 0], color="gray", lw=2, alpha=0.6)
scatter_ant = ax_geo.scatter(
    [], [], color="blue", s=55, zorder=5, label="Antennas"
)
ant_text_labels = [
    ax_geo.text(0, 0, "", ha="center", va="top", fontsize=7, color="navy")
    for _ in range(M)
]

(line_sig_ray,) = ax_geo.plot([], [], "r-", lw=2, label="Signal Ray (AoA)")
(line_steer_ray,) = ax_geo.plot(
    [], [], "g--", lw=1.8, label="Steered Direction"
)
wavefront_lines = [
    ax_geo.plot([], [], color="cyan", lw=1.5, alpha=0.75)[0] for _ in range(5)
]

ax_geo.legend(loc="upper left", fontsize=8)

(line_spec,) = ax_spec.plot([], [], "b-", lw=1.8, label="Bartlett Spectrum")
line_sig_spec = ax_spec.axvline(
    init_theta_sig, color="r", linestyle="--", lw=1.5, label="True AoA"
)
line_steer_spec = ax_spec.axvline(
    init_theta_steer,
    color="g",
    linestyle=":",
    lw=1.8,
    label="Steered Weight Angle",
)
point_out = ax_spec.plot([], [], "go", markersize=7)[0]

ax_spec.set_xlim(-90, 90)
ax_spec.set_ylim(-30, 2)
ax_spec.set_xlabel(r"Angle $\theta$ (deg)")
ax_spec.set_ylabel("Power (dB, normalized)")
ax_spec.set_title("Spatial Bartlett Spectrum")
ax_spec.grid(True, linestyle="--", alpha=0.6)
ax_spec.legend(loc="upper right", fontsize=8)

# --- Sliders ---
ax_slider_sig = plt.axes([0.15, 0.18, 0.7, 0.025])
ax_slider_steer = plt.axes([0.15, 0.11, 0.7, 0.025])
ax_slider_d = plt.axes([0.15, 0.04, 0.7, 0.025])

slider_sig = Slider(
    ax_slider_sig,
    "Signal AoA",
    -75.0,
    75.0,
    valinit=init_theta_sig,
    valstep=0.5,
)
slider_steer = Slider(
    ax_slider_steer,
    "Weight Steer",
    -75.0,
    75.0,
    valinit=init_theta_steer,
    valstep=0.5,
)
slider_d = Slider(
    ax_slider_d,
    r"Spacing $d / \lambda$",
    0.1,
    1.5,
    valinit=init_d,
    valstep=0.05,
)


def update(val=None):
    th_s = slider_sig.val
    th_w = slider_steer.val
    d_val = slider_d.val

    th_s_rad = np.deg2rad(th_s)
    th_w_rad = np.deg2rad(th_w)

    # --- Update Array Positions ---
    array_x = (np.arange(M) - (M - 1) / 2) * d_val
    array_y = np.zeros(M)
    scatter_ant.set_offsets(np.column_stack((array_x, array_y)))

    for i, (x_pos, text_obj) in enumerate(zip(array_x, ant_text_labels)):
        text_obj.set_position((x_pos, -0.25))
        text_obj.set_text(f"{i}")

    k_sig = np.array([-np.sin(th_s_rad), np.cos(th_s_rad)])
    ray_len = 3.0
    line_sig_ray.set_data([0, k_sig[0] * ray_len], [0, k_sig[1] * ray_len])

    k_steer = np.array([-np.sin(th_w_rad), np.cos(th_w_rad)])
    line_steer_ray.set_data(
        [0, k_steer[0] * ray_len], [0, k_steer[1] * ray_len]
    )

    u_tangent = np.array([np.cos(th_s_rad), np.sin(th_s_rad)])
    w_width = 3.5
    for idx, d_offset in enumerate(
        np.linspace(0.4, 2.4, len(wavefront_lines))
    ):
        c = k_sig * d_offset
        p1 = c - 0.5 * w_width * u_tangent
        p2 = c + 0.5 * w_width * u_tangent
        wavefront_lines[idx].set_data([p1[0], p2[0]], [p1[1], p2[1]])

    A_scan = np.array([steering_vector(th, d_val) for th in theta_scan]).T
    a_sig = steering_vector(th_s, d_val).reshape(M, 1)

    X = a_sig @ s.reshape(1, N_samples) + noise
    Rxx = (X @ X.conj().T) / N_samples

    # bartlett spectrum
    P_bf = np.real(np.sum(A_scan.conj() * (Rxx @ A_scan), axis=0)) / (M**2)
    P_bf_dB = 10 * np.log10(P_bf / np.max(P_bf))

    line_spec.set_data(theta_scan, P_bf_dB)
    line_sig_spec.set_xdata([th_s, th_s])
    line_steer_spec.set_xdata([th_w, th_w])

    w = steering_vector(th_w, d_val) / M
    P_curr = np.real(w.conj().T @ Rxx @ w)
    P_curr_dB = 10 * np.log10(P_curr / np.max(P_bf))
    point_out.set_data([th_w], [P_curr_dB])

    fig.canvas.draw_idle()


slider_sig.on_changed(update)
slider_steer.on_changed(update)
slider_d.on_changed(update)

update()
plt.show()
