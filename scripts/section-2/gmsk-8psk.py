import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

import matplotlib

matplotlib.use("TkAgg")

R_s = 1625e3 / 6  # GSM symbol rate, 270.833 ksym/s for GMSK and 8-PSK
sps = 16  # Samples per symbol
fs = R_s * sps
BT = 0.3  # Gaussian filter bandwidth-time product
N_sym = 2000  # Number of simulated symbols
N_show = 400  # Symbols shown in the IQ plane
init_snr = 15.0  # Initial Es/N0 (dB)
init_ibo = 3.0  # Initial amplifier input backoff (dB)
p_rapp = 2.0  # Smoothness of the amplifier model


def q_func(x):
    return 0.5 * np.vectorize(math.erfc)(x / np.sqrt(2))


# Gaussian frequency pulse of TS 45.004 (T = 1), integrates to 1/2
t_g = (np.arange(4 * sps) + 0.5) / sps
a = 2 * np.pi * BT / np.sqrt(np.log(2))
g = 0.5 * (q_func(a * (t_g - 2.5)) - q_func(a * (t_g - 1.5)))
G = np.cumsum(g) / sps

# Linearized GMSK pulse C0 of EDGE, spans 5 symbols
S = np.concatenate([np.sin(np.pi * G), np.sin(np.pi / 2 - np.pi * G)])
c0 = np.array(
    [
        S[n] * S[n + sps] * S[n + 2 * sps] * S[n + 3 * sps]
        for n in range(5 * sps)
    ]
)

np.random.seed(42)
bits = 2 * np.random.randint(0, 2, N_sym) - 1
psk_idx = np.random.randint(0, 8, N_sym)
N_rx = N_sym - 5  # Symbols whose pulse lies completely inside the signal
noise_base = (np.random.randn(N_rx) + 1j * np.random.randn(N_rx)) / np.sqrt(2)
offset = int(np.argmax(c0))  # Pulse peak, sampling instant of each symbol
k_sym = np.arange(N_rx) * sps + offset


def upsample(x):
    y = np.zeros(N_sym * sps, dtype=complex)
    y[::sps] = x
    return y


# GMSK: phase moves by +-pi/2 per bit, constant envelope
freq = np.convolve(upsample(bits).real, g)[: N_sym * sps]
s_gmsk = np.exp(1j * np.pi * np.cumsum(freq) / sps)


# 8-PSK: symbols e^(j 2 pi l/8), optionally rotated by 3 pi/8 per symbol
def psk_waveform(rotate):
    x = np.exp(1j * 2 * np.pi * psk_idx / 8)
    if rotate:
        x = x * np.exp(1j * 3 * np.pi / 8 * np.arange(N_sym))
    s = np.convolve(upsample(x), c0)[: N_sym * sps]
    return s / np.sqrt(np.mean(np.abs(s[k_sym[5:-5]]) ** 2)), x


# Ideal equalizer: remove the known neighbor symbols (C0 is not ISI-free)
taps = {j: c0[offset + j * sps] for j in range(-2, 3)}


def equalize(r, out, x, rotate):
    xp = np.concatenate([np.zeros(2), x, np.zeros(2)])
    isi = {j: xp[2 - j : 2 - j + N_rx] for j in taps}
    model = sum(taps[j] * isi[j] for j in taps)
    gain = np.vdot(model, out[k_sym]) / np.vdot(model, model)
    y = (r - gain * (model - taps[0] * isi[0])) / (gain * taps[0])
    if rotate:
        y = y * np.exp(-1j * 3 * np.pi / 8 * np.arange(N_rx))
    return y


def papr_db(s):
    p = np.abs(s[10 * sps : -10 * sps]) ** 2
    return 10 * np.log10(np.max(p) / np.mean(p))


def min_env(s):
    e = np.abs(s[10 * sps : -10 * sps])
    return np.min(e) / np.sqrt(np.mean(e**2))


# Solid-state amplifier (Rapp model), saturation amplitude 1
def amplifier(s, ibo_db):
    x = s * 10 ** (-ibo_db / 20)
    y = x / (1 + np.abs(x) ** (2 * p_rapp)) ** (1 / (2 * p_rapp))
    return y / np.sqrt(np.mean(np.abs(y) ** 2))


def psd_db(s, nfft=1024):
    win = np.hanning(nfft)
    segs = [s[i : i + nfft] * win for i in range(0, len(s) - nfft, nfft // 2)]
    P = np.mean(np.abs(np.fft.fft(segs, axis=1)) ** 2, axis=0)
    P = np.fft.fftshift(P)
    return 10 * np.log10(P / np.max(P))


f_khz = np.fft.fftshift(np.fft.fftfreq(1024, 1 / fs)) / 1e3

# Simplified spectrum mask relative to the carrier (approximate)
mask_f = np.array([0, 100, 200, 250, 400, 600])
mask_db = np.array([0.5, 0.5, -30, -33, -60, -60])

# Bit error rates in AWGN, ideal amplifier
snr_axis = np.linspace(0, 30, 301)


def ber_gmsk(snr_db):
    return q_func(np.sqrt(2 * 0.68 * 10 ** (np.asarray(snr_db) / 10)))


def ber_8psk(snr_db):
    es_n0 = 10 ** (np.asarray(snr_db) / 10)
    return 2 / 3 * q_func(np.sqrt(2 * es_n0) * np.sin(np.pi / 8))


fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
plt.subplots_adjust(bottom=0.2, hspace=0.38, wspace=0.3)
ax_gmsk, ax_psk, ax_env = axs[0]
ax_psd, ax_ber, ax_info = axs[1]

for ax, name in [
    (ax_gmsk, "GMSK: raw samples"),
    (ax_psk, "8-PSK: after ideal equalizer"),
]:
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.set_xlabel("In-phase I")
    ax.set_ylabel("Quadrature Q")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
(traj_gmsk,) = ax_gmsk.plot(
    [], [], color="tab:blue", lw=0.5, alpha=0.3, label="Trajectory"
)
(pts_gmsk,) = ax_gmsk.plot(
    [], [], ".", color="tab:blue", ms=3, label="Symbol samples"
)
(traj_psk,) = ax_psk.plot(
    [], [], color="tab:orange", lw=0.5, alpha=0.3, label="Trajectory"
)
(pts_psk,) = ax_psk.plot(
    [], [], ".", color="tab:orange", ms=3, label="Symbols"
)
for ax in (ax_gmsk, ax_psk):
    ax.legend(loc="upper right", fontsize=7)

t_env = np.arange(30 * sps) / fs * 1e6
(env_gmsk,) = ax_env.plot([], [], color="tab:blue", label="GMSK")
(env_psk,) = ax_env.plot([], [], color="tab:orange", label="8-PSK")
ax_env.set_xlim(0, t_env[-1])
ax_env.set_ylim(0, 2.0)
ax_env.set_xlabel(r"Time ($\mu$s)")
ax_env.set_ylabel("Envelope |s(t)|")
ax_env.set_title("Envelope after amplifier")
ax_env.grid(True, alpha=0.3)
ax_env.legend(loc="upper right")

(psd_gmsk,) = ax_psd.plot(
    f_khz, np.zeros_like(f_khz), color="tab:blue", label="GMSK"
)
(psd_psk,) = ax_psd.plot(
    f_khz, np.zeros_like(f_khz), color="tab:orange", label="8-PSK"
)
ax_psd.plot(
    np.concatenate([-mask_f[::-1], mask_f]),
    np.concatenate([mask_db[::-1], mask_db]),
    "k--",
    lw=1,
    label="Mask (approx.)",
)
ax_psd.set_xlim(-600, 600)
ax_psd.set_ylim(-80, 5)
ax_psd.set_xlabel("Frequency offset (kHz)")
ax_psd.set_ylabel("PSD (dB)")
ax_psd.set_title("Spectrum after amplifier, 200 kHz carrier")
ax_psd.grid(True, alpha=0.3)
ax_psd.legend(loc="lower center", fontsize=8)

ax_ber.semilogy(snr_axis, ber_gmsk(snr_axis), color="tab:blue", label="GMSK")
ax_ber.semilogy(
    snr_axis, ber_8psk(snr_axis), color="tab:orange", label="8-PSK"
)
line_snr = ax_ber.axvline(init_snr, color="tab:red", ls="--")
ax_ber.set_xlim(0, 30)
ax_ber.set_ylim(1e-6, 0.5)
ax_ber.set_xlabel(r"$E_s/N_0$ (dB)")
ax_ber.set_ylabel("Bit error rate")
ax_ber.set_title("AWGN, ideal amplifier")
ax_ber.grid(True, which="both", alpha=0.3)
ax_ber.legend(loc="lower left")

ax_info.axis("off")
info = ax_info.text(0.0, 1.0, "", va="top", family="monospace", fontsize=10)

rotate = True
s_psk, x_psk = psk_waveform(rotate)


def update(_=None):
    snr = slider_snr.val
    ibo = slider_ibo.val
    n0 = 10 ** (-snr / 10)

    out_gmsk = amplifier(s_gmsk, ibo)
    out_psk = amplifier(s_psk, ibo)

    rx_gmsk = out_gmsk[k_sym] + np.sqrt(n0) * noise_base
    rx_psk = out_psk[k_sym] + np.sqrt(n0) * noise_base
    rx_psk = equalize(rx_psk, out_psk, x_psk, rotate)
    traj = slice(10 * sps, 10 * sps + 100 * sps)
    traj_gmsk.set_data(out_gmsk[traj].real, out_gmsk[traj].imag)
    traj_psk.set_data(out_psk[traj].real, out_psk[traj].imag)
    pts_gmsk.set_data(
        rx_gmsk[10 : 10 + N_show].real, rx_gmsk[10 : 10 + N_show].imag
    )
    pts_psk.set_data(
        rx_psk[10 : 10 + N_show].real, rx_psk[10 : 10 + N_show].imag
    )

    env = slice(10 * sps, 10 * sps + len(t_env))
    env_gmsk.set_data(t_env, np.abs(out_gmsk[env]))
    env_psk.set_data(t_env, np.abs(out_psk[env]))

    psd_gmsk.set_ydata(psd_db(out_gmsk))
    psd_psk.set_ydata(psd_db(out_psk))
    line_snr.set_xdata([snr, snr])

    info.set_text(
        f"Symbol rate     {R_s / 1e3:.3f} ksym/s\n"
        f"Carrier         200 kHz (both)\n\n"
        f"                 GMSK    8-PSK\n"
        f"Bit/symbol          1        3\n"
        f"kbit/s/carrier {R_s / 1e3:7.1f}  {3 * R_s / 1e3:7.1f}\n"
        f"PAPR before PA {papr_db(s_gmsk):5.1f} dB {papr_db(s_psk):5.1f} dB\n"
        f"Min. envelope  {min_env(s_gmsk):8.2f} {min_env(s_psk):8.2f}\n"
        f"BER at {snr:4.1f} dB {ber_gmsk(snr):8.1e} {ber_8psk(snr):8.1e}\n\n"
        f"Same power and bandwidth:\n"
        f"8-PSK triples the bit rate,\n"
        f"but needs a higher SNR and\n"
        f"a linear amplifier."
    )
    fig.canvas.draw_idle()


def toggle(_):
    global s_psk, x_psk, rotate
    rotate = check.get_status()[0]
    s_psk, x_psk = psk_waveform(rotate)
    update()


ax_slider_snr = plt.axes([0.3, 0.09, 0.55, 0.03])
ax_slider_ibo = plt.axes([0.3, 0.04, 0.55, 0.03])
slider_snr = Slider(
    ax_slider_snr, r"$E_s/N_0$ (dB)", 0.0, 30.0, valinit=init_snr
)
slider_ibo = Slider(
    ax_slider_ibo, "Amplifier backoff (dB)", -3.0, 12.0, valinit=init_ibo
)
slider_snr.on_changed(update)
slider_ibo.on_changed(update)

ax_check = plt.axes([0.03, 0.03, 0.14, 0.08])
check = CheckButtons(ax_check, [r"3$\pi$/8 rotation"], [True])
check.on_clicked(toggle)

update()
plt.show()
