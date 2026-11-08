import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

os_ = 32  # Samples per symbol duration T
nfft = 2**16  # FFT length for a smooth spectrum
init_alpha = 0.22  # RRC roll-off (UMTS)
init_bt = 0.3  # Gaussian bandwidth-time product (GSM)
init_span = 8  # Truncation length of RRC and Gaussian pulses (symbols)
umts_edge = 2.5 / 3.84  # Edge of the 5 MHz UMTS channel in units of 1/T

names = ["Rectangular", "Triangular", "Gaussian", "Root-raised-cosine"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def rect(t, *_):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def tri(t, *_):
    return np.clip(1 - np.abs(t), 0, None)


def gauss(t, _, bt):
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    return np.exp(-(t**2) / (2 * sigma**2))


def rrc(t, alpha, _):
    if alpha < 1e-3:
        return np.sinc(t)
    a = alpha
    num = np.sin(np.pi * t * (1 - a)) + 4 * a * t * np.cos(np.pi * t * (1 + a))
    den = np.pi * t * (1 - (4 * a * t) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = num / den
    h[np.isclose(t, 0)] = 1 - a + 4 * a / np.pi
    edge = np.isclose(np.abs(t), 1 / (4 * a))
    h[edge] = (
        a
        / np.sqrt(2)
        * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * a))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a))
        )
    )
    return h


pulse_funcs = [rect, tri, gauss, rrc]


def make_pulse(func, alpha, bt, span):
    # Rectangular and triangular pulses have a fixed, short support
    if func is rect:
        t = (np.arange(-1, os_ + 1) + 0.5) / os_ - 0.5  # One zero per side
    else:
        half = 1.0 if func is tri else span / 2
        t = np.arange(-half * os_, half * os_ + 1) / os_
    p = func(t, alpha, bt)
    return t, p / np.sqrt(np.sum(p**2) / os_)  # Unit energy


def spectrum_db(p):
    P = np.abs(np.fft.fftshift(np.fft.fft(p, nfft))) ** 2
    return 10 * np.log10(P / P.max() + 1e-15)


def bw99(p):
    # Two-sided bandwidth containing 99 % of the pulse energy (in 1/T)
    P = np.abs(np.fft.fft(p, nfft)) ** 2
    P_pos = np.r_[P[0] / 2, P[1 : nfft // 2]]
    cum = np.cumsum(P_pos) / np.sum(P_pos)
    return 2 * np.argmax(cum >= 0.99) * os_ / nfft


def isi(p):
    # Transmit and matched receive filter, sampled at multiples of T
    g = np.convolve(p, p[::-1])
    c = len(g) // 2
    k = np.arange(1, c // os_ + 1)
    return np.max(np.abs(g[c + k * os_])) / g[c] if k.size else 0.0


f_axis = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / os_))

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1], width_ratios=[1, 1.4])
ax_time = fig.add_subplot(gs[0, 0])
ax_freq = fig.add_subplot(gs[0, 1])
ax_info = fig.add_subplot(gs[1, :])
plt.subplots_adjust(bottom=0.2, hspace=0.35, wspace=0.25, top=0.94)

lines_t = [ax_time.plot([], [], color=c, lw=2)[0] for c in colors]
lines_f = [ax_freq.plot([], [], color=c, lw=1.8)[0] for c in colors]

ax_time.set_xlim(-5, 5)
ax_time.set_ylim(-0.4, 1.6)
ax_time.set_xlabel("Time $t / T$")
ax_time.set_ylabel("Amplitude (unit energy)")
ax_time.set_title("Pulse shapes")
ax_time.grid(True, alpha=0.3)

ax_freq.set_xlim(-3, 3)
ax_freq.set_ylim(-80, 3)
ax_freq.set_xlabel(r"Frequency $f \cdot T$")
ax_freq.set_ylabel("Energy spectrum (dB)")
ax_freq.set_title("Spectra")
ax_freq.grid(True, alpha=0.3)
for x in (-0.5, 0.5):
    ax_freq.axvline(x, color="gray", ls=":", lw=1)
for x in (-umts_edge, umts_edge):
    ax_freq.axvline(x, color="k", ls="--", lw=1)
ax_freq.text(umts_edge + 0.05, -5, "UMTS channel\nedge", fontsize=8)
ax_freq.text(0.05, -78, "Nyquist\n1/2T", fontsize=8, color="gray")

ax_info.axis("off")
info = ax_info.text(0.0, 1.0, "", va="top", family="monospace", fontsize=10)


def update(_=None):
    alpha = slider_alpha.val
    bt = slider_bt.val
    span = int(slider_span.val)
    visible = check.get_status()

    rows = [
        f"{'Pulse':20s}{'99 % bandwidth':>16s}{'ISI after matched filter':>27s}"
    ]
    for i, func in enumerate(pulse_funcs):
        t, p = make_pulse(func, alpha, bt, span)
        lines_t[i].set_data(t, p)
        lines_f[i].set_data(f_axis, spectrum_db(p))
        lines_t[i].set_visible(visible[i])
        lines_f[i].set_visible(visible[i])
        rows.append(f"{names[i]:20s}{bw99(p):13.2f} /T{100 * isi(p):24.1f} %")
    rows.append("")
    rows.append(
        f"RRC: alpha = {alpha:.2f}, ideal bandwidth (1 + alpha)/T = "
        f"{1 + alpha:.2f}/T, truncated to {span} symbols"
    )
    info.set_text("\n".join(rows))
    fig.canvas.draw_idle()


ax_slider_alpha = plt.axes([0.38, 0.1, 0.5, 0.025])
ax_slider_bt = plt.axes([0.38, 0.06, 0.5, 0.025])
ax_slider_span = plt.axes([0.38, 0.02, 0.5, 0.025])
slider_alpha = Slider(
    ax_slider_alpha, r"RRC roll-off $\alpha$", 0.0, 1.0, valinit=init_alpha
)
slider_bt = Slider(ax_slider_bt, "Gaussian BT", 0.2, 1.0, valinit=init_bt)
slider_span = Slider(
    ax_slider_span,
    "Truncation (symbols)",
    2,
    16,
    valinit=init_span,
    valstep=1,
)

ax_check = plt.axes([0.02, 0.02, 0.17, 0.12])
check = CheckButtons(ax_check, names, [True] * len(names))
for lbl, c in zip(check.labels, colors):
    lbl.set_color(c)

for s in (slider_alpha, slider_bt, slider_span):
    s.on_changed(update)
check.on_clicked(update)

update()
plt.show()
