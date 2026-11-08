from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

T = 1.0  # Symbol (chip) duration
alphas = [0.0, 0.22, 1.0]  # Roll-off factors, 0.22 as in UMTS
colors = ["#1f77b4", "#d62728", "#2ca02c"]

t = np.linspace(-5, 5, 2001)
f = np.linspace(-1.2, 1.2, 1001)


# Raised-cosine pulse: si function times a window that decays like 1/t^2
def rc_time(t, alpha):
    p = np.sinc(t / T)
    if alpha == 0:
        return p
    denom = 1 - (2 * alpha * t / T) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(
            np.isclose(denom, 0),
            np.pi / 4,
            np.cos(np.pi * alpha * t / T) / denom,
        )
    return p * w


# Raised-cosine spectrum: flat up to (1-a)/2T, cosine roll-off to (1+a)/2T
def rc_freq(f, alpha):
    f1 = (1 - alpha) / (2 * T)
    f2 = (1 + alpha) / (2 * T)
    af = np.abs(f)
    P = np.where(af <= f1, T, 0.0)
    if alpha > 0:
        roll = (af > f1) & (af <= f2)
        P = np.where(
            roll,
            T / 2 * (1 + np.cos(np.pi * T / alpha * (af - f1))),
            P,
        )
    return P


fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(9, 7))
fig.tight_layout(pad=4.0)

for alpha, c in zip(alphas, colors):
    ax_time.plot(
        t,
        rc_time(t, alpha),
        color=c,
        linewidth=2.5,
        label=rf"$\alpha$ = {alpha}",
    )
    ax_freq.plot(
        f,
        rc_freq(f, alpha),
        color=c,
        linewidth=2.5,
        label=rf"$\alpha$ = {alpha}",
    )

k = np.arange(-5, 6)
ax_time.plot(
    k, np.where(k == 0, 1.0, 0.0), "ko", markersize=5, label="Samples $kT$"
)
ax_time.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax_time.set_title("Time Domain: Raised-Cosine Pulse", fontsize=14, pad=10)
ax_time.set_xlabel("Time $t / T$", fontsize=11)
ax_time.set_ylabel("Amplitude", fontsize=11)
ax_time.set_xticks(k)
ax_time.grid(True, linestyle=":", alpha=0.6)
ax_time.legend(loc="upper right", fontsize=10)

for x in [-0.5, 0.5]:
    ax_freq.axvline(x, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
ax_freq.set_title(
    "Frequency Domain: Raised-Cosine Spectrum", fontsize=14, pad=10
)
ax_freq.set_xlabel("Frequency $f \\cdot T$", fontsize=11)
ax_freq.set_ylabel("Spectral Density", fontsize=11)
ax_freq.set_xticks([-1, -0.5, 0, 0.5, 1])
ax_freq.set_xticklabels([r"$-1/T$", r"$-1/2T$", "$0$", r"$1/2T$", r"$1/T$"])
ax_freq.grid(True, linestyle=":", alpha=0.6)
ax_freq.legend(loc="upper right", fontsize=10)

out = Path(__file__).resolve().parents[2] / "presentation/images/section-2"
fig.savefig(out / "raised_cosine.pdf")

plt.show()
