import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

N_bits = 320  # Data bits per user
N_show = 4  # Bits shown in the time-domain plots
K_max = 16  # Maximum number of users
os_ = 8  # Samples per chip for the spectrum
init_log_sf = 3  # Initial spreading factor 2^3 = 8
init_users = 4
init_pwr = 0.0  # Power of the other users relative to user 1 (dB)
init_ebn0 = 10.0  # Eb/N0 of user 1 (dB)

rng = np.random.default_rng(42)
bits = rng.choice([-1, 1], size=(K_max, N_bits))
scrambling = rng.choice([-1, 1], size=64 * N_bits)  # Common to all users


# Orthogonal Walsh codes: rows of a Hadamard matrix (Sylvester construction)
def hadamard(n):
    H = np.array([[1]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def codes(sf, kind):
    if kind == "Walsh (orthogonal)":
        return hadamard(sf)[1:]  # Row 0 (all ones) does not spread
    return np.random.default_rng(sf).choice([-1, 1], size=(K_max, sf))


def psd_db(x, nfft, fs):
    win = np.hanning(nfft)
    segs = [
        x[i : i + nfft] * win for i in range(0, len(x) - nfft + 1, nfft // 2)
    ]
    P = np.mean(np.abs(np.fft.rfft(segs, axis=1)) ** 2, axis=0)
    P = P / (fs * np.sum(win**2))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f[1:], 10 * np.log10(P[1:] + 1e-12)


fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
plt.subplots_adjust(bottom=0.3, hspace=0.4, wspace=0.25)
ax_tx, ax_rx = axs[0]
ax_corr, ax_psd = axs[1]


def update(_=None):
    log_sf = int(slider_sf.val)
    sf = 2**log_sf
    slider_sf.valtext.set_text(f"{sf}")
    kind = radio.value_selected
    C = codes(sf, kind)
    K = min(int(slider_users.val), len(C))
    amp_other = 10 ** (slider_pwr.val / 20)
    ebn0 = 10 ** (slider_ebn0.val / 10)

    # Spreading: every data bit is multiplied with the user's code
    tx = (bits[:K, :, None] * C[:K, None, :]).reshape(K, -1)
    scr = check.get_status()[0]
    s = scrambling[: N_bits * sf] if scr else np.ones(N_bits * sf)
    tx = tx * s  # Same scrambling for all users keeps codes orthogonal
    amps = np.r_[1.0, np.full(K - 1, amp_other)]
    sigma = np.sqrt(sf / (2 * ebn0))  # Eb = sf chips of energy 1
    noise = np.random.default_rng(3).standard_normal(N_bits * sf) * sigma
    r = amps @ tx + noise

    # Despreading of user 1: correlate each bit interval with its code
    z = (r * s).reshape(N_bits, sf) @ C[0] / sf
    ber = np.mean(np.sign(z) != bits[0])

    n = N_show * sf
    chips = np.arange(n + 1)
    data = np.repeat(bits[0, :N_show], sf)
    ax_tx.clear()
    ax_tx.step(chips, np.r_[data, data[-1]] + 3, where="post", label="Data")
    ax_tx.step(
        chips, np.r_[tx[0, :n], tx[0, n - 1]], where="post", label="Chips"
    )
    ax_rx.clear()
    ax_rx.step(chips, np.r_[r[:n], r[n - 1]], where="post", color="tab:gray")
    for ax in (ax_tx, ax_rx):
        for b in range(N_show + 1):
            ax.axvline(b * sf, color="k", lw=0.5, ls="--")
        ax.set_xlim(0, n)
        ax.set_xlabel("Chip index")
        ax.grid(True, alpha=0.3)
    ax_tx.set_yticks([-1, 1, 2, 4], ["-1", "+1", "-1", "+1"])
    ax_tx.set_title(f"User 1: data bits and chips (SF = {sf})")
    ax_tx.legend(loc="upper right", fontsize=8)
    ax_rx.set_title(f"Received: sum of {K} users + noise")

    ax_corr.clear()
    idx = np.arange(100)
    ax_corr.scatter(idx, z[:100], c=bits[0, :100], cmap="coolwarm", s=12)
    ax_corr.axhline(0, color="k", lw=0.8)
    ax_corr.set_ylim(-4, 4)
    ax_corr.set_xlabel("Bit index")
    ax_corr.set_ylabel("Correlator output")
    note = " (max. SF-1 Walsh codes)" if K < int(slider_users.val) else ""
    ax_corr.set_title(f"User 1 after despreading: BER = {ber:.3f}{note}")
    ax_corr.grid(True, alpha=0.3)

    # Spectrum: rectangular data pulses vs. chips, same power
    fs = sf * os_  # Samples per bit, frequency axis in units of the bit rate
    f, P_data = psd_db(np.repeat(bits[0], sf * os_), 32 * fs, fs)
    _, P_chip = psd_db(np.repeat(tx[0], os_), 32 * fs, fs)
    ax_psd.clear()
    ax_psd.semilogx(f, P_data, label="Data")
    ax_psd.semilogx(f, P_chip, label="Spread signal")
    ax_psd.set_xlim(0.05, 100)
    ax_psd.set_ylim(-50, 5)
    ax_psd.set_xlabel(r"Frequency $f / R_\mathrm{b}$")
    ax_psd.set_ylabel("PSD (dB)")
    if scr:
        title = f"{sf}x wider, {10 * np.log10(sf):.0f} dB lower density"
    else:
        title = "short code repeats every bit: spectral lines"
    ax_psd.set_title(f"Spectrum: {title}")
    ax_psd.grid(True, which="both", alpha=0.3)
    ax_psd.legend(loc="lower left", fontsize=8)
    fig.canvas.draw_idle()


ax_slider_sf = plt.axes([0.35, 0.19, 0.5, 0.025])
ax_slider_users = plt.axes([0.35, 0.14, 0.5, 0.025])
ax_slider_pwr = plt.axes([0.35, 0.09, 0.5, 0.025])
ax_slider_ebn0 = plt.axes([0.35, 0.04, 0.5, 0.025])
slider_sf = Slider(
    ax_slider_sf, "Spreading factor", 1, 6, valinit=init_log_sf, valstep=1
)
slider_users = Slider(
    ax_slider_users, "Users", 1, K_max, valinit=init_users, valstep=1
)
slider_pwr = Slider(
    ax_slider_pwr, "Other users power (dB)", -10.0, 30.0, valinit=init_pwr
)
slider_ebn0 = Slider(
    ax_slider_ebn0, r"$E_b/N_0$ user 1 (dB)", 0.0, 20.0, valinit=init_ebn0
)

ax_radio = plt.axes([0.03, 0.04, 0.17, 0.12])
radio = RadioButtons(ax_radio, ["Walsh (orthogonal)", "Random"])
ax_check = plt.axes([0.03, 0.18, 0.17, 0.05])
check = CheckButtons(ax_check, ["Scrambling code"], [False])

for s in (slider_sf, slider_users, slider_pwr, slider_ebn0):
    s.on_changed(update)
radio.on_clicked(update)
check.on_clicked(update)

update()
plt.show()
