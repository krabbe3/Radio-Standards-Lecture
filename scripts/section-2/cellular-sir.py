import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

init_n = 4.0  # Initial path loss exponent
init_sir_req = 18.0  # Initial required SIR (dB)
total_channels = 832  # AMPS duplex channels

# Valid cluster sizes N = i^2 + ij + j^2
clusters = sorted(
    {
        i**2 + i * j + j**2
        for i in range(0, 6)
        for j in range(0, 6)
        if i + j > 0
    }
)
clusters = np.array([N for N in clusters if N <= 28])


# Co-channel SIR with six first-tier interferers
def sir_db(N, n):
    return 10 * np.log10(np.sqrt(3 * N) ** n / 6)


def smallest_cluster(n, sir_req):
    valid = clusters[sir_db(clusters, n) >= sir_req]
    return valid[0] if valid.size else None


fig, (ax_sir, ax_cap) = plt.subplots(1, 2, figsize=(11, 5))
plt.subplots_adjust(bottom=0.25, wspace=0.3)

(line_sir,) = ax_sir.plot(
    clusters, sir_db(clusters, init_n), "o-", color="tab:blue"
)
line_req = ax_sir.axhline(
    init_sir_req, color="tab:red", ls="--", label="Required SIR"
)
ax_sir.set_xlabel("Cluster size N")
ax_sir.set_ylabel("SIR (dB)")
ax_sir.set_ylim(0, 45)
ax_sir.grid(True, alpha=0.3)
ax_sir.legend(loc="lower right")

bars = ax_cap.bar(clusters, total_channels / clusters, color="tab:gray")
ax_cap.set_xlabel("Cluster size N")
ax_cap.set_ylabel("Channels per cell")
ax_cap.grid(True, alpha=0.3)


def update(_):
    n = slider_n.val
    sir_req = slider_sir.val
    line_sir.set_ydata(sir_db(clusters, n))
    line_req.set_ydata([sir_req, sir_req])

    N_min = smallest_cluster(n, sir_req)
    for bar, N in zip(bars, clusters):
        bar.set_color("tab:green" if N == N_min else "tab:gray")
    if N_min is None:
        ax_cap.set_title("Required SIR not reachable")
    else:
        ax_cap.set_title(
            f"N = {N_min}: {total_channels / N_min:.0f} channels per cell"
        )
    fig.canvas.draw_idle()


ax_slider_n = plt.axes([0.15, 0.1, 0.7, 0.03])
ax_slider_sir = plt.axes([0.15, 0.05, 0.7, 0.03])
slider_n = Slider(ax_slider_n, "Path loss n", 2.0, 5.0, valinit=init_n)
slider_sir = Slider(
    ax_slider_sir, "Required SIR (dB)", 5.0, 30.0, valinit=init_sir_req
)
slider_n.on_changed(update)
slider_sir.on_changed(update)

update(None)
plt.show()
