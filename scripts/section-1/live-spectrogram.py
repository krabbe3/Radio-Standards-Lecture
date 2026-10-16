import queue
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
HISTORY_CHUNKS = 150  # How many time slices to display horizontally

# Rolling buffer for the spectrogram (Frequency x Time)
num_freq_bins = CHUNK_SIZE // 2 + 1
spectrogram_data = np.full((num_freq_bins, HISTORY_CHUNKS), -80.0)

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        pass
    audio_queue.put(indata[:, 0].copy())


fig, ax = plt.subplots(figsize=(10, 5))
max_freq_khz = (SAMPLE_RATE / 2) / 1000

im = ax.imshow(
    spectrogram_data,
    aspect="auto",
    origin="lower",
    cmap="magma",
    extent=[0, HISTORY_CHUNKS, 0, max_freq_khz],
    vmin=-70,
    vmax=0,
)

ax.set_title("Live Audio Spectrogram (MacBook Mic)")
ax.set_xlabel("Time History (Chunks)")
ax.set_ylabel("Frequency (kHz)")
fig.colorbar(im, ax=ax, label="Magnitude (dB)")


def update_plot(frame):
    global spectrogram_data
    updated = False

    while not audio_queue.empty():
        audio_chunk = audio_queue.get_nowait()
        windowed = audio_chunk * np.hanning(len(audio_chunk))
        spectrum = np.abs(np.fft.rfft(windowed))

        db_spectrum = 20 * np.log10(np.maximum(spectrum, 1e-5))

        spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
        spectrogram_data[:, -1] = db_spectrum
        updated = True

    if updated:
        im.set_data(spectrogram_data)
    return [im]


ani = animation.FuncAnimation(fig, update_plot, interval=30, blit=True)

try:
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    ):
        plt.show()
except KeyboardInterrupt:
    pass
