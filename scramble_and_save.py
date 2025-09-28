import tkinter as tk
from tkinter import messagebox
import threading
import pyaudio
import numpy as np
import wave
import hashlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Audio Configuration ---
CHUNK = 1024 * 2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OUTPUT_FILENAME = "scrambled_output.wav"


def get_seed_from_passkey(passkey: str, chunk_idx: int) -> int:
    combo = f"{passkey}:{chunk_idx}"
    hashed = hashlib.sha256(combo.encode()).digest()
    return int.from_bytes(hashed[:4], "big")


def scramble_chunk(data_np: np.ndarray, passkey: str, chunk_idx: int) -> np.ndarray:
    fft_data = np.fft.rfft(data_np)
    seed = get_seed_from_passkey(passkey, chunk_idx)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(fft_data))
    rng.shuffle(indices)
    scrambled_fft = fft_data[indices]
    scrambled_audio = np.fft.irfft(scrambled_fft, n=len(data_np))
    scrambled_audio = np.clip(scrambled_audio, -32768, 32767)
    return scrambled_audio.astype(np.int16)


class AudioScramblerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔊 Audio Scrambler")
        self.root.geometry("600x500")
        self.root.configure(bg="#2b2b2b")

        tk.Label(root, text="Enter Passkey:", fg="white", bg="#2b2b2b", font=("Arial", 12)).pack(pady=10)
        self.passkey_entry = tk.Entry(root, show="*", font=("Arial", 12), width=25, justify="center")
        self.passkey_entry.pack(pady=5)

        self.start_btn = tk.Button(root, text="▶ Start Recording", command=self.start_recording,
                                   font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", width=20, height=2)
        self.start_btn.pack(pady=10)

        self.stop_btn = tk.Button(root, text="⏹ Stop Recording", command=self.stop_recording,
                                  font=("Arial", 12, "bold"), bg="#f44336", fg="white", width=20, height=2,
                                  state=tk.DISABLED)
        self.stop_btn.pack(pady=5)

        # Matplotlib waveform
        self.fig = Figure(figsize=(6, 3), dpi=100, facecolor="#2b2b2b")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax.set_ylim([-32768, 32767])
        self.ax.set_xlim([0, CHUNK])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.line, = self.ax.plot([], [], color="lime")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        self.recording = False
        self.thread = None
        self.frames = []

    def start_recording(self):
        passkey = self.passkey_entry.get()
        if not passkey:
            messagebox.showerror("Error", "Please enter a passkey before recording.")
            return
        self.passkey = passkey
        self.frames = []
        self.recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.thread = threading.Thread(target=self.record)
        self.thread.start()

    def stop_recording(self):
        self.recording = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Stopped", f"⏹ Recording stopped.\nSaved to {OUTPUT_FILENAME}")

    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        chunk_idx = 0
        scrambled_frames = []

        while self.recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16)

            scrambled_chunk_np = scramble_chunk(audio_np, self.passkey, chunk_idx)
            scrambled_frames.append(scrambled_chunk_np.tobytes())

            # Update waveform
            self.line.set_data(np.arange(len(audio_np)), audio_np)
            self.ax.set_xlim(0, len(audio_np))
            self.canvas.draw_idle()

            chunk_idx += 1

        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(OUTPUT_FILENAME, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(scrambled_frames))


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioScramblerGUI(root)
    root.mainloop()
