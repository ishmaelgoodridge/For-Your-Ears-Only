import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import pyaudio
import numpy as np
import wave
import hashlib
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Audio Configuration ---
CHUNK = 1024 * 2
CHANNELS = 1
RATE = 44100
OUTPUT_FILENAME = "unscrambled_output.wav"


def get_seed_from_passkey(passkey: str, chunk_idx: int) -> int:
    """Derive a stable seed for each chunk from passkey + chunk index."""
    combo = f"{passkey}:{chunk_idx}"
    hashed = hashlib.sha256(combo.encode()).digest()
    return int.from_bytes(hashed[:4], "big")


def unscramble_chunk(data_np: np.ndarray, passkey: str, chunk_idx: int) -> np.ndarray:
    """Reverses scrambling for a single audio chunk."""
    fft_data = np.fft.rfft(data_np)

    # Generate the *same shuffle* used in scrambling
    seed = get_seed_from_passkey(passkey, chunk_idx)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(fft_data))
    rng.shuffle(indices)

    # Invert shuffle
    unscrambled_fft = np.empty_like(fft_data)
    unscrambled_fft[indices] = fft_data

    unscrambled_audio = np.fft.irfft(unscrambled_fft, n=len(data_np))
    unscrambled_audio = np.clip(unscrambled_audio, -32768, 32767)
    return unscrambled_audio.astype(np.int16)


class AudioUnscramblerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔊 Audio Unscrambler")
        self.root.geometry("650x550")
        self.root.configure(bg="#2b2b2b")

        tk.Label(root, text="Enter Passkey:", fg="white", bg="#2b2b2b", font=("Arial", 12)).pack(pady=10)
        self.passkey_entry = tk.Entry(root, show="*", font=("Arial", 12), width=25, justify="center")
        self.passkey_entry.pack(pady=5)

        self.file_btn = tk.Button(root, text="📂 Choose Scrambled WAV", command=self.choose_file,
                                  font=("Arial", 11), bg="#2196F3", fg="white", width=25, height=2)
        self.file_btn.pack(pady=10)

        self.start_btn = tk.Button(root, text="▶ Unscramble File", command=self.start_unscramble,
                                   font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", width=20, height=2,
                                   state=tk.DISABLED)
        self.start_btn.pack(pady=5)

        self.play_btn = tk.Button(root, text="🎵 Play Unscrambled Audio", command=self.play_audio,
                                  font=("Arial", 11), bg="#9C27B0", fg="white", width=25, height=2,
                                  state=tk.DISABLED)
        self.play_btn.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

        # Waveform visualization
        self.fig = Figure(figsize=(6, 3), dpi=100, facecolor="#2b2b2b")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax.set_ylim([-32768, 32767])
        self.ax.set_xlim([0, CHUNK])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.line, = self.ax.plot([], [], color="cyan")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        self.filepath = None
        self.thread = None
        self.unscrambled_data = None  # store full audio for visualization

    def choose_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if self.filepath:
            self.start_btn.config(state=tk.NORMAL)
        else:
            self.start_btn.config(state=tk.DISABLED)

    def start_unscramble(self):
        passkey = self.passkey_entry.get()
        if not passkey:
            messagebox.showerror("Error", "Please enter a passkey before unscrambling.")
            return
        if not self.filepath:
            messagebox.showerror("Error", "Please select a scrambled WAV file.")
            return

        self.passkey = passkey
        self.thread = threading.Thread(target=self.unscramble)
        self.thread.start()

    def unscramble(self):
        with wave.open(self.filepath, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            if n_channels != CHANNELS or framerate != RATE or sampwidth != 2:
                messagebox.showerror("Error", "WAV format does not match expected scrambled audio format.")
                return

            scrambled_data = wf.readframes(n_frames)

        total_samples = len(scrambled_data) // 2
        scrambled_np = np.frombuffer(scrambled_data, dtype=np.int16)

        unscrambled_frames = []
        chunk_idx = 0

        num_chunks = total_samples // CHUNK
        self.progress["maximum"] = num_chunks

        result_np = []

        for i in range(0, total_samples, CHUNK):
            chunk = scrambled_np[i:i + CHUNK]
            if len(chunk) < CHUNK:
                break
            unscrambled_chunk_np = unscramble_chunk(chunk, self.passkey, chunk_idx)
            unscrambled_frames.append(unscrambled_chunk_np.tobytes())
            result_np.extend(unscrambled_chunk_np.tolist())
            chunk_idx += 1
            self.progress["value"] = chunk_idx
            self.root.update_idletasks()

        # Save unscrambled audio
        with wave.open(OUTPUT_FILENAME, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(unscrambled_frames))

        self.unscrambled_data = np.array(result_np, dtype=np.int16)
        self.play_btn.config(state=tk.NORMAL)

        # Draw waveform of full file
        self.line.set_data(np.arange(len(self.unscrambled_data)), self.unscrambled_data)
        self.ax.set_xlim(0, len(self.unscrambled_data))
        self.canvas.draw_idle()

        messagebox.showinfo("Done", f"✅ Unscrambled file saved as:\n{os.path.abspath(OUTPUT_FILENAME)}")

    def play_audio(self):
        def _play():
            wf = wave.open(OUTPUT_FILENAME, "rb")
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            data = wf.readframes(CHUNK)
            idx = 0
            while data:
                stream.write(data)
                audio_np = np.frombuffer(data, dtype=np.int16)

                # update waveform chunk
                self.line.set_data(np.arange(len(audio_np)) + idx, audio_np)
                self.ax.set_xlim(idx, idx + len(audio_np))
                self.canvas.draw_idle()

                idx += len(audio_np)
                data = wf.readframes(CHUNK)

            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()

        threading.Thread(target=_play).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioUnscramblerGUI(root)
    root.mainloop()
