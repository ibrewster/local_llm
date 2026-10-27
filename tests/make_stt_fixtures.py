"""Generates the STT test fixtures in tests/data (macOS only: uses `say`).

Clean speech is synthesized with `say`; "difficult" variants are built by mixing in
seeded noise / competing speech at a fixed signal-to-noise ratio, so the files are
reproducible. Run from the project root:
    python tests/make_stt_fixtures.py
"""
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RATE = 16000
SEED = 1234

FOX = "The quick brown fox jumps over the lazy dog."
# No numbers: models disagree on "72" vs "seventy two", which would pollute the WER
COMMAND = "Turn off the living room lights and set the thermostat to a cooler temperature before the guests arrive tonight."
PARAGRAPH = (
    "Yesterday I walked to the market near the old train station, and bought three oranges, "
    "a loaf of bread, and some fresh basil. The vendor told me that the rain would "
    "probably arrive before noon, so I hurried home."
)
BABBLE = (
    "Honestly I told them last week that the shipment would be late. Nobody listens to "
    "the schedule anymore, and the meeting about the budget got moved again to Thursday. "
    "Did you see the game last night? It went into overtime and the crowd went completely wild."
)


def say(text: str, voice: str, wpm: int | None = None) -> np.ndarray:
    """Synthesize `text` with macOS `say`, returning float32 mono samples at 16 kHz."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "speech.wav"
        cmd = ["say", "-v", voice, "--data-format=LEI16@16000", "-o", str(out)]
        if wpm:
            cmd += ["-r", str(wpm)]
        subprocess.run(cmd + [text], check=True)
        with wave.open(str(out), "rb") as wav:
            assert wav.getframerate() == RATE and wav.getnchannels() == 1
            return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Noise with a 1/f power spectrum (closer to room/traffic noise than white noise)."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    return np.fft.irfft(spectrum / np.sqrt(freqs), n)


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Adds `noise` (looped/trimmed to fit) to `speech` at the given SNR in dB."""
    noise = np.resize(noise, speech.shape)
    noise = noise * (rms(speech) / rms(noise)) / (10 ** (snr_db / 20))
    return speech + noise


def pad(x: np.ndarray, seconds: float = 0.5) -> np.ndarray:
    """Leading/trailing silence, like a real recording (also avoids clipped onsets)."""
    silence = np.zeros(int(seconds * RATE), dtype=np.float32)
    return np.concatenate([silence, x, silence])


def write_wav(name: str, samples: np.ndarray) -> None:
    peak = np.max(np.abs(samples))
    if peak > 0.99:  # normalize only if mixing clipped; otherwise keep the level as designed
        samples = samples * (0.99 / peak)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    path = DATA_DIR / name
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())
    print(f"{path.name}: {len(pcm) / RATE:.1f}s")


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    fox = pad(say(FOX, "Samantha"))
    write_wav("fox_pink_noise_5db.wav", mix(fox, pink_noise(len(fox), rng), snr_db=5))

    command = pad(say(COMMAND, "Daniel"))
    write_wav("command_white_noise_10db.wav", mix(command, rng.standard_normal(len(command)), snr_db=10))

    paragraph = pad(say(PARAGRAPH, "Daniel"))
    write_wav("paragraph_clean.wav", paragraph)

    # Competing speaker ("cocktail party"): the hardest case for most ASR models
    babble = say(BABBLE, "Samantha", wpm=190)
    write_wav("paragraph_babble_5db.wav", mix(paragraph, babble, snr_db=5))

    # Fast, quiet speech over pink noise: far-field voice-assistant conditions
    fast = pad(say(COMMAND, "Samantha", wpm=290))
    quiet = fast * 0.05
    write_wav("command_fast_quiet_noisy.wav", mix(quiet, pink_noise(len(quiet), rng), snr_db=8))


if __name__ == "__main__":
    main()
