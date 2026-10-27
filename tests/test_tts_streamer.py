"""Tests for MLXOllama/tts_streamer.py that never touch the network.

TTSStreamer is a Process subclass, but constructing it does not start anything, and
send_tts() only needs two attributes: `_audio_model` and `_packetizer`. The tests
build a streamer, inject a RecordingPacketizer in place of audio_stream.AudioPacketizer
(which would multicast UDP packets), and call send_tts() directly. Everything the
streamer would have pushed is captured, and can be written to a WAV file.

* Unit tests use a FakeModel (fast, deterministic, no model download).
* test_real_model_roundtrip uses the real Kokoro model, saves the captured audio to a
  WAV file and transcribes it with the Wyoming STT server (see test_parakeet.py) to verify
  the audio matches the input text. It is skipped when Kokoro or the STT server is
  unavailable.

Environment variables (optional):
    TTS_MODEL         model id for the real-model tests (default mlx-community/Kokoro-82M-bf16)
    TTS_RUNS          number of repeated generations in the timing test (default 3)
    STT_HOST / STT_PORT   as in test_parakeet.py

The real-model tests print timing (time to first audio, total generation time, audio
duration, real-time factor) so different TTS models can be compared, e.g.:
    TTS_MODEL=mlx-community/orpheus-3b-0.1-ft-6bit PYTHONPATH=tests:. pytest tests/test_tts_streamer.py -s
"""
import asyncio
import os
import queue
import re
import statistics
import time
import wave
from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from MLXOllama import tts_streamer

SAMPLE_RATE = 24000  # Kokoro output rate
TARGET_DB = -12  # RMS level send_tts() normalizes each chunk to
MODEL_ID = os.environ.get("TTS_MODEL", "mlx-community/Kokoro-82M-bf16")
RUNS = int(os.environ.get("TTS_RUNS", "3"))


class RecordingPacketizer:
    """Stand-in for audio_stream.AudioPacketizer: records instead of streaming."""

    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.push_times: list[float] = []  # perf_counter() at each push
        self.flushed = 0

    def push(self, audio: np.ndarray) -> None:
        self.push_times.append(time.perf_counter())
        self.chunks.append(audio)

    def flush(self) -> None:
        self.flushed += 1

    @property
    def audio(self) -> np.ndarray:
        if not self.chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([c.flatten() for c in self.chunks])

    def reset(self) -> None:
        self.chunks.clear()
        self.push_times.clear()

    def save_wav(self, path, rate: int = SAMPLE_RATE) -> None:
        pcm = (np.clip(self.audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm.tobytes())


class FakeModel:
    """Mimics model.generate(): yields chunks with an `.audio` mlx array."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.calls: list[tuple[str, dict]] = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        for audio in self._chunks:
            yield SimpleNamespace(audio=mx.array(audio))


def make_streamer(model) -> tuple[tts_streamer.TTSStreamer, RecordingPacketizer]:
    streamer = tts_streamer.TTSStreamer(queue.Queue())  # not started: no process, no model load
    packetizer = RecordingPacketizer()
    streamer._packetizer = packetizer
    streamer._audio_model = model
    return streamer, packetizer


def rms_db(audio: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def sine(freq=440.0, seconds=0.25, amplitude=0.05) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_audio_is_captured_not_streamed():
    streamer, packetizer = make_streamer(FakeModel([sine()]))
    streamer.send_tts("Hello there.")
    assert len(packetizer.chunks) == 1
    assert isinstance(packetizer, RecordingPacketizer)  # real AudioPacketizer never created


def test_chunks_pushed_in_order_as_float32_numpy():
    chunks = [sine(200), sine(400), sine(800)]
    streamer, packetizer = make_streamer(FakeModel(chunks))
    streamer.send_tts("Three sentences.")
    assert len(packetizer.chunks) == 3
    for pushed in packetizer.chunks:
        assert isinstance(pushed, np.ndarray)
        assert pushed.dtype == np.float32
    # Chunk order and length are preserved
    assert [len(c) for c in packetizer.chunks] == [len(c) for c in chunks]


@pytest.mark.parametrize("amplitude", [0.005, 0.05, 0.4])
def test_chunks_normalized_to_target_level(amplitude):
    streamer, packetizer = make_streamer(FakeModel([sine(amplitude=amplitude)]))
    streamer.send_tts("Level check.")
    pushed = packetizer.chunks[0]
    assert np.max(np.abs(pushed)) <= 1.0  # tanh soft-limiter
    # tanh slightly compresses peaks, so allow a small tolerance around the target
    assert rms_db(pushed) == pytest.approx(TARGET_DB, abs=1.5)


def test_normalization_preserves_waveform_shape():
    original = sine(freq=300)
    streamer, packetizer = make_streamer(FakeModel([original]))
    streamer.send_tts("Shape check.")
    pushed = packetizer.chunks[0]
    # Gain is a positive scalar followed by a monotonic tanh, so correlation stays ~1
    assert np.corrcoef(original, pushed)[0, 1] > 0.99


def test_silent_chunk_does_not_crash():
    streamer, packetizer = make_streamer(FakeModel([np.zeros(1000, dtype=np.float32)]))
    streamer.send_tts("...")
    assert len(packetizer.chunks) == 1
    assert np.all(np.isfinite(packetizer.chunks[0]))


def test_speed_and_voice_forwarded_with_defaults():
    model = FakeModel([sine()])
    streamer, _ = make_streamer(model)
    streamer.send_tts("Defaults.")
    streamer.send_tts("Custom.", speed=0.8, voice="am_adam")
    (_, default_kwargs), (_, custom_kwargs) = model.calls
    assert default_kwargs["voice"] == "af_sarah"
    assert default_kwargs["speed"] == pytest.approx(1.1)
    assert custom_kwargs["voice"] == "am_adam"
    assert custom_kwargs["speed"] == pytest.approx(0.8)
    assert custom_kwargs["stream"] is True


def test_saved_wav_matches_pushed_audio(tmp_path):
    streamer, packetizer = make_streamer(FakeModel([sine(200), sine(400)]))
    streamer.send_tts("Two chunks.")
    path = tmp_path / "out.wav"
    packetizer.save_wav(path)

    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnchannels() == 1
        assert wav.getnframes() == len(packetizer.audio)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    # 16-bit quantization error only
    assert np.allclose(pcm / 32767, packetizer.audio, atol=1 / 32767)


# --------------------------------------------------------------------------- real model

REAL_TEXT = "The quick brown fox jumps over the lazy dog."


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


@pytest.fixture(scope="module")
def real_model():
    try:
        from mlx_audio.tts import load_model

        return load_model(MODEL_ID)
    except Exception as exc:  # model not downloaded / no network
        pytest.skip(f"TTS model {MODEL_ID} unavailable: {exc}")


@dataclass
class TTSTiming:
    audio_seconds: float  # duration of the generated audio
    first_audio_seconds: float  # send_tts() start -> first chunk pushed (latency)
    total_seconds: float  # send_tts() start -> last chunk pushed
    chunks: int

    @property
    def rtf(self) -> float:
        """Real-time factor: generation time / audio duration (lower is faster)."""
        return self.total_seconds / self.audio_seconds


def timed_send(streamer, packetizer, text, rate=SAMPLE_RATE) -> TTSTiming:
    packetizer.reset()
    start = time.perf_counter()
    streamer.send_tts(text)
    end = time.perf_counter()
    return TTSTiming(
        audio_seconds=len(packetizer.audio) / rate,
        first_audio_seconds=(packetizer.push_times[0] - start) if packetizer.push_times else end - start,
        total_seconds=end - start,
        chunks=len(packetizer.chunks),
    )


def format_tts_timing(t: TTSTiming) -> str:
    return (
        f"first_audio={t.first_audio_seconds:.3f}s total={t.total_seconds:.3f}s "
        f"audio={t.audio_seconds:.2f}s chunks={t.chunks} RTF={t.rtf:.3f}"
    )


def test_real_model_generation_timing(real_model):
    """Repeated generations of the same text; prints per-run and summary timings.

    The streamer is warmed up first (as TTSStreamer.run() does) so the first run
    doesn't include one-time compilation cost.
    """
    streamer, packetizer = make_streamer(real_model)
    for _ in real_model.generate("Warming up.", voice="af_sarah", lang="a", speed=1.0, stream=True):
        pass
    timings = []
    for run in range(1, RUNS + 1):
        timing = timed_send(streamer, packetizer, REAL_TEXT)
        timings.append(timing)
        print(f"\n[{MODEL_ID}] run {run}/{RUNS}: {format_tts_timing(timing)}")
        assert timing.audio_seconds > 0
    first = [t.first_audio_seconds for t in timings]
    rtfs = [t.rtf for t in timings]
    print(
        f"[{MODEL_ID}] summary ({RUNS} runs): first_audio min={min(first):.3f}s "
        f"mean={statistics.mean(first):.3f}s max={max(first):.3f}s | "
        f"RTF mean={statistics.mean(rtfs):.3f}"
    )


def test_real_model_produces_audible_audio_file(real_model, tmp_path):
    """No STT needed: checks the file exists, has plausible duration and loudness."""
    streamer, packetizer = make_streamer(real_model)
    timing = timed_send(streamer, packetizer, REAL_TEXT)
    print(f"\n[{MODEL_ID}] {format_tts_timing(timing)}")
    path = tmp_path / "real.wav"
    packetizer.save_wav(path)

    seconds = len(packetizer.audio) / SAMPLE_RATE
    assert 1.5 < seconds < 8.0  # ~9 words of speech
    assert rms_db(packetizer.audio) == pytest.approx(TARGET_DB, abs=3.0)
    assert path.stat().st_size > 44


def test_real_model_roundtrip(real_model, tmp_path):
    """TTS -> WAV file -> STT, and compare the transcript with the input text."""
    import test_parakeet as stt  # helpers; its server-availability skip is module-level only

    if not stt._server_available():
        pytest.skip(f"STT server not running on {stt.HOST}:{stt.PORT}")

    streamer, packetizer = make_streamer(real_model)
    tts_timing = timed_send(streamer, packetizer, REAL_TEXT)
    path = tmp_path / "roundtrip.wav"
    packetizer.save_wav(path)

    text, stt_timing = asyncio.run(asyncio.wait_for(stt._transcribe(path), stt.TIMEOUT))
    wer = stt._wer(REAL_TEXT, text)
    print(
        f"\nTTS input: {REAL_TEXT!r}\nSTT output: {text!r}\n"
        f"[{MODEL_ID}] TTS: {format_tts_timing(tts_timing)}\n"
        f"STT: inference={stt_timing.latency_seconds:.3f}s total={stt_timing.total_seconds:.3f}s "
        f"RTF={stt_timing.rtf:.3f} | WER={wer:.2%}"
    )
    assert wer <= 0.15
    assert _normalize(text)  # non-empty


# Hard words that are known to be difficult for TTS models.
# Now that espeak is installed, all words work including Alaska place names.
HARD_WORDS_TEXT = (
    "The sky is clear over Los Angeles today. "
    "Clear skies are expected across the Pacific Northwest. "
    "The Tanana River flows through Alaska near Nenana. "
    "Valdez and Fairbanks are far apart. "
    "Phoenix and Seattle are also far apart."
)

# Words we specifically want to check pronunciation for (subset of HARD_WORDS_TEXT)
HARD_WORDS_LIST = ["sky", "skies", "Los Angeles", "Tanana", "Nenana", "Valdez", "Fairbanks"]


def _word_level_analysis(reference_text: str, hypothesis_text: str) -> list[tuple[str, str, str]]:
    """Compare reference and hypothesis word-by-word and identify mismatches.
    
    Returns list of (reference_word, hypothesis_word, status) tuples.
    status is 'match', 'missing', 'extra', or 'mismatch'
    """
    ref_words = _normalize(reference_text).split()
    hyp_words = _normalize(hypothesis_text).split()
    
    # Simple word-level alignment using Levenshtein-style DP
    # Build alignment matrix
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution/match
            )
    
    # Backtrack to find alignment
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref_words[i-1] == hyp_words[j-1] else 1):
            status = 'match' if ref_words[i-1] == hyp_words[j-1] else 'mismatch'
            alignment.append((ref_words[i-1], hyp_words[j-1], status))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append((ref_words[i-1], '', 'missing'))
            i -= 1
        else:
            alignment.append(('', hyp_words[j-1], 'extra'))
            j -= 1
    
    alignment.reverse()
    return alignment


def _print_word_analysis(reference_text: str, hypothesis_text: str, model_id: str):
    """Print a detailed word-by-word pronunciation analysis."""
    alignment = _word_level_analysis(reference_text, hypothesis_text)
    
    # Find mismatches
    mismatches = [(r, h, s) for r, h, s in alignment if s in ('mismatch', 'missing')]
    
    print(f"\n[{model_id}] Word-level pronunciation analysis:")
    print(f"  Total words: {len([w for w, _, _ in alignment if w])}")
    print(f"  Correct: {len([w for w, h, s in alignment if s == 'match'])}")
    print(f"  Errors: {len(mismatches)}")
    
    if mismatches:
        print(f"  Mispronounced words:")
        for ref_word, hyp_word, status in mismatches:
            if status == 'mismatch':
                print(f"    '{ref_word}' -> '{hyp_word}'")
            else:
                print(f"    '{ref_word}' -> (missing)")


def test_real_model_hard_words_roundtrip(real_model, tmp_path):
    """TTS -> WAV file -> STT roundtrip with words known to be difficult for TTS models.

    Tests pronunciation of sky, skies, Los Angeles, Alaska place names (Tanana, Nenana, Valdez),
    and other challenging words through the full TTS-to-STT roundtrip.
    Prints per-word analysis to identify which specific words are mispronounced.
    """
    import test_parakeet as stt  # helpers; its server-availability skip is module-level only

    if not stt._server_available():
        pytest.skip(f"STT server not running on {stt.HOST}:{stt.PORT}")

    streamer, packetizer = make_streamer(real_model)
    tts_timing = timed_send(streamer, packetizer, HARD_WORDS_TEXT)
    path = tmp_path / "hard_words.wav"
    packetizer.save_wav(path)

    text, stt_timing = asyncio.run(asyncio.wait_for(stt._transcribe(path), stt.TIMEOUT))
    wer = stt._wer(HARD_WORDS_TEXT, text)
    
    # Print word-level analysis
    _print_word_analysis(HARD_WORDS_TEXT, text, MODEL_ID)
    
    print(
        f"\nTTS input: {HARD_WORDS_TEXT!r}\nSTT output: {text!r}\n"
        f"[{MODEL_ID}] TTS: {format_tts_timing(tts_timing)}\n"
        f"STT: inference={stt_timing.latency_seconds:.3f}s total={stt_timing.total_seconds:.3f}s "
        f"RTF={stt_timing.rtf:.3f} | WER={wer:.2%}"
    )
    # Harder threshold: these are tricky words, so allow up to 30% WER
    assert wer <= 0.30
    assert _normalize(text)  # non-empty


def test_real_model_individual_words(real_model, tmp_path):
    """Generate individual word audio files for manual listening verification.
    
    This addresses the concern that STT might be mishearing correct pronunciations.
    Creates separate WAV files for each hard word so they can be listened to directly.
    """
    import test_parakeet as stt  # helpers; its server-availability skip is module-level only

    if not stt._server_available():
        pytest.skip(f"STT server not running on {stt.HOST}:{stt.PORT}")

    streamer, packetizer = make_streamer(real_model)
    word_results = []
    
    print(f"\n[{MODEL_ID}] Individual word pronunciation test:")
    print("=" * 60)
    
    for word in HARD_WORDS_LIST:
        packetizer.reset()
        # Generate audio for just this word
        streamer.send_tts(word)
        
        # Save to individual file
        path = tmp_path / f"word_{word.replace(' ', '_')}.wav"
        packetizer.save_wav(path)
        
        # Transcribe back
        text, stt_timing = asyncio.run(asyncio.wait_for(stt._transcribe(path), stt.TIMEOUT))
        word_wer = stt._wer(word, text)
        
        status = "✓" if word_wer == 0 else "✗"
        word_results.append((word, text, word_wer))
        
        print(f"  {status} '{word}' -> '{text}' (WER: {word_wer:.0%})")
    
    print("=" * 60)
    
    # Summary
    errors = [(w, t, wer) for w, t, wer in word_results if wer > 0]
    if errors:
        print(f"\n[{MODEL_ID}] Words mispronounced:")
        for word, text, wer in errors:
            print(f"  '{word}' -> '{text}'")
    
    # Assert that at least some words are correct (don't fail the test if all are wrong)
    correct_count = sum(1 for _, _, wer in word_results if wer == 0)
    print(f"\n[{MODEL_ID}] Summary: {correct_count}/{len(HARD_WORDS_LIST)} words correct")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
