"""Integration test for the Wyoming ASR server (services/wyoming_stt.py, any model).

Requires the server to be running (default localhost:10300). The tests are skipped if
the port is not reachable. It streams a known recording
(tests/data/quick_brown_fox.wav) using the Wyoming protocol and verifies
that the transcript matches the spoken sentence.

Environment variables (all optional):
    STT_HOST          server host (default 127.0.0.1), e.g. 10.27.81.2 for a remote machine
    STT_PORT          server port (default 10300)
    STT_SSH           ssh target (e.g. user@10.27.81.2) used to measure the *remote* server's
                      memory; without it, memory is only reported for a local server
    STT_EXPECT_MODEL  substring the server must advertise, e.g. whisper-small.en-mlx
    PARAKEET_RUNS     number of repeated transcriptions in the timing test (default 3)

Regenerate the fixture (macOS) with:
    say -v Samantha -o /tmp/p.aiff "The quick brown fox jumps over the lazy dog."
    afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/p.aiff tests/data/quick_brown_fox.wav
"""
import asyncio
import os
import re
import shlex
import socket
import statistics
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

HOST = os.environ.get("STT_HOST", "127.0.0.1")
PORT = int(os.environ.get("STT_PORT", "10300"))
SSH_TARGET = os.environ.get("STT_SSH")
EXPECT_MODEL = os.environ.get("STT_EXPECT_MODEL")
IS_LOCAL = HOST in ("127.0.0.1", "localhost", "::1")
AUDIO_FILE = Path(__file__).parent / "data" / "quick_brown_fox.wav"
EXPECTED_TEXT = "The quick brown fox jumps over the lazy dog."
SAMPLES_PER_CHUNK = 1024
TIMEOUT = 60
# Number of transcription runs, for timing comparisons (first run may include warm-up)
RUNS = int(os.environ.get("PARAKEET_RUNS", "3"))


@dataclass
class Timing:
    audio_seconds: float  # duration of the audio sent
    send_seconds: float  # time to stream all audio to the server
    latency_seconds: float  # AudioStop sent -> Transcript received (pure inference)
    total_seconds: float  # connect -> Transcript received

    @property
    def rtf(self) -> float:
        """Real-time factor: processing time / audio duration (lower is faster)."""
        return self.total_seconds / self.audio_seconds


def _server_available() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(), reason=f"STT server not running on {HOST}:{PORT}"
)


def _run_on_server_host(cmd: list[str], timeout: int) -> str:
    """Runs a command on the machine hosting the server (locally, or via ssh for a remote one)."""
    if not IS_LOCAL:
        if not SSH_TARGET:
            raise OSError("remote server and STT_SSH not set")
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", SSH_TARGET, shlex.join(cmd)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


def _server_pid() -> int | None:
    """PID of the process listening on PORT (the model lives in that process)."""
    try:
        out = _run_on_server_host(["lsof", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"], 20).split()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _server_memory_mb() -> dict[str, float]:
    """Best-effort server memory in MB (macOS): resident set size and physical footprint.

    phys_footprint (from `footprint`) includes Metal/GPU allocations that RSS can miss,
    so it is the better number for comparing MLX models. Returns {} if unavailable.
    """
    pid = _server_pid()
    if pid is None:
        return {}
    mem: dict[str, float] = {}
    try:
        rss_kb = _run_on_server_host(["ps", "-o", "rss=", "-p", str(pid)], 20).strip()
        if rss_kb:
            mem["rss"] = int(rss_kb) / 1024
        out = _run_on_server_host(["footprint", "-p", str(pid)], 40)
        for key in ("phys_footprint", "phys_footprint_peak"):
            match = re.search(rf"{key}:\s+([\d.]+)\s*(KB|MB|GB)", out)
            if match:
                scale = {"KB": 1 / 1024, "MB": 1, "GB": 1024}[match.group(2)]
                mem[key] = float(match.group(1)) * scale
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return mem


def _format_memory(mem: dict[str, float]) -> str:
    if not mem:
        return "unavailable"
    return " ".join(f"{k}={v:.0f}MB" for k, v in mem.items())


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation so comparison ignores formatting."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _wer(reference: str, hypothesis: str) -> float:
    """Word error rate (word-level Levenshtein distance / reference length)."""
    ref, hyp = _normalize(reference).split(), _normalize(hypothesis).split()
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


# Difficult recordings, generated by tests/make_stt_fixtures.py (keep texts in sync).
# max_wer is the pass threshold; the actual WER is always printed for model comparison.
COMMAND_TEXT = (
    "Turn off the living room lights and set the thermostat to a cooler temperature "
    "before the guests arrive tonight."
)
PARAGRAPH_TEXT = (
    "Yesterday I walked to the market near the old train station, and bought three oranges, "
    "a loaf of bread, and some fresh basil. The vendor told me that the rain would "
    "probably arrive before noon, so I hurried home."
)
DIFFICULT_CASES = [
    # (file, expected text, max WER)
    ("fox_pink_noise_5db.wav", EXPECTED_TEXT, 0.25),
    ("command_white_noise_10db.wav", COMMAND_TEXT, 0.25),
    ("paragraph_clean.wav", PARAGRAPH_TEXT, 0.10),
    # Competing speaker at 5 dB SNR: measured 38% (parakeet v2) / 64% (phonon-2).
    # Threshold is a regression guard, not a quality bar; compare the printed WER.
    ("paragraph_babble_5db.wav", PARAGRAPH_TEXT, 0.75),
    ("command_fast_quiet_noisy.wav", COMMAND_TEXT, 0.40),
]


async def _transcribe(wav_path: Path) -> tuple[str, Timing]:
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        width = wav.getsampwidth()
        channels = wav.getnchannels()
        frames = wav.getnframes()
        audio = wav.readframes(frames)

    audio_seconds = frames / rate

    bytes_per_chunk = SAMPLES_PER_CHUNK * width * channels

    start = time.perf_counter()
    async with AsyncTcpClient(HOST, PORT) as client:
        await client.write_event(Transcribe().event())
        await client.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
        for i in range(0, len(audio), bytes_per_chunk):
            chunk = AudioChunk(
                rate=rate, width=width, channels=channels, audio=audio[i:i + bytes_per_chunk]
            )
            await client.write_event(chunk.event())
        await client.write_event(AudioStop().event())
        sent = time.perf_counter()

        while True:
            event = await client.read_event()
            if event is None:
                raise RuntimeError("Server closed connection without sending a transcript")
            if Transcript.is_type(event.type):
                done = time.perf_counter()
                timing = Timing(
                    audio_seconds=audio_seconds,
                    send_seconds=sent - start,
                    latency_seconds=done - sent,
                    total_seconds=done - start,
                )
                return Transcript.from_event(event).text, timing


async def _describe() -> Info:
    async with AsyncTcpClient(HOST, PORT) as client:
        await client.write_event(Describe().event())
        while True:
            event = await client.read_event()
            assert event is not None
            if Info.is_type(event.type):
                return Info.from_event(event)


@pytest.mark.skipif(not EXPECT_MODEL, reason="STT_EXPECT_MODEL not set")
def test_server_runs_expected_model():
    """Guards against benchmarking the wrong model (e.g. a stale server on the remote box)."""
    info = asyncio.run(asyncio.wait_for(_describe(), TIMEOUT))
    names = [m.name for program in info.asr for m in program.models]
    print(f"Server {HOST}:{PORT} advertises models: {names}")
    assert any(EXPECT_MODEL in name for name in names), f"{EXPECT_MODEL!r} not in {names}"


def test_describe_returns_info():
    async def run():
        async with AsyncTcpClient(HOST, PORT) as client:
            await client.write_event(Describe().event())
            while True:
                event = await client.read_event()
                assert event is not None
                if Info.is_type(event.type):
                    return Info.from_event(event)

    info = asyncio.run(asyncio.wait_for(run(), TIMEOUT))
    assert info.asr, "Server should advertise at least one ASR program"
    for program in info.asr:
        print(f"ASR program: {program.name}, models: {[m.name for m in program.models]}")


def test_transcription_matches_known_speech():
    assert AUDIO_FILE.exists(), f"Missing fixture {AUDIO_FILE}"
    memory_before = _server_memory_mb()
    timings = []
    for run in range(1, RUNS + 1):
        text, timing = asyncio.run(asyncio.wait_for(_transcribe(AUDIO_FILE), TIMEOUT))
        timings.append(timing)
        print(
            f"Run {run}/{RUNS}: transcript={text!r} | audio={timing.audio_seconds:.2f}s "
            f"send={timing.send_seconds:.3f}s inference={timing.latency_seconds:.3f}s "
            f"total={timing.total_seconds:.3f}s RTF={timing.rtf:.3f}"
        )
        assert _normalize(text) == _normalize(EXPECTED_TEXT)

    latencies = [t.latency_seconds for t in timings]
    totals = [t.total_seconds for t in timings]
    print(
        f"Timing summary ({RUNS} runs): inference min={min(latencies):.3f}s "
        f"mean={statistics.mean(latencies):.3f}s max={max(latencies):.3f}s | "
        f"total mean={statistics.mean(totals):.3f}s"
    )
    print(f"Server memory before runs: {_format_memory(memory_before)}")
    print(f"Server memory after runs:  {_format_memory(_server_memory_mb())}")


@pytest.mark.parametrize(
    "filename,expected,max_wer", DIFFICULT_CASES, ids=[c[0] for c in DIFFICULT_CASES]
)
def test_difficult_audio(filename, expected, max_wer):
    """Noisy / fast / quiet / long / overlapping-speech clips, scored by word error rate."""
    path = AUDIO_FILE.parent / filename
    assert path.exists(), f"Missing fixture {path}; run tests/make_stt_fixtures.py"
    text, timing = asyncio.run(asyncio.wait_for(_transcribe(path), TIMEOUT))
    wer = _wer(expected, text)
    print(
        f"\n{filename}: WER={wer:.2%} (max {max_wer:.0%}) inference={timing.latency_seconds:.3f}s "
        f"audio={timing.audio_seconds:.1f}s RTF={timing.rtf:.3f}\n  got: {text!r}"
    )
    assert wer <= max_wer


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
