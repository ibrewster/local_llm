import asyncio
import logging
import multiprocessing
import os
import resource
import threading
import time

from functools import partial

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

import MLXOllama # Sets HF_HOME config and initializes logging

URI = "tcp://0.0.0.0:10300"
MODEL = "mlx-community/parakeet-tdt-0.6b-v2"
#MODEL = "mlx-community/phonon-2"
#MODEL = "littlebearlabs/parakeet-unified-en-0.6b-mlx-int8"  # Not yet supported by mlx_audio
# Allows comparing models without editing this file (inherited by the spawned process)
MODEL = os.environ.get("WYOMING_STT_MODEL", MODEL)
LANGUAGE = "en"
SAMPLE_RATE = 16000
WARMUP_SECONDS = 3


def _log_memory(label: str):
    """Logs MLX (GPU/unified) and process memory, in MB."""
    import mlx.core as mx

    mb = 1024 ** 2
    # ru_maxrss is bytes on macOS
    logging.info(
        f"Memory [{label}]: mlx_active={mx.get_active_memory() / mb:.0f}MB "
        f"mlx_peak={mx.get_peak_memory() / mb:.0f}MB "
        f"mlx_cache={mx.get_cache_memory() / mb:.0f}MB "
        f"process_peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / mb:.0f}MB"
    )


class SttEventHandler(AsyncEventHandler):
    """Buffers audio from a Wyoming client and transcribes it with mlx-audio on AudioStop."""

    def __init__(self, wyoming_info: Info, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.audio = bytes()
        self.audio_converter = AudioChunkConverter(rate=SAMPLE_RATE, width=2, channels=1)

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = self.audio_converter.convert(AudioChunk.from_event(event))
            self.audio += chunk.audio
            return True

        if AudioStop.is_type(event.type):
            text = ""
            try:
                text = self._transcribe()
            except Exception:
                logging.exception("Error during transcription")
            finally:
                self.audio = bytes()

            logging.info(f"Transcription: {text}")
            await self.write_event(Transcript(text=text).event())
            return False

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        if Transcribe.is_type(event.type):
            return True

        return True

    def _transcribe(self) -> str:
        import mlx.core as mx

        if not self.audio:
            return ""

        samples = np.frombuffer(self.audio, dtype=np.int16).astype(np.float32) / 32768.0

        start = time.monotonic()
        result = self.model.generate(mx.array(samples))
        mx.clear_cache()
        logging.debug(f"Transcription took {time.monotonic() - start:.3f}s")
        _log_memory("after transcription")

        return result.text.strip()


def _warm_up(model):
    """Runs one throwaway transcription so the first real request doesn't pay for
    kernel compilation / lazy initialization (phonon-2 took ~1.7s cold vs ~0.04s warm).
    """
    import mlx.core as mx

    # A few seconds of quiet noise exercises the full feature-extraction + decode path
    noise = np.random.default_rng(0).standard_normal(WARMUP_SECONDS * SAMPLE_RATE) * 0.01
    start = time.monotonic()
    try:
        model.generate(mx.array(noise.astype(np.float32)))
        mx.clear_cache()
    except Exception:
        logging.exception("Warm-up transcription failed (continuing without it)")
        return
    logging.info(f"Warm-up transcription took {time.monotonic() - start:.3f}s")


async def _serve():
    from mlx_audio.stt import load

    logging.info(f"Loading STT model: {MODEL}")
    model = load(MODEL)
    logging.info("STT model loaded")
    _warm_up(model)
    _log_memory("after warm-up")

    info = Info(
        asr=[
            AsrProgram(
                name="mlx-audio-stt",
                description="Speech-to-text via mlx-audio on Apple Silicon",
                attribution=Attribution(name="Blaizzy", url="https://github.com/Blaizzy/mlx-audio"),
                installed=True,
                version="0.1.0",
                models=[
                    AsrModel(
                        name=MODEL,
                        description=f"mlx-audio STT ({MODEL})",
                        attribution=Attribution(name="MLX Community", url=f"https://huggingface.co/{MODEL}"),
                        installed=True,
                        languages=[LANGUAGE],
                        version="0.1.0",
                    )
                ],
            )
        ],
    )

    server = AsyncServer.from_uri(URI)
    logging.info(f"Wyoming STT listening on {URI}")
    await server.run(partial(SttEventHandler, info, model))


def _run_wyoming_stt():
    import setproctitle

    setproctitle.setproctitle("Hermes Listener")
    threading.current_thread().name = 'HERMES_LISTENER_MAIN'

    asyncio.run(_serve())


def start_wyoming_stt_process():
    spawn_context = multiprocessing.get_context('spawn')
    stt_process = spawn_context.Process(
        target=_run_wyoming_stt,
        daemon=True,
        name="Hermes Listener"
    )
    stt_process.start()

    return stt_process


if __name__ == "__main__":
    # Standalone: python -m services.wyoming_stt  (run from the project root)
    try:
        _run_wyoming_stt()
    except KeyboardInterrupt:
        pass
