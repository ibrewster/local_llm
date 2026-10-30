import asyncio
import logging
import multiprocessing
import re
import resource
import sys
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
from MLXOllama.config import STT_MODEL

URI = "tcp://0.0.0.0:10300"
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

        text = result.text.strip()
        words = text.split()
        if len(words) > 10 and len(set(words)) / len(words) < 0.1:
            logging.warning("Repetition loop detected, suppressing: %s", text[:80])
            return ""

        return text


def _ensure_whisper_processor(model, model_name: str):
    """mlx-community Whisper repos ship only weights + config, so mlx-audio can't build
    its tokenizer and every transcription fails. Borrow it from the original openai repo.
    """
    if not hasattr(model, "_processor") or model._processor is not None:
        return
    from transformers import WhisperProcessor

    # e.g. "whisper-small.en-mlx" -> "whisper-small.en"
    base_name = re.sub(r"[-_](mlx|\d+bit|q\d+|fp16|4bit|8bit)$", "", model_name.split('/')[-1])
    source = f"openai/{base_name}"
    logging.info(f"Model has no tokenizer; loading WhisperProcessor from {source}")
    model._processor = WhisperProcessor.from_pretrained(source)


def _fix_english_only_whisper_no_speech():
    """English-only Whisper models (*.en) have no <|nospeech|> token (they use <|nocaptions|>),
    so mlx-audio's lookup falls back to <|endoftext|>. That id then lands in the suppress list,
    which bans the end-of-text token and makes generation run on past the end of speech
    (hallucinated trailing text). Mirror openai/whisper: prefer <|nospeech|>, else <|nocaptions|>.
    """
    from mlx_audio.stt.models.whisper.whisper import HFTokenizerWrapper

    def no_speech(self):
        vocab = self.hf_tokenizer.get_vocab()
        for name in ("<|nospeech|>", "<|nocaptions|>"):
            if name in vocab:
                return vocab[name]
        return None

    HFTokenizerWrapper.no_speech = property(no_speech)


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


async def _serve(model_name: str):
    from mlx_audio.stt import load

    logging.info(f"Loading STT model: {model_name}")
    model = load(model_name)
    logging.info("STT model loaded")
    _ensure_whisper_processor(model, model_name)
    if hasattr(model, "_processor"):
        _fix_english_only_whisper_no_speech()
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
                        name=model_name,
                        description=f"mlx-audio STT ({model_name})",
                        attribution=Attribution(name="MLX Community", url=f"https://huggingface.co/{model_name}"),
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


def _run_wyoming_stt(model_name: str):
    import setproctitle

    setproctitle.setproctitle("Hermes Listener")
    threading.current_thread().name = 'HERMES_LISTENER_MAIN'

    asyncio.run(_serve(model_name))


def start_wyoming_stt_process(model: str | None = None):
    """Starts the STT server in a child process. `model` defaults to config.STT_MODEL."""
    spawn_context = multiprocessing.get_context('spawn')
    stt_process = spawn_context.Process(
        target=_run_wyoming_stt,
        args=(model or STT_MODEL,),
        daemon=True,
        name="Hermes Listener"
    )
    stt_process.start()

    return stt_process


if __name__ == "__main__":
    # Standalone: python -m services.wyoming_stt [model]  (run from the project root)
    try:
        _run_wyoming_stt(sys.argv[1] if len(sys.argv) > 1 else STT_MODEL)
    except KeyboardInterrupt:
        pass
