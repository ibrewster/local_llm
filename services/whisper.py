import asyncio
import logging
import time

from typing import Any

import numpy as np
import mlx_whisper

from wyoming_mlx_whisper.handler import WhisperEventHandler, _LOGGER, NDArray

class PatchedWhisperEventHandler(WhisperEventHandler):
    
    def _transcribe(self, audio: NDArray[np.float32]) -> str:
        start_time = time.time()
        kwargs: dict[str, Any] = {
            "path_or_hf_repo": self._model,
            # Hallucination/loop prevention
            "compression_ratio_threshold": 1.8,
            "condition_on_previous_text": False,
        }
        if self._language:
            kwargs["language"] = self._language
        if self._initial_prompt:
            kwargs["initial_prompt"] = self._initial_prompt
        
        result = mlx_whisper.transcribe(audio, **kwargs)
        elapsed = time.time() - start_time
        _LOGGER.debug("Transcription completed in %.2f seconds", elapsed)
        
        text = str(result["text"])
        
        # Safety net: catch loops that slip through anyway
        words = text.split()
        if len(words) > 10:
            if len(set(words)) / len(words) < 0.1:
                _LOGGER.warning("Repetition loop detected, suppressing: %s", text[:80])
                return ""
            
        return text
    
def run_whisper_process():
    from wyoming.server import AsyncServer
    from wyoming_mlx_whisper.server import _create_wyoming_info
    from mlx_whisper.load_models import load_model
    import setproctitle
    setproctitle.setproctitle("wyoming-mlx-whisper")    
    
    WHISPER_URI = "tcp://0.0.0.0:10300"          # Pick a free port, separate from Ollama's 11434
    WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
    WHISPER_LANGUAGE = 'en'
    logging.info("Starting Wyoming MLX Whisper server")
    
    logging.info("Loading whisper model...")
    load_model(WHISPER_MODEL)
    logging.info("Whisper model loaded!")

    wyoming_info = _create_wyoming_info(WHISPER_MODEL)
    
    async def main():
        server = AsyncServer.from_uri(WHISPER_URI)
        await server.run(
            lambda *args, **kwargs: PatchedWhisperEventHandler(
                wyoming_info,
                WHISPER_MODEL,
                WHISPER_LANGUAGE,
                None,
                *args,
                **kwargs
            )
        )
        
    asyncio.run(main())