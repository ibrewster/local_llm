import logging
import multiprocessing
import time

import setproctitle

import numpy

import mlx.core as mx
from mlx_audio.tts import load_model

from . import audio_stream

spawn_context = multiprocessing.get_context('spawn')
class TTSStreamer(spawn_context.Process):
    def __init__(self, queue:multiprocessing.Queue):
        super().__init__(daemon=True, name="Hermes Speaker")

        self.TARGET_GAIN = 3.0
        self._queue = queue
        self._msgs_streamed = 0
        self._packetizer = None
        self._audio_model = None

    def run(self):
        setproctitle.setproctitle("Hermes Speaker")
        self._packetizer = audio_stream.AudioPacketizer()
        logging.info("Loading Audio model")
        self._audio_model = load_model("mlx-community/Kokoro-82M-bf16")
        # self._audio_model = load_model("mlx-community/orpheus-3b-0.1-ft-6bit")
        # self._audio_model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit")
        
        dummy = "Warming up."
        for _ in self._audio_model.generate(dummy, voice='af_sarah', lang="a", speed=1.0, stream=True):
            pass  # consume and discard
        
        logging.info("TTS Streamer process running")
        try:
            while True:
                msg = self._queue.get()
                if msg == "__FLUSH__":
                    self._packetizer.flush()
                    self._msgs_streamed = 0
                    continue

                if not isinstance(msg, (tuple, list)):
                    msg = (msg, )

                self.send_tts(*msg)
                self._msgs_streamed += 1


        except KeyboardInterrupt:
            logging.info("TTS Streamer process exiting")

    def send_tts(self, data, speed=1.1, voice="af_sarah"):
        replacements = {
            # Sky / skies fixes (most important for you)
            "skies": "skize",
            "Skies": "Skize",
            "sky": "skai",
            "Sky": "Skai",
            "Tanana": "tan-uh-nuh",
            "tanana": "tan-uh-nuh",
            "Nenana": "knee-nann-uh",
            # ",": "-",
        }

        for old, new in replacements.items():
            data = data.replace(old, new)

        t0 = time.time()
        for chunk in self._audio_model.generate(data, voice=voice, lang="a", speed=speed, stream=True):
            logging.info(f"Streamed sentence after {time.time() - t0} (length: {len(data)})")
            audio = chunk.audio
            rms = mx.sqrt(mx.mean(audio**2))
            curr_db = 20 * numpy.log10(rms) if rms > 0 else -100
            gain = 10**((-12 - curr_db) / 20)
            normalized_mx = mx.tanh(audio * gain)
            # normalized_mx = mx.tanh(chunk.audio * self.TARGET_GAIN)
            audio_np = numpy.array(normalized_mx).astype(numpy.float32)
            ####### DEBUG REMOVE #######
            # logging.info(data)
            # continue
            ##################
            self._packetizer.push(audio_np)
