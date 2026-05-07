import logging
import multiprocessing
import time

import setproctitle

import numpy

import mlx.core as mx
from mlx_audio.tts import load_model

from . import audio_stream

class TTSStreamer():
    def __init__(self, queue:multiprocessing.Queue):
        self.TARGET_GAIN = 3.0
        self._queue = queue
        self._packetizer = audio_stream.AudioPacketizer()
        logging.info("Loading Audio model")
        self._audio_model = load_model("mlx-community/Kokoro-82M-bf16")
        # self._audio_model = load_model("mlx-community/orpheus-3b-0.1-ft-6bit")
        # self._audio_model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit")
        self._msgs_streamed = 0

    def run(self):
        logging.info("TTS Streamer process running")
        try:
            while True:
                msg = self._queue.get()
                ###### DEBUG REMOVE ######
                # if not isinstance(msg, (tuple, list)):
                    # msg = (msg, )

                # try:
                    # logging.info(f"{msg[0]}")
                # except Exception as e:
                    # logging.info(str(msg))
                # continue
                ###########################
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
            logging.info(f"Streamed sentence after {time.time() - t0}")
            numpy_audio = numpy.array(chunk.audio)
            rms = numpy.sqrt(numpy.mean(numpy_audio**2))
            curr_db = 20 * numpy.log10(rms) if rms > 0 else -100
            gain = 10**((-12 - curr_db) / 20)
            audio_np = numpy.tanh(numpy_audio * gain)
            # normalized_mx = mx.tanh(chunk.audio * self.TARGET_GAIN)
            # audio_np = numpy.array(normalized_mx).astype(numpy.float32)
            self._packetizer.push(audio_np)


def run_tts_streamer(text_queue:multiprocessing.Queue):
    setproctitle.setproctitle("Hermes Speaker")
    streamer = TTSStreamer(text_queue)
    streamer.run()
