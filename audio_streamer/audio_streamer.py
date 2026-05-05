#!/Users/israel/Development/tts_pipeline/mr_env/bin/python -u

import logging
import os
import socket
import signal
import sys

from multiprocessing import Event

import setproctitle

import numpy as np
import sounddevice as sd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/var/log/tts/server.log", mode="a"),
        logging.StreamHandler(sys.stderr)
    ]
)

def _sigterm_handler(signum, frame):
    os.kill(os.getpid(), signal.SIGINT)
    
signal.signal(signal.SIGTERM, _sigterm_handler)

def stream_audio(exit_event: Event):
    setproctitle.setproctitle("Hermes Streamer")
    logging.info("Audio streamer starting") 
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
    
    def callback(indata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
            
        # indata is float32 by default → convert to int16
        #audio_int16 = (indata * 32767).astype(np.int16, copy=False)
        
        # send raw bytes
        try:
            #sock.sendto(audio_int16.tobytes(), ("239.0.0.1", 5004))
            sock.sendto(indata.tobytes(), ("239.0.0.1", 5004))
        except OSError as e:
            logging.warning(f"Error when writing to socket: {e}")
    
    try:
        with sd.InputStream(
            samplerate=48000,
            channels=2,
            device="BlackHole 2ch",
            dtype='int16',
            callback=callback,
            blocksize=2048     # important: controls latency/packet size
        ):
            exit_event.wait()
            
    except Exception:
        logging.exception("Exception occured while streaming audio:")
    except KeyboardInterrupt:
        logging.info("Exiting due to keyboard interupt")

    logging.info("Audio streaming ended")
    
if __name__ == "__main__":
    EXIT_EVENT = Event()    
    stream_audio(EXIT_EVENT) # Will block forever