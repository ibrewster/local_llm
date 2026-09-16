#!/Users/israel/Development/tts_pipeline/llm_env/bin/python -u
import setproctitle
import sys

setproctitle.setproctitle("HermesMLX")

import asyncio
import logging
import multiprocessing
import os
import signal

from pathlib import Path

from hypercorn.config import Config
from hypercorn.asyncio import serve

import MLXOllama # Sets the HF_HOME config
from MLXOllama import tts_streamer

def _sigterm_handler(_signum, _frame):
    os.kill(os.getpid(), signal.SIGINT)
    
signal.signal(signal.SIGTERM, _sigterm_handler)


def start_parakeet_process():
    import setproctitle
    import threading

    setproctitle.setproctitle("wyoming-mlx-parakeet")
    threading.current_thread().name = 'PARAKEET_MAIN'

    parakeet_path=Path(__file__).parent / "wyoming-parakeet-mlx"

    if str(parakeet_path) not in sys.path:
        sys.path.insert(0, str(parakeet_path))

    URI = "tcp://0.0.0.0:10300"
    MODEL = "mlx-community/parakeet-tdt-0.6b-v2"

    logging.info(f"Starting Parakeet process from {parakeet_path}")
    sys.argv = [
        "wyoming-parakeet-mlx",
        "--uri", URI,
        "--model", MODEL,
        "--debug",  # uncomment if you want verbose logs
    ]

    # Must be imported after sys.path is modified, otherwise it will fail to find the package
    from wyoming_parakeet_mlx.__main__ import run
    run()

if __name__ == "__main__":    
    logging.info("Starting Parakeet Server")
    spawn_context = multiprocessing.get_context('spawn')
    # _whisper_process = spawn_context.Process(
    #     target=run_whisper_process,
    #     daemon=True,
    #     name="wyoming-mlx-whisper"
    # )
    # _whisper_process.start()

    parakeet_process=spawn_context.Process(
        target=start_parakeet_process,
        daemon=True,
        name="wyoming-mlx-parakeet"
    )
    parakeet_process.start()

    logging.info("Parakeet process PID: %d", parakeet_process.pid)
    
    logging.info("Starting TTS process")
    MLXOllama.tts_queue = multiprocessing.Queue()
    tts_process = tts_streamer.TTSStreamer(MLXOllama.tts_queue)
    tts_process.start()    
    
    hconfig = Config()
    hconfig.bind = ["0.0.0.0:11434"]
    hconfig.accesslog = logging.getLogger()   # pass root logger directly
    hconfig.errorlog = logging.getLogger()
    
    MLXOllama.setup_app() # Sets up speak_queue and process_thread

    if MLXOllama.app is None:
        raise RuntimeError("APP initilization failed. MLXOllama.app is None")

    asyncio.run(serve(MLXOllama.app, hconfig))    

    MLXOllama.app.logger.info("Server shutting down")
    MLXOllama.speak_queue.put("QUIT")
    MLXOllama.process_thread.join()
    
    if parakeet_process and parakeet_process.is_alive():
        MLXOllama.app.logger.info("Shutting down Parakeet Process...")
        parakeet_process.terminate()
        parakeet_process.join(timeout=5)
        if parakeet_process.is_alive():
            logging.warning("Parakeet process did not terminate within 5 seconds. Killing...")
            parakeet_process.kill()
        MLXOllama.app.logger.info("Parakeet process stopped")
        
    if tts_process and tts_process.is_alive():
        MLXOllama.app.logger.info("Shutting down Speaker process")
        tts_process.terminate()
        tts_process.join(timeout=5)
        if tts_process.is_alive():
            tts_process.kill()
        MLXOllama.app.logger.info("Speaker process stopped")
    
    MLXOllama.utils.inference_worker.shutdown()
        
    MLXOllama.app.logger.info("MLXOllama has exited")