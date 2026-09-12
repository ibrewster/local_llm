#!/Users/israel/Development/tts_pipeline/llm_env/bin/python -u
import setproctitle
setproctitle.setproctitle("HermesMLX")

import asyncio
import logging
import multiprocessing
import os
import signal
import subprocess
import threading

from pathlib import Path

from hypercorn.config import Config
from hypercorn.asyncio import serve

import MLXOllama # Sets the HF_HOME config
from MLXOllama import tts_streamer

from whisper import run_whisper_process

def _sigterm_handler(_signum, _frame):
    os.kill(os.getpid(), signal.SIGINT)
    
signal.signal(signal.SIGTERM, _sigterm_handler)

if __name__ == "__main__":    
    logging.info("Starting Whisper Server")
    spawn_context = multiprocessing.get_context('spawn')
    # _whisper_process = spawn_context.Process(
    #     target=run_whisper_process,
    #     daemon=True,
    #     name="wyoming-mlx-whisper"
    # )
    # _whisper_process.start()

    parakeet_path=Path(__file__).parent / "wyoming-parakeet-mlx"
    parakeet_py = parakeet_path /".venv"/"bin"/"python"
    URI = "tcp://0.0.0.0:10300"
    MODEL = "mlx-community/parakeet-tdt-0.6b-v2"

    logging.info(f"Starting Parakeet process from {parakeet_py}")
    cmd = [
        str(parakeet_py),
        "-u",
        "-m", "wyoming_parakeet_mlx",
        "--uri", URI,
        "--model", MODEL,
        "--debug",          # uncomment if you want verbose logs
    ]

    parakeet_process = subprocess.Popen(
        cmd,
        cwd=str(parakeet_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        start_new_session=False,
    )

    logging.info("Parakeet process PID: %d", parakeet_process.pid)

    def log_parakeet_output():
        assert parakeet_process.stdout is not None

        for line in parakeet_process.stdout:
            logging.info("Parakeet: %s", line.rstrip())

    threading.Thread(
        target=log_parakeet_output,
        name="parakeet-logger",
        daemon=True,
    ).start()
    
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
    
    if parakeet_process and parakeet_process.poll() is None:
        MLXOllama.app.logger.info("Shutting down Parakeet Process...")
        parakeet_process.terminate()
        try:
            parakeet_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.warning("Parakeet process did not terminate within 5 seconds. Killing...")
            parakeet_process.kill()
            parakeet_process.wait()

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