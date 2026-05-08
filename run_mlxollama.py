#!/Users/israel/Development/tts_pipeline/llm_env/bin/python -u
import setproctitle
setproctitle.setproctitle("HermesMLX")

import asyncio
import logging
import multiprocessing
import signal
import os

from hypercorn.config import Config
from hypercorn.asyncio import serve

import MLXOllama # Sets the HF_HOME config
from MLXOllama import tts_streamer

from whisper import run_whisper_process

def _sigterm_handler(signum, frame):
    os.kill(os.getpid(), signal.SIGINT)
    
signal.signal(signal.SIGTERM, _sigterm_handler)

if __name__ == "__main__":    
    logging.info("Starting Whisper Server")
    spawn_context = multiprocessing.get_context('spawn')
    _whisper_process = spawn_context.Process(
        target=run_whisper_process,
        daemon=True,
        name="wyoming-mlx-whisper"
    )
    _whisper_process.start()
    logging.info("Whisper process PID: %d", _whisper_process.pid)
    
    logging.info("Starting TTS process")
    MLXOllama.tts_queue = multiprocessing.Queue()
    tts_process = tts_streamer.TTSStreamer(MLXOllama.tts_queue)
    tts_process.start()    
    
    hconfig = Config()
    hconfig.bind = ["0.0.0.0:11434"]
    hconfig.accesslog = logging.getLogger()   # pass root logger directly
    hconfig.errorlog = logging.getLogger()
    
    MLXOllama.setup_app()

    asyncio.run(serve(MLXOllama.app, hconfig))    

    MLXOllama.app.logger.info("Server shutting down")
    MLXOllama.speak_queue.put("QUIT")
    MLXOllama.process_thread.join()
    
    if _whisper_process and _whisper_process.is_alive():
        MLXOllama.app.logger.info("Shutting down Whisper Process...")        
        _whisper_process.terminate()
        _whisper_process.join(timeout=5)
        if _whisper_process.is_alive():
            _whisper_process.kill()
        MLXOllama.app.logger.info("Whisper process stopped")
        
    if tts_process and tts_process.is_alive():
        MLXOllama.app.logger.info("Shutting down Speaker process")
        tts_process.terminate()
        tts_process.join(timeout=5)
        if tts_process.is_alive():
            tts_process.kill()
        MLXOllama.app.logger.info("Speaker process stopped")
    
    MLXOllama.utils.inference_worker.shutdown()
        
    MLXOllama.app.logger.info("MLXOllama has exited")