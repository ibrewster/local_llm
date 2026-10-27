#!/Users/israel/Development/tts_pipeline/llm_env/bin/python -u
import setproctitle

setproctitle.setproctitle("HermesMLX")

import asyncio
import logging
import multiprocessing
import os
import signal

from hypercorn.config import Config
from hypercorn.asyncio import serve

import MLXOllama # Sets the HF_HOME config
from MLXOllama import tts_streamer

from services import wyoming_stt, litellm_runner

def _sigterm_handler(_signum, _frame):
    os.kill(os.getpid(), signal.SIGINT)
    
signal.signal(signal.SIGTERM, _sigterm_handler)


if __name__ == "__main__":    
    logging.info("Starting Wyoming STT Server")
    stt_process=wyoming_stt.start_wyoming_stt_process()
    logging.info("Wyoming STT process PID: %d", stt_process.pid)
    
    logging.info("Starting TTS process")
    MLXOllama.tts_queue = multiprocessing.Queue()
    tts_process = tts_streamer.TTSStreamer(MLXOllama.tts_queue)
    tts_process.start()

    # Creates a litellm proxy in front of our server
    # Usefull if we want to add additional protocols or authentication in the future
    # logging.info("Starting litellm proxy server")
    # litellm_process = litellm_runner.start_litellm_process()
    # logging.info("litellm process PID: %d", litellm_process.pid)
    
    hconfig = Config()
    hconfig.bind = ["0.0.0.0:11434"]
    hconfig.accesslog = logging.getLogger()   # pass root logger directly
    hconfig.errorlog = logging.getLogger()
    
    MLXOllama.setup_app() # Sets up speak_queue and process_thread

    if MLXOllama.app is None:
        raise RuntimeError("APP initilization failed. MLXOllama.app is None")

    asyncio.run(serve(MLXOllama.app, hconfig))

    ################ SHUTDOWN ################

    MLXOllama.app.logger.info("Server shutting down")
    MLXOllama.speak_queue.put("QUIT")
    MLXOllama.process_thread.join()
    
    for name, process in [
        ("Wyoming STT", stt_process),
        ("Speaker", tts_process),
#        ("litellm", litellm_process),
    ]:
        MLXOllama.app.logger.info(f"Shutting down {name} process...")
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                MLXOllama.app.logger.warning(f"{name} process did not terminate within 5 seconds. Killing...")
                process.kill()
            MLXOllama.app.logger.info(f"{name} process stopped")
        else:
            MLXOllama.app.logger.info(f"{name} process was not running")
    
    MLXOllama.utils.inference_worker.shutdown()
        
    MLXOllama.app.logger.info("MLXOllama has exited")