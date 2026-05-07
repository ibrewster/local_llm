import logging
import multiprocessing
import os
import threading
import queue

threading.current_thread().name = "MLX_MAIN"

from . import config
os.environ["HF_HOME"] = config.HF_HOME
os.environ["HF_TOKEN"] = config.HF_TOKEN

from . import utils
utils.init_logging()

from quart import Quart
from quart.logging import default_handler

app = None
speak_queue = None
process_thread = None
tts_queue: multiprocessing.Queue = None

def setup_app():    
    global app, speak_queue, process_thread, tts_queue, tts_process
    
    for name, path in config.models.items():
        future = utils.inference_worker.submit(utils.load_model, name, path)
        future.result() #  Wait for the result
        # utils.load_model(name, path)
    
    logging.info('Models loaded')
    
    # Start the prompt processing thread
    from .worker import speak_thread
    speak_queue = queue.Queue()
    
    process_thread = threading.Thread(target = speak_thread, name="TTS_Worker", args=(speak_queue, ))
    process_thread.start()
    
    app = Quart(__name__)
    logging.getLogger(app.name).removeHandler(default_handler)
    app.logger.propagate = True
    
    from . import main
