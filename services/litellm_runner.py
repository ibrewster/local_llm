import multiprocessing
import os
import threading

import yaml

from pathlib import Path

import MLXOllama # Sets HF_HOME config and initializes logging
from MLXOllama.config import MAIN_MODEL

FILE_DIR=Path(__file__).parent
def _run_litellm_worker():
    # Import inside the function to keep the parent Quart process clean
    import sys
    import setproctitle

    config_file= FILE_DIR / "litellm_config.yaml"
    # Mock sys.argv to simulate the command line arguments for Click
    sys.argv = ["litellm", "--config", config_file, "--port", "4000"]

    setproctitle.setproctitle("Hermes Translater")
    threading.current_thread().name = 'LITELLM_MAIN'

    from litellm import run_server
    # Run the click command
    run_server()

def start_litellm_process():
    config_file = FILE_DIR / "litellm_config.yaml"

    # Generate the YAML config before spawning the worker
    if not os.path.exists(config_file):
        config_data = {
            "general_settings": {
                "dangerously_permit_weak_or_unset_master_key": True
            },
            "model_list": [{
                "model_name": MAIN_MODEL,
                "litellm_params": {
                    "model": f"openai/{MAIN_MODEL}",
                    "api_base": "http://localhost:11434/v1",
                    "api_key": "local"
                }
            }]
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

    spawn_context = multiprocessing.get_context('spawn')
    litellm_process = spawn_context.Process(
        target=_run_litellm_worker,
        daemon=True,
        name="Hermes Translater"
    )
    litellm_process.start()

    return litellm_process

if __name__ == "__main__":
    # Standalone: python -m services.litellm_runner  (run from the project root)
    # Goes through start_litellm_process() so the YAML config gets generated first.
    try:
        start_litellm_process().join()
    except KeyboardInterrupt:
        pass
