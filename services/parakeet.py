import logging
import multiprocessing
import sys
import threading

from pathlib import Path

import MLXOllama # Sets HF_HOME config and initializes logging

def _run_parakeet():
    import setproctitle

    setproctitle.setproctitle("wyoming-mlx-parakeet")
    threading.current_thread().name = 'PARAKEET_MAIN'

    parakeet_path = Path(__file__).parents[1] / "wyoming-parakeet-mlx"

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


def start_parakeet_process():
    spawn_context = multiprocessing.get_context('spawn')
    parakeet_process = spawn_context.Process(
        target=_run_parakeet,
        daemon=True,
        name="wyoming-mlx-parakeet"
    )
    parakeet_process.start()

    return parakeet_process