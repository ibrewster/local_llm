"""Manual "try me" script for the bedtime poem generation in MLXOllama.worker.gen_goodnight.

Runs the same request as gen_goodnight, but prints the LLM output instead of sending it to TTS.
Not collected by pytest (filename doesn't match test_*.py).

Usage (from the repo root or the tests folder):
    python tests/try_goodnight.py
    python tests/try_goodnight.py Monday --aurora
    python tests/try_goodnight.py friday --forecast "Low of 40°F, high of 62°F. Sunny."
"""
import argparse
import calendar
import datetime
import sys
import time
import types

from pathlib import Path
from unittest import mock

# Make the MLXOllama package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MLXOllama import config, utils, worker

DEFAULT_FORECAST = "Low of -12°F, high of 3°F. Light snow in the morning, clearing to partly cloudy skies by afternoon."


def try_gen_goodnight(forecast=DEFAULT_FORECAST, aurora_alert=False, day_of_week=None):
    """Generate a goodnight poem and print it.

    day_of_week (e.g. "Monday") makes the goodnight code think today is the next occurrence of that day,
    by monkeypatching datetime.datetime.now in the worker module only.
    """
    # Load the models if we aren't running inside the full app
    if config.MAIN_MODEL not in utils.loaded_models:
        for name, path in config.models.items():
            utils.inference_worker.submit(utils.load_model, name, path).result()

    fake_now = datetime.datetime.now()
    if day_of_week:
        target = list(calendar.day_name).index(day_of_week.capitalize())
        fake_now += datetime.timedelta(days=(target - fake_now.weekday()) % 7)

    class _FakeDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fake_now

    # Replace worker's "datetime" name only, so nothing outside worker.py sees the fake date
    fake_datetime_module = types.SimpleNamespace(**{**vars(datetime), "datetime": _FakeDatetime})

    tomorrow_name = (fake_now + datetime.timedelta(days=1)).strftime("%A")
    aurora_line = "An aurora alert IS active tonight." if aurora_alert else "No aurora alert is active."

    # Must start with "It is bedtime" to match the routing in speak_thread
    prompt = f"""It is bedtime. Please write tonight's goodnight poem.
Tomorrow is {tomorrow_name}.
Tomorrow's forecast: {forecast}
{aurora_line}"""

    with mock.patch.dict(vars(worker), {"datetime": fake_datetime_module}):
        message, sampler = worker._build_goodnight_request(prompt)

    # Print the full user message, so the selected POEM_STYLE can be verified
    print(f"PROMPT:\n{message[1]['content']}\n\nRESPONSE:")

    t0 = time.time()
    token_queue, _ = worker.submit_inference_job(message, sampler=sampler, priority=True)

    response = ""
    while True:
        text, token, is_done, err = token_queue.get()
        if err:
            print(f"\nUnable to generate result: {err}")
            break

        response += text
        print(text, end="", flush=True)

        if is_done:
            break

    print(f"\n\nGenerated in {time.time() - t0:.2f} seconds")
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and print a goodnight poem")
    parser.add_argument("day_of_week", nargs="?", help="Pretend today is this day (e.g. Monday)")
    parser.add_argument("--aurora", action="store_true", help="Say an aurora alert is active")
    parser.add_argument("--forecast", default=DEFAULT_FORECAST, help="Tomorrow's forecast text")
    args = parser.parse_args()

    try_gen_goodnight(forecast=args.forecast, aurora_alert=args.aurora, day_of_week=args.day_of_week)
