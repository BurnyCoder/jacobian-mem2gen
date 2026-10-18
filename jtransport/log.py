"""Timestamped logging to a file and the terminal, and subprocess output streamed into the same log.

Local: `setup_logging(name, log_dir)` configures the root logger once with a FileHandler (logs/<YYYYmmdd_HHMMSS>_<name>.log)
and a StreamHandler (stderr), both with "time | level | logger | message" lines
(https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations).
`run_cmd(cmd)` runs a paper script and logs each of its output lines as it arrives (realtime, line-buffered).
Global context: the wrapper (jtransport/run_experiment.py) runs long GPU phases; one log file per run holds every
phase, including the paper scripts' prints, so a run can be inspected afterwards.
"""
import logging
import os
import subprocess
import sys
import time

FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"  # timestamp first for grep/sort


def setup_logging(name, log_dir):
    """Configure root logging (idempotent); returns the path of the log file."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.log")  # one timestamped file per run
    root = logging.getLogger()
    if not any(getattr(h, "_jtransport", False) for h in root.handlers):  # don't double-attach on re-entry
        root.setLevel(logging.INFO)
        for h in (logging.FileHandler(path), logging.StreamHandler(sys.stderr)):
            h.setFormatter(logging.Formatter(FMT))
            h._jtransport = True  # marker for idempotency
            root.addHandler(h)
        for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):  # keep HTTP chatter out of the log
            logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger(__name__).info("logging to %s", path)
    return path


def run_cmd(cmd, cwd=None, env=None):
    """Run `cmd` (list), streaming merged stdout/stderr into the log line by line; raise on a non-zero exit.

    Source: subprocess.Popen with a pipe and `bufsize=1, text=True` is line-buffered
    (https://docs.python.org/3/library/subprocess.html#popen-constructor); PYTHONUNBUFFERED makes the child flush.
    """
    log = logging.getLogger("cmd")
    log.info("$ %s", " ".join(cmd))
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", **(env or {})}
    t0 = time.time()
    with subprocess.Popen(cmd, cwd=cwd, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as p:
        for line in p.stdout:  # yields as soon as the child writes a line
            log.info("  %s", line.rstrip())
    if p.returncode:
        raise RuntimeError(f"command failed with exit code {p.returncode}: {' '.join(cmd)}")
    log.info("done in %.1f min", (time.time() - t0) / 60)
