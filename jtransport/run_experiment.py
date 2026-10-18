"""Run the J-lens transport experiments end to end: `uv run python -m jtransport.run_experiment`.

Local: loads .env (jtransport/config.py), opens a timestamped log (logs/<ts>_run_experiment.log + terminal), and calls
the phases of jtransport/phases.py in order; PHASES=... in .env (or the environment) selects a subset. Every phase
skips work whose outputs exist, so re-running resumes an interrupted run.
Global context: experiment 1 compares the paper's raw cross-layer copy with the Jacobian-lens transport in the
self-patching oracle; experiment 2 compares LRSD with a J-space LRSD loss. See README.md and docs/jtransport.md.
"""
import logging
import time

# Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is NOT set: under WSL2 it fails with
# "CUDA driver error: out of memory" in the training subprocess (observed on the RTX 5070 laptop).
from jtransport.config import load_config
from jtransport.log import setup_logging
from jtransport.phases import PHASES


def main():
    cfg = load_config()
    setup_logging("run_experiment", cfg.log_dir)
    log = logging.getLogger("run_experiment")
    log.info("config: %s", cfg.as_dict())
    selected = [p for p in PHASES if not cfg.phases or p.__name__ in cfg.phases]
    unknown = set(cfg.phases) - {p.__name__ for p in PHASES}
    if unknown:
        raise ValueError(f"unknown PHASES {sorted(unknown)}; known: {[p.__name__ for p in PHASES]}")
    t_all = time.time()
    for phase in selected:
        log.info("===== phase %s =====", phase.__name__)
        t0 = time.time()
        phase(cfg)
        log.info("===== phase %s done in %.1f min =====", phase.__name__, (time.time() - t0) / 60)
    log.info("all phases done in %.1f min", (time.time() - t_all) / 60)


if __name__ == "__main__":
    main()
