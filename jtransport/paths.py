"""Repository paths and import setup for the paper's script directories.

Local: REPO is the repository root; `use_repo_modules()` puts patch/, distill/ and train/ on sys.path.
Global context: the paper's code is a set of scripts, not packages (patch/utils.py, distill/common.py, ... import each
other by bare module name, e.g. `from common import ...`). jtransport reuses their functions instead of copying them,
so it imports them the same way the scripts do (https://docs.python.org/3/library/sys.html#sys.path).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../Mem2Gen (parent of jtransport/)
PATCH_DIR = os.path.join(REPO, "patch")  # TransformerLens self-patching code (layer_patching.py, utils.py)
DISTILL_DIR = os.path.join(REPO, "distill")  # LRSD code (common.py, train_inject_lrsd.py, make_inject_data.py)
TRAIN_DIR = os.path.join(REPO, "train")  # injection data sampler (training_utils.py)


def use_repo_modules():
    """Make `import layer_patching`, `import utils`, `import common` resolve to the paper's modules (idempotent)."""
    for d in (PATCH_DIR, DISTILL_DIR, TRAIN_DIR):
        if d not in sys.path:
            sys.path.append(d)  # append, not insert: never shadow installed packages with the scripts' short names


def repo_path(*parts):
    """Absolute path inside the repository (config values may be repo-relative)."""
    p = os.path.join(*parts)
    return p if os.path.isabs(p) else os.path.join(REPO, p)
