"""jtransport: Jacobian-lens (J-lens) corrected cross-layer transport for the Mem2Gen experiments.

Local: package marker; the modules are imported explicitly (jtransport.transport, jtransport.grid, ...).
Global context: the paper (arXiv 2607.08393) copies residual states between layers raw (self-patching oracle, LRSD).
This package replaces the raw copy with the J-lens map of Gurnee et al. 2026
(https://transformer-circuits.pub/2026/workspace/index.html, code https://github.com/anthropics/jacobian-lens)
and runs both experiments end to end: `uv run python -m jtransport.run_experiment`.
"""
