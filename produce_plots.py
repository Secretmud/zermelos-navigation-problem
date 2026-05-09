"""
produce_plots.py
----------------
Top-level orchestrator: runs every plot-producing script and places all
output under the plots/ directory.

  plots/
    fidelity/      — bisection traces, diagnostics, summary t*(p) figure
    energy_gap/    — minimal spectral gap vs n
    convergence/   — dt convergence study
    eigenplots/    — avoided crossings, ground-state populations, paths

Run from the project root:
  python produce_plots.py
"""

from pathlib import Path

import scripts.fidelity_threshold as fidelity_threshold
import scripts.energy_gap as energy_gap
import scripts.convergence as convergence
import scripts.eigenplots as eigenplots

PLOTS = Path("plots")

print("=== fidelity threshold ===")
fidelity_threshold.main(PLOTS / "fidelity")

print("\n=== energy gap ===")
energy_gap.main(PLOTS / "energy_gap")

print("\n=== eigenplots ===")
eigenplots.main(PLOTS / "eigenplots")

print(f"\nAll plots written to {PLOTS.resolve()}/")
