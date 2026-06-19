"""
Reproduce Figure 1 (scenario-comparison) from the Frontiers in Education paper.

Runs the corrected CAIE model defined in math/caie_experiment.py and renders the
three Section-3.3 scenarios (Traditional, CAIE without RPE, CAIE with RPE) with
English labels. Output: caie_fig1_scenarios.png in this directory.

Usage:
    python make_fig1.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- import the model -------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MATH_DIR = os.path.join(HERE, "math")
sys.path.insert(0, MATH_DIR)
import caie_experiment as caie  # noqa: E402

# --- English dimension labels (same order as caie.DIM_LABELS) ---------------
EN_LABELS = [
    "WM: Working memory",
    "TS: Task-switching",
    "IM: Indexical memory",
    "MA: Multi-agent management",
    "MC: Metacognition",
    "EI: Emotional intelligence",
    "IS: Identity stability",
]

# Three scenarios matching Section 3.3 (Traditional = red, CAIE no-RPE = orange, CAIE+RPE = green)
SCENARIOS = [
    ("traditional", "Traditional education", "red", "--"),
    ("caie_norpe", "CAIE education (no RPE)", "darkorange", "-."),
    ("caie", "CAIE education (with RPE)", "green", "-"),
]


def main():
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "CAIE model simulation: cognitive trajectories under three education "
        "scenarios (2000-2040)",
        fontsize=16, fontweight="bold",
    )

    results = {sc: caie.run_simulation(scenario=sc) for sc, *_ in SCENARIOS}

    for i in range(caie.N_DIM):
        ax = axes[i // 4][i % 4]
        for sc, label, color, ls in SCENARIOS:
            sol = results[sc]
            ax.plot(sol.t, sol.y[i], color=color, linestyle=ls,
                    label=label, linewidth=2)
        ax.set_title(EN_LABELS[i], fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Ability (normalised)")
        ax.axvline(x=2023, color="black", linestyle=":", alpha=0.5)
        ax.text(2023.3, 0.02, "AI inflection (~2023)", fontsize=7,
                rotation=90, va="bottom", alpha=0.7)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2000, 2040)

    # 8th panel: total technology pressure
    ax = axes[1][3]
    t_range = np.linspace(1800, 2040, 500)
    total_pressure = np.array([np.sum(caie.tech_pressure(t)) for t in t_range])
    ax.plot(t_range, total_pressure, "k-", linewidth=2)
    ax.fill_between(t_range, 0, total_pressure, alpha=0.2, color="orange")
    ax.set_title("Total technology pressure", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Pressure intensity")
    ax.axvline(x=1850, color="brown", linestyle=":", alpha=0.5, label="$T_1$ steam engine")
    ax.axvline(x=2007, color="blue", linestyle=":", alpha=0.5, label="$T_2$ internet")
    ax.axvline(x=2023, color="red", linestyle=":", alpha=0.5, label="$T_3$ AI")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(HERE, "caie_fig1_scenarios.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
