"""
Runs the parameter sweep and saves results/sweep_dataset.csv.

Experiment 1: baseline (p=0) vs mechanism (p>0), cooperation-over-time curves
Experiment 2: full parameter sweep -> dataset for the Exp6 regression task
"""
import csv
import numpy as np
from game import run_single, N

ROUNDS = 200
BURN_IN_FRAC = 0.5
C_BASE = 1.0
REPS_PER_SETTING = 8  # repetitions (different seeds) per parameter combo

# ---- Experiment 1: baseline vs mechanism cooperation-over-time ----
strategy_mix_default = {
    "AlwaysDefect": 0.25,
    "AlwaysCooperate": 0.25,
    "TitForTat": 0.20,
    "Adaptive": 0.30,
}

exp1_rows = []
for label, p in [("baseline_no_mechanism", 0.0), ("with_mechanism", 0.6)]:
    coop_runs = []
    for seed in range(20):
        res = run_single(
            c_base=C_BASE, b_base=3.0, p=p,
            strategy_mix=strategy_mix_default,
            rounds=ROUNDS, burn_in_frac=0.0, seed=seed,
        )
        coop_runs.append(res["coop_history"])
    mean_curve = np.mean(coop_runs, axis=0)
    for t, val in enumerate(mean_curve):
        exp1_rows.append({"scenario": label, "round": t, "cooperation_rate": val})

with open("results/exp1_cooperation_over_time.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scenario", "round", "cooperation_rate"])
    writer.writeheader()
    writer.writerows(exp1_rows)
print(f"Experiment 1 saved: {len(exp1_rows)} rows")

# ---- Experiment 2: parameter sweep -> regression dataset ----
p_values = np.linspace(0.0, 1.2, 7)            # mechanism strength
b_over_c_values = np.linspace(1.5, 5.0, 6)      # benefit/cost ratio
titfortat_shares = np.linspace(0.0, 0.7, 5)     # fraction of TitForTat agents

sweep_rows = []
seed_counter = 1000
for p in p_values:
    for b_over_c in b_over_c_values:
        b_base = C_BASE * b_over_c
        for tft_share in titfortat_shares:
            remaining = 1.0 - tft_share
            strategy_mix = {
                "TitForTat": tft_share,
                "AlwaysDefect": remaining * 0.40,
                "AlwaysCooperate": remaining * 0.20,
                "Adaptive": remaining * 0.40,
            }
            coop_rates = []
            welfares = []
            for rep in range(REPS_PER_SETTING):
                seed_counter += 1
                res = run_single(
                    c_base=C_BASE, b_base=b_base, p=p,
                    strategy_mix=strategy_mix,
                    rounds=ROUNDS, burn_in_frac=BURN_IN_FRAC, seed=seed_counter,
                )
                coop_rates.append(res["steady_coop_rate"])
                welfares.append(res["steady_welfare"])
            sweep_rows.append({
                "mechanism_strength_p": round(p, 3),
                "benefit_cost_ratio": round(b_over_c, 3),
                "titfortat_share": round(tft_share, 3),
                "steady_cooperation_rate": np.mean(coop_rates),
                "steady_welfare": np.mean(welfares),
                "coop_rate_std": np.std(coop_rates),
            })

with open("results/sweep_dataset.csv", "w", newline="") as f:
    fieldnames = list(sweep_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sweep_rows)

print(f"Experiment 2 sweep saved: {len(sweep_rows)} parameter combinations "
      f"({REPS_PER_SETTING} reps each)")
