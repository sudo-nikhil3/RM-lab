# Experiment 6 — Statistical Analysis, Model Validation, Performance Evaluation

**Research problem:** Asymmetric repeated public-goods game of G20 climate cooperation. Does a tariff-style enforcement mechanism increase cooperation, and what drives cooperation most: enforcement strength, the benefit/cost ratio, or the share of reciprocal (tit-for-tat) actors?

**Dataset used:** Generated, not external. A parameter sweep over our own simulation — 210 parameter combinations (mechanism strength `p` × benefit/cost ratio × tit-for-tat population share), 8 repetitions each, 200 rounds per run, steady-state averaged over the second half of each run. Agent parameters (abatement cost asymmetry) are drawn from the G20 table built for Exp5.

---

## 1. Descriptive Statistics (steady-state cooperation rate, n=210 parameter settings)

| Statistic | Value |
|---|---|
| Mean | 0.452 |
| Std. deviation | 0.242 |
| Min | 0.119 |
| 25th percentile | 0.233 |
| Median | 0.370 |
| 75th percentile | 0.703 |
| Max | 0.875 |

Cooperation varies widely across the parameter space (12%–88%), confirming the sweep covers both near-collapse and near-full-cooperation regimes — good, since a regression model needs that spread to be learnable.

## 2. Exploratory Data Analysis

- **Figure 0** — cooperation rate over time, baseline (no mechanism) vs. mechanism (p=0.6): baseline collapses to a steady 31.6% cooperation; the mechanism stabilizes at 38.8% — a real, visible uplift.
- **Figure 1** — histogram of steady-state cooperation rate across all 210 settings.
- **Figure 2** — scatter of mechanism strength vs. cooperation rate.
- **Figure 3** — correlation heatmap across all swept parameters and outcomes.

## 3. Inferential Statistics

Pearson correlation between mechanism strength (`p`) and steady-state cooperation rate: **r = 0.176, p = 0.011** — statistically significant (p < 0.05) but a modest effect size. The mechanism helps, but it is not the dominant lever (see §6).

## 4. Model Development

**Research requirement:** predict a numerical value (steady-state cooperation rate) from the swept parameters → **regression**.

- Features: `mechanism_strength_p`, `benefit_cost_ratio`, `titfortat_share`
- Target: `steady_cooperation_rate`
- Models compared: **Linear Regression** and **Random Forest Regression**

**Justification:** Linear Regression gives an interpretable baseline (does cooperation respond linearly to these levers?). Random Forest captures non-linear interactions (e.g., the mechanism may only matter once enough reciprocal agents are present) and gives feature importance for free — directly useful for the paper's discussion of which lever matters most.

## 5. Model Validation

- Train/test split: 75/25 (157 train, 53 test)
- 5-fold cross-validation on the full dataset (to check the split wasn't lucky/unlucky)

## 6. Performance Evaluation

| Model | Test MAE | Test RMSE | Test R² | 5-fold CV R² (mean ± std) |
|---|---|---|---|---|
| Linear Regression | 0.096 | 0.111 | 0.750 | 0.820 ± 0.026 |
| **Random Forest Regression** | **0.069** | **0.097** | **0.809** | **0.863 ± 0.022** |

**Best-performing model: Random Forest Regression** (lower error, higher R² on both the held-out test set and cross-validation, with a tight CV std — the model generalizes consistently, not just on one lucky split).

**Figure 4** shows actual vs. predicted cooperation rate for the Random Forest on the test set — points cluster tightly around the diagonal, confirming the R² visually.

**Feature importance (Figure 5):**

| Feature | Importance |
|---|---|
| Tit-for-tat population share | 0.907 |
| Mechanism strength (p) | 0.063 |
| Benefit/cost ratio | 0.030 |

## 7. Interpretation

The model performs well (R² ≈ 0.81–0.86), meaning steady-state cooperation is largely predictable from these three parameters. The key finding, consistent with the inferential test in §3: **the tariff mechanism has a real, statistically significant effect on cooperation, but it is a secondary lever.** What dominates is the share of reciprocal (tit-for-tat) actors in the population — cooperation rises sharply once enough agents are willing to condition their behavior on others' past actions, with or without formal enforcement. Substantively, this echoes a recurring point in the climate-cooperation literature: enforcement mechanisms (tariffs, sanctions) help, but durable cooperation depends more on how many actors are predisposed to reciprocate than on punishment strength alone.

**Does this support the research objective?** Yes — the original question was whether a tariff mechanism meaningfully increases cooperation. It does (31.6% → 38.8%, p=0.011), but the sweep additionally reveals *why* the effect is modest: enforcement alone cannot compensate for a population dominated by unconditional defectors.

## 8. Limitations and Future Scope

- The Adaptive (reinforcement-learning) agent's payoff-counterfactual update is a simplification — it assumes a single agent's action switch does not materially change the global cooperation rate, which breaks down for very small N (not an issue here with N=19, but worth stating explicitly).
- Only one mechanism (tariff) is tested; the model does not compare it against side-payments or pure reputation effects.
- Abatement cost asymmetry is modeled as proportional to GDP share only — benefit (climate damage avoided) is treated as a symmetric global public good, which abstracts away real heterogeneity in climate vulnerability across G20 members.
- Next step: sweep a wider `p` range and test whether the tariff's importance grows once the tit-for-tat share is held fixed at a realistic, literature-informed value rather than varied freely.
