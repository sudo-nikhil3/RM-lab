# Asymmetric Climate Cooperation Game (G20)

A repeated public-goods game simulation testing whether a tariff-style
enforcement mechanism increases international climate cooperation, using
G20 countries as asymmetric agents (cost of abatement scaled by economic
size).

## Problem statement

In a repeated N-player public-goods game representing G20 climate
mitigation, does introducing a tariff enforcement mechanism meaningfully
increase long-run cooperation compared to a no-enforcement baseline — and
what actually drives cooperation more: enforcement strength, the
benefit/cost ratio, or the share of reciprocal (tit-for-tat) agents in the
population?

## Folder structure

```
climate_prototype/
├── README.md
├── requirements.txt
├── data/
│   ├── g20_params.json              # GDP/emission shares per G20 country (used by game.py)
│   └── G20_Climate_Game_Parameters.xlsx   # source table with formulas
├── src/
│   ├── game.py                      # core game model (agents, payoffs, mechanism)
│   ├── run_experiment.py            # runs the simulation sweep, writes results/*.csv
│   └── analyze.py                   # EDA, regression models, validation, figures
├── results/
│   ├── sweep_dataset.csv            # 210 parameter combos x outcomes (the "dataset")
│   ├── exp1_cooperation_over_time.csv
│   ├── exp6_report.json             # all stats/metrics as structured output
│   └── figures/                     # 6 PNG plots
└── docs/
    └── exp6_report.md               # written analysis (Exp6 submission)
```

## How to run

```bash
pip install -r requirements.txt

# from the project root (climate_prototype/)
python src/run_experiment.py    # runs the simulation, writes results/*.csv
python src/analyze.py           # EDA + regression models, writes results/figures/*.png
```

## Model (summary — see `docs/exp6_report.md` for full write-up)

- **Agents:** 19 G20 countries (EU excluded to avoid double-counting
  France/Germany/Italy)
- **Abatement cost:** scaled by each country's GDP share (bigger economies
  face proportionally higher absolute abatement cost)
- **Benefit:** shared global public good, split equally
- **Mechanism:** cooperators impose a tariff `p` on defectors, scaled by
  the enforcing cooperators' emission share
- **Strategies:** Always Defect, Always Cooperate, Tit-for-Tat, and an
  Adaptive (reinforcement-learning) agent that is the only strategy type
  that actually responds to the mechanism

## Key result

The mechanism raises steady-state cooperation from **31.6% → 38.8%**
(statistically significant, p = 0.011), but a Random Forest regression
(R² ≈ 0.81 test / 0.86 cross-validated) shows the share of reciprocal
(tit-for-tat) agents matters roughly **10x more** than enforcement
strength — cooperation depends more on population composition than on
punishment alone.

## Assumptions to confirm before using in the paper

- Abatement cost formula (`cost ∝ GDP share`) and the tariff mechanism
  design are defaults chosen to get a working prototype — not yet signed
  off by the group's methodology section.
- Benefit side of the payoff is symmetric (equal share of a global public
  good); real climate vulnerability differs a lot across G20 members and
  isn't modeled here.
- Only one mechanism (tariff) is implemented; side-payments and pure
  reputation effects are not yet compared.

## Limitations / future scope

See `docs/exp6_report.md`, section 8.
