"""
Asymmetric repeated public-goods game for G20 climate cooperation.

Model (ASSUMPTION - confirm with group / methodology section):
  - N = 19 agents (G20 countries, EU excluded)
  - Each round, agent i chooses Cooperate (abate) or Defect (pollute)
  - Abatement cost for agent i:  c_i = c_base * (gdp_share_i * N)
        -> bigger economies face proportionally higher absolute abatement cost
  - Benefit is a shared global public good, split equally among all N agents:
        benefit_i(t) = b_base * (num_cooperators(t) / N)
  - Baseline payoff:  pi_i(t) = benefit_i(t) - c_i * [agent i cooperated]
  - Mechanism (tariff): cooperators impose a per-defector penalty `p` on each
    defector, scaled by the enforcing cooperators' emission_share (bigger
    emitters "enforce" harder). Defector's payoff is reduced by
    p * sum(emission_share_j for j in cooperators).

  - Strategies (mixed population, assigned per-run):
        AlwaysDefect, AlwaysCooperate, TitForTat (cooperate if last-round
        global cooperation rate >= 0.5), Adaptive (reinforcement-learning
        agent: tracks a cooperation propensity q_i, nudges it toward
        whichever action yielded the higher payoff last round). Adaptive
        is the only strategy type that actually responds to the mechanism
        strength p -- without it, p has no behavioral effect, since the
        other three strategies ignore realized payoffs entirely.
"""
import json
import numpy as np

with open("data/g20_params.json") as f:
    COUNTRIES = json.load(f)

N = len(COUNTRIES)
GDP_SHARE = np.array([c["gdp_share"] for c in COUNTRIES])
EMISSION_SHARE = np.array([c["emission_share"] for c in COUNTRIES])

STRATEGIES = ["AlwaysDefect", "AlwaysCooperate", "TitForTat", "Adaptive"]
ADAPTIVE_LR = 0.08


def assign_strategies(strategy_mix, rng):
    names = list(strategy_mix.keys())
    probs = list(strategy_mix.values())
    return rng.choice(names, size=N, p=probs)


def run_single(c_base, b_base, p, strategy_mix, rounds, burn_in_frac, seed):
    rng = np.random.default_rng(seed)
    cost = c_base * (GDP_SHARE * N)  # asymmetric abatement cost
    strategies = assign_strategies(strategy_mix, rng)
    is_adaptive = strategies == "Adaptive"

    propensity = np.full(N, 0.5)  # only meaningful for Adaptive agents
    last_coop_rate = None

    coop_history = []
    welfare_history = []

    for t in range(rounds):
        actions = np.empty(N, dtype=int)
        for i, s in enumerate(strategies):
            if s == "AlwaysDefect":
                actions[i] = 0
            elif s == "AlwaysCooperate":
                actions[i] = 1
            elif s == "TitForTat":
                actions[i] = 1 if (last_coop_rate is None or last_coop_rate >= 0.5) else 0
            elif s == "Adaptive":
                actions[i] = 1 if rng.random() < propensity[i] else 0

        coop_mask = actions == 1
        num_coop = coop_mask.sum()
        coop_rate = num_coop / N
        benefit = b_base * coop_rate

        payoffs = benefit - cost * coop_mask
        enforcement = p * EMISSION_SHARE[coop_mask].sum() if (p > 0 and num_coop > 0) else 0.0
        payoffs[~coop_mask] -= enforcement

        # Reinforcement update for Adaptive agents: compare realized payoff
        # structure for cooperating vs. defecting this round (approximating
        # the global cooperation rate as unchanged by one agent's switch,
        # since N=19 and a single switch moves it by ~5%).
        if is_adaptive.any():
            cooperate_payoff = benefit - cost          # if agent had cooperated
            defect_payoff = benefit - enforcement        # if agent had defected
            diff = cooperate_payoff - defect_payoff
            propensity[is_adaptive] = np.clip(
                propensity[is_adaptive] + ADAPTIVE_LR * diff[is_adaptive], 0.0, 1.0
            )

        coop_history.append(coop_rate)
        welfare_history.append(payoffs.mean())
        last_coop_rate = coop_rate

    burn_in = int(rounds * burn_in_frac)
    steady_coop = float(np.mean(coop_history[burn_in:]))
    steady_welfare = float(np.mean(welfare_history[burn_in:]))
    return {
        "coop_history": coop_history,
        "welfare_history": welfare_history,
        "steady_coop_rate": steady_coop,
        "steady_welfare": steady_welfare,
    }
