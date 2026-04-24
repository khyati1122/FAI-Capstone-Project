# Comparing Tabular Temporal Difference Control Algorithms
**CS5100 — Foundations of Artificial Intelligence | Northeastern University**
Khyati Nirenkumar Amin · Ihika Narayana Reddy Gari

---

## What This Project Does

Four major Temporal Difference control algorithms — Q-learning, SARSA, Expected SARSA, and Double Q-learning — were each published in different papers and never systematically compared under controlled conditions. This project fills that gap.

We implement all four from scratch and empirically answer: **under what environment conditions does each algorithm's design tradeoff actually matter?**

---

## Algorithms

| Algorithm | Type | Key Idea | Paper |
|---|---|---|---|
| Q-learning | Off-policy | `max Q(s',a')` — assumes optimal next action | Watkins & Dayan (1992) |
| SARSA | On-policy | `Q(s',a')` where a' follows current policy | Rummery & Niranjan (1994) |
| Expected SARSA | On-policy | Weighted average over all next actions | Van Seijen et al. (2009) |
| Double Q-learning | Off-policy | Two Q-tables to correct maximization bias | Van Hasselt (2010) |

---

## Experiments

| # | Question | Lead |
|---|---|---|
| Exp 1 | What breaks when convergence conditions are violated? | Khyati |
| Exp 2 | How do all four algorithms compare across five environments? | Joint |
| Exp 3 | How sensitive is each algorithm to hyperparameter choices? | Ihika |
| Exp 4 | Is Q-learning's maximization bias measurable — and does Double Q-learning fix it? | Ihika |

---

## Environments

| Environment | States | Actions | Varies |
|---|---|---|---|
| Custom 4×4 GridWorld | 16 | 4 | Baseline (Exp 1) |
| FrozenLake 4×4 det / slip | 16 | 4 | Stochasticity |
| FrozenLake 8×8 det / slip | 64 | 4 | Scale + Stochasticity |
| CliffWalking | 48 | 4 | Risk structure |
| Taxi-v3 | 500 | 6 | Scale |

---

## Key Findings

- **Exploration failure is catastrophic** — greedy conditions flatline immediately regardless of learning rate schedule
- **Q-learning converges fastest** in tabular settings across most environments
- **SARSA outperforms Q-learning on CliffWalking** during training — on-policy conservatism avoids the −100 cliff penalty
- **Double Q-learning's bias correction** is measurable but trades off sample efficiency at small tabular scale
- **Expected SARSA** is the most robust to hyperparameter choices across all environments

---

## Stack

```
numpy · gymnasium · matplotlib · seaborn · scipy · pandas · joblib
```
All algorithms implemented from scratch — no RL libraries used.

---

## Structure

```
├── notebooks/
│   ├── experiment_1_convergence.ipynb
│   ├── experiment_2_comparison.ipynb
│   ├── experiment_3_sensitivity.ipynb
│   └── experiment_4_bias.ipynb
├── results/
│   ├── exp1/
│   ├── exp2/
│   ├── exp3/
│   └── exp4/
└── README.md
```

---

## References

1. Watkins & Dayan (1992). Q-learning. *Machine Learning*, 8(3–4).
2. Rummery & Niranjan (1994). On-line Q-learning using connectionist systems. Cambridge Technical Report.
3. Van Seijen et al. (2009). A theoretical and empirical analysis of Expected SARSA. *IEEE ADPRL*.
4. Van Hasselt (2010). Double Q-learning. *NeurIPS 23*.
5. Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
