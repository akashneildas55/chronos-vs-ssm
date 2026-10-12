# chronos-vs-ssm

> **Evaluating Foundation Time-Series Models (Amazon Chronos-2) vs. Bayesian State-Space Formulations for European Day-Ahead BESS Arbitrage across DE-LU, DK1, and DK2.**

---

## 1. The Core Thesis: Calibration vs. Economic Value

Standard quantitative pipelines evaluate time-series forecasting models using statistical point or distributional losses—primarily Mean Absolute Error ($\text{MAE}$), Root Mean Squared Error ($\text{RMSE}$), or Continuous Ranked Probability Score ($\text{CRPS}$):

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left( F(z) - \mathbb{I}(z \ge y) \right)^2 dz$$

In electricity market algorithmic trading, **minimizing statistical loss does not maximize realized economic profit ($\text{PnL}$)**. 

A Battery Energy Storage System (BESS) generating revenue via Day-Ahead arbitrage does not capture value from baseline price accuracy. It captures value strictly from **cross-block price spreads** exceeding the round-trip efficiency ($\text{RTE}$) degradation hurdle. A conservatively calibrated model with low $\text{CRPS}$ often underestimates extreme tail events, causing the optimizer to skip profitable dispatch windows. Conversely, a structurally informed model that captures peak and trough spreads—even with higher residual variance—drives higher realized $\text{PnL}$.

This project establishes an empirical benchmark between:
1. **Amazon Chronos-2:** A zero-shot probabilistic foundation time-series model (transformer-based, tokenized architecture) conditioned on known day-ahead covariates.
2. **Bayesian State-Space Models (SSM):** A domain-tailored structural Dynamic Linear Model with variance-stabilizing transforms ($\text{asinh}$), diurnal harmonic seasonality, time-varying residual load coefficients, and Student-$t$ observation likelihoods.

---

## 2. Market Architecture & Constraints

The evaluation simulates physical asset operations within the European **Single Day-Ahead Coupling (SDAC)** framework across three coupled bidding zones:

* **DE-LU (Germany-Luxembourg):** High-volume, renewable merit-order suppression, thermal baseload, and high frequency of negative clearing prices.
* **DK1 (West Denmark - Jutland):** High onshore/offshore wind penetration with synchronous interconnectors to DE-LU and Norway; frequent negative residual load.
* **DK2 (East Denmark - Zealand):** Eastern Nordic AC synchronous zone connected via DC links, exposing interconnector bottlenecks and locational marginal spreads against DK1/DE-LU.

### Operational Timelines & Information Set
* **Auction Mechanism:** Pay-as-cleared uniform pricing cleared via the Euphemia algorithm.
* **Gate Closure:** Daily at **12:00 CET** on day $D$.
* **Delivery Horizon:** Day $D+1$ ($00:00$ to $23:00$, 96 quarter-hour blocks under the 15-minute resolution standard).
* **Information Set ($\mathcal{I}_{D, 12:00}$):** Strictly restricted to spot prices cleared up to hour 12:00 of day $D$, along with TSO load, wind, and solar forecasts officially published for day $D+1$ prior to gate closure. Strict temporal guardrails enforce zero look-ahead leakage.

---

## 3. Battery Storage Dispatch Formulation

The downstream benchmark evaluates a standardized grid-scale BESS:
* **Nominal Power ($P_{\max}$):** 2.0 MW
* **Usable Capacity ($E_{\max}$):** 4.0 MWh (2-hour duration)
* **Round-Trip Efficiency ($\eta_{\text{rt}}$):** 85% ($\eta_{\text{ch}} = \eta_{\text{dis}} = \sqrt{0.85} \approx 0.922$)
* **Marginal Degradation Cost ($c_{\text{deg}}$):** €5.00/MWh throughput

### Mathematical Dispatch Program
For each delivery day $D+1$ across intervals $t \in \{1, \dots, T\}$:

$$\max_{\{p_t^{\text{ch}}, p_t^{\text{dis}}\}} \sum_{t=1}^T \Delta t \left( \hat{y}_t \cdot p_t^{\text{dis}} - \hat{y}_t \cdot p_t^{\text{ch}} - c_{\text{deg}} \cdot (p_t^{\text{ch}} + p_t^{\text{dis}}) \right)$$

$$\text{subject to:}$$
$$0 \le p_t^{\text{ch}} \le P_{\max}, \quad 0 \le p_t^{\text{dis}} \le P_{\max}$$
$$\text{SoC}_t = \text{SoC}_{t-1} + \Delta t \left( \eta_{\text{ch}} p_t^{\text{ch}} - \frac{1}{\eta_{\text{dis}}} p_t^{\text{dis}} \right)$$
$$\text{SoC}_{\min} \le \text{SoC}_t \le \text{SoC}_{\max}$$
$$\text{SoC}_0 = \text{SoC}_T = 0.5 \cdot E_{\max}$$

The dispatch schedule $\{p_t^{\text{ch}}, p_t^{\text{dis}}\}$ derived from forecast distributions $\hat{y}_t$ is executed against actual historical clearing prices $y_t$ to record realized $\text{PnL}$.

---

## 4. Repository Structure

```text
chronos-vs-ssm/
├── .env.example             # Template for API keys
├── .gitignore               # Enforces data and secret isolation
├── README.md                # Project documentation and architectural spec
├── MODEL_CARD.md            # Calibration, training regimes, and failure modes
├── data/
│   ├── raw/                 # Raw ENTSO-E parquet/CSV extracts (untracked)
│   └── processed/           # Canonical localized data contract tables
├── notebooks/               # Exploratory data analysis and visual reports
├── configs/                 # Bidding zone YAML configs (DE-LU, DK1, DK2)
├── tests/
│   ├── test_leakage.py      # Gate closure & information set leakage assertions
│   └── test_battery.py      # Conservation of energy & dispatch constraint checks
└── src/
    └── radar/
        ├── __init__.py
        ├── data/            # ENTSO-E client, synthetic generator, rolling splits
        ├── features/        # Residual load and calendar feature extractors
        ├── models/          # Chronos-2 inference and NumPyro/Pyro Bayesian SSM
        ├── battery/         # Linear programming (LP) dispatch simulator
        └── eval/            # CRPS, pinball loss, and realized PnL evaluators
