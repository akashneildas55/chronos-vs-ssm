# chronos-vs-ssm

> **Does a better probabilistic price forecast make a battery earn more?**
> A benchmark of Amazon Chronos-2, a Bayesian state-space model (Pyro) and standard electricity-price-forecasting baselines, scored on forecast quality (MAE, RMSE, CRPS) and on simulated day-ahead BESS arbitrage profit. Run independently for **DK1, DK2 and DE-LU**.

**Status:** specification stage. Nothing below is a result yet. See the [roadmap](#6-roadmap).

---

## 1. Research question and hypotheses

Forecasts are usually ranked by statistical loss (MAE, RMSE, CRPS). A battery, however, earns money from **price spreads between time slots** that exceed its round-trip loss and wear cost, so the best statistical forecast is not guaranteed to be the most profitable one.

**Question.** How do statistical forecast quality and realized battery profit relate across forecasting paradigms: a zero-shot foundation model, a structural Bayesian model, and strong classical baselines?

The hypotheses below were written **before any experiment was run**. They are expectations to be tested, not findings.

- **H1.** Rankings by CRPS and rankings by realized profit will disagree for at least some models and zones.
- **H2.** In a pure day-ahead setting (one fixed schedule submitted before prices are known), forecast uncertainty can affect the schedule only through risk aversion (CVaR). A risk-neutral stochastic program depends only on the mean price, so it is equivalent to a point-forecast LP on the scenario mean. Calibrated joint scenarios should therefore matter for CVaR dispatch but not for risk-neutral dispatch.
- **H3.** Zero-shot Chronos-2 will not clearly beat strong market-specific baselines (LEAR with conformal prediction, MSTL) in every zone.

---

## 2. Market setting and information set

The simulation covers the European **Single Day-Ahead Coupling (SDAC)** auction (Euphemia, uniform pay-as-cleared pricing) for three bidding zones, each run **independently**:

| Zone | Why it is included |
|---|---|
| **DE-LU** | Large market with high renewable share and frequent negative prices |
| **DK1** | West Denmark: high wind share, interconnected with DE-LU and Norway |
| **DK2** | East Denmark: Nordic synchronous area, different spread behaviour from DK1 |

### Timeline for the auction held on day D

1. **~12:45 on day D-1:** prices for all of day D are published.
2. **12:00 CET on day D:** gate closure. A battery schedule for day **D+1** must be submitted.
3. **Delivery day D+1:** 96 quarter-hour slots (92 or 100 on the two daylight-saving days).

### Information set at decision time

At gate closure on day D, a model may use:

- **All realized prices up to the end of day D** (they were cleared on D-1).
- **No prices from D+1 or later.**
- **TSO load, wind and solar forecasts for D+1 published before 12:00 on day D.**
- **Fuel and carbon prices** (gas, carbon; optionally oil and coal), using the latest closing price available at bidding time (a two-day lag, following the convention in Lipiecki & Weron, 2026).
- Calendar features (hour, weekday, holidays).

Anything else is leakage. This rule is enforced in code by a single splitter function and tested in `tests/test_leakage.py` (see [section 8](#8-leakage-and-fairness-controls)).

### Resolution

Day-ahead market time unit moved from hourly to 15 minutes in autumn 2025 (confirm the exact date against the ENTSO-E and exchange notices before relying on it). Therefore:

- **Main benchmark:** hourly (15-minute data aggregated to hourly), over several years, so that results are comparable with published studies and cover more than one year.
- **Extension:** native 15-minute data, which covers a much shorter period.

---

## 3. Models

All forecasters implement one interface and return samples shaped `[n_samples, horizon]`.

### Baselines
- **Seasonal naive** (same slot yesterday and last week) and **persistence**.
- **MSTL**: multiple-seasonality decomposition. A strong, cheap benchmark in the literature.
- **LEAR + conformal prediction**: Lasso-estimated autoregressive model with an asinh transform, averaged over several training windows, turned probabilistic with conformal prediction. Implemented via or checked against `epftoolbox`.
- **LightGBM quantile regression** on the same features.
- *Optional:* a distributional neural network (DDNN-JSU) as in Marcjasz et al. (2023).

### Amazon Chronos-2 (zero-shot)
- Encoder-only transformer with a quantile output head. It returns **marginal quantiles per time step**, not joint sample paths.
- **Track A:** price history only.
- **Track B:** with covariates (load, wind and solar forecasts, fuels).
- **Chronos-2-synth** (pretrained on synthetic data only) is used as a **contamination control**.
- *Optional:* fine-tuned Chronos-2, trained only on the training window.

### Bayesian state-space model (Pyro)
- Structural dynamic linear model with an **asinh** variance-stabilizing transform, diurnal and weekly harmonic seasonality, time-varying coefficients on residual load, and a **Student-t** observation likelihood.
- Returns **joint posterior-predictive sample paths** natively.
- Inference scheme (exact filtering, SVI or MCMC) is documented per experiment. Mean-field SVI is known to understate uncertainty, so calibration is checked explicitly (PIT, coverage).
- Metrics are always computed **back in EUR/MWh**.

### Scenario layer
Chronos-2 and LEAR+CP produce marginal quantiles only. To drive stochastic dispatch they are converted into joint scenarios with **one shared method and seed** for all such models (Gaussian copula with correlation estimated from past forecast errors, or Schaake shuffle). This choice affects profit and is recorded in the decisions log.

---

## 4. Covariates

Every variable is typed and carries an availability rule, and the leakage tests cover all of them.

| Type | Examples | Rule |
|---|---|---|
| Past-only | Realized prices, realized generation | Context window only |
| Known-future | Calendar, TSO load/wind/solar forecasts | Allowed for the horizon if published before 12:00 on day D |
| Slow-moving | Gas, carbon, (oil, coal) | Latest close available at bidding time |
| Optional | Archived weather forecasts, neighbouring-zone prices | Forecasts only, never observations or reanalysis |

Tracks are run as **price-only → + TSO forecasts → + fuels (→ + weather)** to show how much each model gains from extra information. Both Chronos-2 and the SSM receive the same variables.

**Known limitation:** ENTSO-E serves the latest published version of each forecast, not the version available at decision time. Forecast vintages are therefore not modelled in v1.0 and are documented as a limitation.

---

## 5. Battery and dispatch

### Asset
| Parameter | Value |
|---|---|
| Power $P_{\max}$ | 2.0 MW |
| Usable energy $E_{\max}$ | 4.0 MWh (2 h) |
| Round-trip efficiency | 85% ($\eta_{ch}=\eta_{dis}=\sqrt{0.85}\approx 0.922$) |
| Degradation cost $c_{deg}$ | €5.00 per MWh charged and per MWh discharged (a full cycle costs twice this) |

The same battery is used in every zone so profits are comparable.

### Dispatch LP
For each delivery day, over slots $t=1,\dots,T$:

$$\max \sum_{t}\Delta t\Big(y_t\,(p^{dis}_t-p^{ch}_t)-c_{deg}\,(p^{ch}_t+p^{dis}_t)\Big)$$

subject to

$$0\le p^{ch}_t,\,p^{dis}_t\le P_{\max},\qquad \mathrm{SoC}_t=\mathrm{SoC}_{t-1}+\Delta t\Big(\eta_{ch}p^{ch}_t-\tfrac{1}{\eta_{dis}}p^{dis}_t\Big),$$

$$\mathrm{SoC}_{\min}\le \mathrm{SoC}_t\le \mathrm{SoC}_{\max},\qquad \mathrm{SoC}_0=\mathrm{SoC}_T=0.5\,E_{\max}.$$

Simultaneous charging and discharging (which a pure LP can exploit at negative prices) is checked by a test and prevented if it occurs. The cyclic SoC constraint rules out multi-day arbitrage by design.

### Strategies compared (one optimizer for every forecaster)

1. **Point dispatch:** LP on the median forecast.
2. **Risk-neutral stochastic:** maximize average profit over $S$ scenarios. By linearity this equals the LP on the scenario mean price (see H2).
3. **CVaR dispatch:** maximize
   $(1-\lambda)\,\overline{\text{profit}} + \lambda\,\mathrm{CVaR}_\alpha(\text{profit})$ using the Rockafellar–Uryasev formulation with auxiliary variables $\zeta$ and $u_s\ge 0$, $u_s\ge\zeta-\text{profit}_s$.
4. **Oracle:** perfect foresight, the upper bound on profit.

*Planned for comparability with published work:* the unlimited-bid and quantile-based limit-order strategies used in the literature.

### Settlement assumptions
- The schedule is fixed before prices are known (self-scheduling) and settled at the realized cleared price. No price-dependent bid curves.
- Price-taker: the battery does not move the market.
- Day-ahead only: no intraday or balancing revenue.

---

## 6. Roadmap

Each version is a git tag and leaves the repository in a presentable state.

- [ ] **v0.1** Data ingest and validation (DK1), splitter, battery simulator, oracle and persistence, leakage and battery-invariant tests, CI
- [ ] **v0.2** Seasonal naive, Chronos-2 (price-only), Pyro SSM, MAE/RMSE/CRPS, forecasts cached to disk
- [ ] **v0.3** Backtest runner, point dispatch, capture ratio
- [ ] **v0.4** Scenario layer, risk-neutral and CVaR dispatch
- [ ] **v0.5** Baselines (MSTL, LEAR+CP, LightGBM), covariate tracks, fine-tuned Chronos-2, Chronos-2-synth control
- [ ] **v0.6** DK2 and DE-LU, energy score, PIT, significance tests, multi-year runs
- [ ] **v1.0** Results, plots, limitations, reproduction guide

Build order inside each version: **data → splitter → battery → forecasters**, so the simulator is trusted before any model is attached.

---

## 7. Evaluation

**Statistical (in EUR/MWh, per zone):**
MAE, RMSE, CRPS, pinball loss, empirical coverage and interval width, PIT histograms, energy score and variogram score for whole trajectories. CRPS is estimated on a **shared quantile grid** for all models so estimators are comparable. Additional measures of forecast usefulness for storage (dispersion of intraday errors, association with realized price shape, identification of daily extrema) follow Maciejowska et al.

**Financial (per zone):**
Total and daily profit, **capture ratio** (profit divided by oracle profit), number of cycles, distribution of daily profit.

**Inference (within each zone):**
Paired block bootstrap on daily profit, Diebold–Mariano tests, and the multivariate conditional predictive ability test over the 24-hour vector. Zones are coupled in the same auction, so they are **not pooled** into a single significance test or a single profit total.

**Compute:** fit time, inference time and hardware are logged for every model.

---

## 8. Leakage and fairness controls

- **One splitter** (`src/radar/data/splits.py`) decides what each model sees, and `tests/test_leakage.py` calls that real function.
- **Availability rules** for every variable, plus a vintage test once an `issued_at` field exists.
- **Frozen test window**, set before v0.4 and not used for tuning. SSM hyperparameters are tuned on a validation window with an embargo gap before the test period.
- **Same information, same preprocessing, same tuning budget** for all models.
- **Same dispatch code** for all forecasters and the oracle.
- **Chronos contamination checks:** Chronos-2 vs Chronos-2-synth, relative performance before and after the model's release date, and a perturbation test (rescale or shift the series). Contamination cannot be fully excluded and is reported as a limitation.
- **Battery invariant tests:** SoC within bounds, power limits respected, energy balance with losses, no simultaneous charge and discharge.
- Pinned seeds, a lockfile, pinned model revisions, and one config per experiment.

---

## 9. Repository structure

```
chronos-vs-ssm/
├── .env.example             # API key template (never commit .env)
├── README.md
├── MODEL_CARD.md
├── RESULTS.md               # running results and decisions log
├── pyproject.toml
├── configs/
│   ├── dk1.yaml             # one config per zone
│   ├── dk2.yaml
│   ├── de_lu.yaml
│   └── battery.yaml         # shared battery specification
├── data/
│   ├── raw/{zone}/          # untracked
│   └── processed/{zone}/
├── forecasts/{zone}/{model}/  # cached forecast samples
├── results/{zone}/
├── notebooks/
├── tests/
│   ├── test_leakage.py
│   └── test_battery.py
└── src/radar/
    ├── data/                # ENTSO-E client, validation, splitter
    ├── features/            # calendar, residual load, covariates
    ├── models/              # baselines, Chronos-2, Pyro SSM
    ├── scenarios/           # copula / Schaake shuffle
    ├── battery/             # LP, stochastic and CVaR dispatch, simulator
    ├── backtest/            # rolling-origin runner
    └── eval/                # CRPS, scores, profit, significance tests
```

Forecasting and trading are decoupled: models write forecasts to disk, and the simulator reads only those files.

---

## 10. Reproducing

*Planned interface (to be finalized in v0.1):*

```bash
cp .env.example .env            # add your ENTSO-E API token
pip install -e .
python -m radar.data.download zone=dk1
python -m radar.backtest.run zone=dk1 model=chronos2
python -m radar.eval.report zone=dk1
```

Switching zone means changing `zone=` only. If the code needs editing to add a zone, something was hard-coded.

Data comes from the ENTSO-E Transparency Platform. Raw data is not redistributed; follow ENTSO-E's terms of use.

---

## 11. Related work

This project builds on, and should be read next to, the following (the list reflects the works reviewed so far and is not an exhaustive survey):

- **Lipiecki & Weron (2026).** Foundation models for electricity price forecasting and battery arbitrage (Chronos-2, Moirai-2, TimesFM-2.5, TabPFN variants vs LEAR and DDNN benchmarks in Germany, Poland and Spain; arXiv 2609.00089).
- **Hirsch & Ziel (2026).** Probabilistic day-ahead forecasts, battery trading strategies and economic evaluation; decision quality under stochastic programs (arXiv 2604.19580).
- **Maciejowska, Lipiecki & Uniejewski (2025–26).** Which forecast properties matter for BESS arbitrage (arXiv 2511.13616).
- **Hornek et al. (2025).** Benchmarking pre-trained time series models for electricity price forecasting (arXiv 2506.08113).
- **Nitka & Weron (2023).** Does minimizing CRPS lead to optimal decisions in day-ahead bidding? (arXiv 2308.15443).
- **Lago et al. (2021).** LEAR and DNN benchmarks and the open-access `epftoolbox` (arXiv 2008.08004).
- **Marchesi et al. (2025)** and **Lettner et al. (2026)** on probabilistic zero-shot forecasting with Moirai and Chronos.
- Code references: [`jeslago/epftoolbox`](https://github.com/jeslago/epftoolbox), [`locuslab/e2e-model-learning`](https://github.com/locuslab/e2e-model-learning).

**How this project differs.** In the works reviewed, a Bayesian structural model built in Pyro is not one of the compared methods, DK1 and DK2 are not evaluated with this economic setup, and Chronos-2 quantiles are not turned into joint scenarios for CVaR dispatch. Novelty claims will be re-checked against a broader literature search before any write-up.

---

## 12. Limitations

- Price-taker, self-scheduling assumption; no order-book or market impact.
- One battery specification; results may differ for other durations and efficiencies.
- Day-ahead only. Intraday and balancing revenue are out of scope.
- Latest-vintage TSO forecasts rather than forecasts as issued at decision time.
- Test-period contamination of Chronos-2 cannot be fully ruled out.
- The three zones are coupled in one auction, so cross-zone agreement is weaker evidence than three independent markets.
- The native 15-minute period is short; the main benchmark uses hourly aggregation.

---

## 13. License and citation

License: *to be added* (a LICENSE file will be committed with v0.1).
Citation: *to be added once a version is tagged.*
