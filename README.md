# Signal Extraction and Mean-Reversion Strategies in Noisy Financial Time Series

This project builds a compact statistical arbitrage research pipeline around a single idea: if a price spread is noisy, a state-space filter can estimate the latent equilibrium more cleanly than a naive rolling average. The repository compares a classic rolling z-score mean-reversion signal with a Kalman-filtered signal on a `V/MA` spread, includes transaction costs, preserves a train/test split, and exports reproducible diagnostic plots.

## Research Question

Can a latent-state estimate improve a mean-reversion strategy on a noisy relative-value spread?

- Pair: `V` vs `MA`
- Data: daily close prices from Stooq
- Start date: `2012-01-01`
- Split: first 60% train, final 40% test
- Costs: 5 bps per side

## Method

1. Estimate a static hedge ratio on the training sample.
2. Construct the residual spread `V - (alpha + beta * MA)`.
3. Compare two signals:
   - Naive baseline: rolling z-score of the spread.
   - Filtered signal: Kalman local-level estimate of the latent spread equilibrium, then z-score the deviation from that equilibrium.
4. Trade the spread:
   - Long spread when z-score is sufficiently negative.
   - Short spread when z-score is sufficiently positive.
   - Exit when the signal mean-reverts inside an exit band.
5. Evaluate Sharpe, drawdown, total return, and turnover out of sample.

## Project Layout

- `src/quant_project/data.py`: market data loader and pair alignment
- `src/quant_project/signals.py`: hedge ratio estimation, rolling-z signal, Kalman filter
- `src/quant_project/backtest.py`: transaction-cost-aware spread backtest
- `src/quant_project/research.py`: tuning, train/test split, and experiment orchestration
- `src/quant_project/plots.py`: publication-style output figures
- `scripts/run_research.py`: one-command reproduction of the study

## Run It

```bash
python3 scripts/run_research.py --asset-a V --asset-b MA --start-date 2012-01-01 --output-dir results/v_ma
```

The script writes:

- `summary.json`
- daily strategy frames for both models
- tuning grids
- signal, equity, and parameter-stability plots

## Results

Generated outputs live in `results/v_ma` and were produced from a run covering `2012-01-03` through `2026-03-20`, with the train set ending on `2020-07-10` and the out-of-sample test set beginning on `2020-07-13`.

The fitted spread is:

```text
V - (10.24 + 0.6071 * MA)
```

Best train-set parameters:

- Naive baseline: `window=10`, `entry_z=2.0`, `exit_z=0.5`
- Kalman filter: `process_var_multiplier=0.001`, `entry_z=2.0`, `exit_z=0.75`, `vol_span=20`

Out-of-sample comparison:

| Strategy | Test Sharpe | Test Max DD | Test Total Return | Test Trades |
| --- | ---: | ---: | ---: | ---: |
| Rolling z-score | -0.58 | -11.62% | -9.16% | 59 |
| Kalman filtered | 0.14 | -8.02% | 2.77% | 31 |

This is the core takeaway: the rolling-z baseline still looks acceptable on the training window, but it overfits and turns negative after July 2020. The Kalman-filtered signal stays profitable out of sample, cuts turnover almost in half, and suffers a smaller drawdown.

![Kalman signal vs spread](results/v_ma/signal_vs_spread.png)

![Equity curves](results/v_ma/equity_curves.png)

![Kalman parameter stability](results/v_ma/kalman_stability.png)

## Conclusions

- The latent-state estimate gives a more robust spread signal than a raw rolling average.
- The filtered strategy retains positive out-of-sample performance after costs when the naive baseline does not.
- Lower turnover matters: the Kalman model takes `31` test-set trades versus `59` for the naive signal.
- The parameter-stability plot shows that the Kalman setup is not driven by a single knife-edge parameter choice.
