# Forecast Scenario Test Battery

> 122 tests passing across 32 real-world scenarios + original test suites.
> All scenarios use the full `run_forecast()` pipeline end-to-end.

## Scenario Index

| # | Scenario | Frequency | History | Horizon | Key Test |
|---|----------|-----------|---------|---------|----------|
| 1 | Monthly revenue (trend + seasonality) | ME | 36 mo | 6 | Trend capture, seasonal decomposition, backtest |
| 2 | Flat noisy cost center | ME | 24 mo | 3 | Mean-reversion, noise tolerance |
| 3 | Short history (5 months) | ME | 5 mo | 3 | Graceful degradation, baseline fallback |
| 4 | Minimal 2-point series | ME | 2 mo | 2 | Absolute minimum viable forecast |
| 5 | Multi-series mixed history | ME | 6-36 mo | 3 | Panel with unequal lengths, per-series selection |
| 6 | Daily weekly seasonality | D | 90 d | 14 | Day-of-week pattern, daily frequency |
| 7 | Business day frequency | B | 60 bd | 10 | Weekday-only calendar, season=5 |
| 8 | Weekly data | W-SUN | 52 wk | 8 | Weekly aggregation, trend |
| 9 | Quarterly fiscal | QE | 20 q | 4 | Quarterly seasonality, low obs count |
| 10 | Missing internal timestamps | ME | 21 mo (3 gaps) | 3 | Gap detection, interpolation repair |
| 11 | Intermittent demand | ME | 24 mo | 3 | Sparse zeros, ZeroForecast candidate |
| 12 | Negative values (costs) | ME | 24 mo | 3 | COGS handling, negative forecasts |
| 13 | Level shift (acquisition) | ME | 36 mo | 6 | Structural break detection |
| 14 | Exponential growth | ME | 30 mo | 6 | Compound growth, non-linear trend |
| 15 | Declining trend | ME | 30 mo | 6 | Downward trend capture |
| 16 | High noise / low signal | ME | 36 mo | 3 | Noise >> signal tolerance |
| 17 | Event/driver overlay | ME | 24 mo | 6 | DriverEvent multiplicative scenario |
| 18 | Prediction intervals | ME | 48 mo | 6 | 80%/95% conformal intervals |
| 19 | Multi-series panel (10) | ME | 24-48 mo | 3 | utilsforecast panel generation |
| 20 | Large panel (50 series) | ME | 24-36 mo | 3 | Scalability, 50-series batch |
| 21 | Constant series | ME | 12 mo | 3 | Zero variance edge case |
| 22 | Outlier in training | ME | 24 mo | 3 | One-time spike robustness |
| 23 | Large values (millions) | ME | 24 mo | 6 | Float precision at scale |
| 24 | Tiny values (fractional) | ME | 24 mo | 3 | Float precision at micro-scale |
| 25 | No weighted ensemble | ME | 24 mo | 3 | Single-model selection |
| 26 | Profile-only (no forecast) | ME | 36 mo | n/a | Data quality assessment |
| 27 | Baseline-only policy | ME | 18 mo | 3 | Conservative model restriction |
| 28a-c | Fill methods (ffill/interpolate/zero) | ME | 22 mo (gaps) | 3 | Gap repair strategies |
| 29 | Full interpretation artifacts | ME | 36 mo | 6 | Backtest windows, seasonality, naive comparison |
| 30 | Manifest completeness | ME | 24 mo | 3 | All expected output files present |

## Scenario Details

### Scenario 1: Monthly Revenue (Trend + Seasonality)
```
y = 100 + 3.5t + 15*sin(2*pi*t/12) + N(0,3), t = 0..35
```
The bread-and-butter finance use case. Tests frequency detection (ME), season length (12), backtest metrics, and seasonality interpretation.

### Scenario 5: Multi-Series Mixed History
```
Mature:  100 + 2t + N(0,5),   36 months from 2023-01
Medium:   50 + 1.5t + N(0,3), 18 months from 2024-07
New:      20 + N(0,2),         6 months from 2025-07
```
Real finance has mixed-maturity product lines. Tests per-series model selection across different history lengths.

### Scenario 11: Intermittent Demand
```
y = [0, 0, 150, 0, 0, 0, 200, 0, 0, 80, 0, 0, 0, 0, 300, 0, 0, 0, 120, 0, 0, 0, 0, 250]
```
Sporadic deal closings. Tests ZeroForecast candidate availability and sparse data handling.

### Scenario 13: Level Shift (Acquisition)
```
y = 100 + 2t + I(t>=18)*80 + N(0,5), t = 0..35
```
M&A and pricing changes create permanent level changes. Tests post-shift forecast captures the new level.

### Scenario 17: Event/Driver Overlay
```
Base: 200 + 5t + 20*sin(2*pi*t/12) + N(0,5)
Event: "Q1 Launch", 2025-01 to 2025-03, multiplicative +15%
```
Finance users overlay known future events. Tests yhat_scenario column and scenario adjustments.

### Scenario 20: Large Panel (50 Series)
```python
utilsforecast.data.generate_series(n_series=50, freq="ME", min_length=24, max_length=36, seed=7)
```
Product catalog or SKU-level forecasting. Tests pipeline scalability.

## Data Sources

| Source | Usage |
|--------|-------|
| `numpy.random.default_rng(seed)` | Reproducible synthetic patterns with known properties |
| `utilsforecast.data.generate_series()` | Nixtla's data generator for realistic multi-series panels |
| Hand-crafted edge cases | Zeros, negatives, outliers, gaps, tiny/large values |

## Research-Informed Design

These scenarios incorporate 26 verified knowledge facts from 10 autoresearch iterations:

1. **Intermittent demand** (Scenario 11): Croston models don't help on Bernoulli-zero patterns; ZeroForecast is WAPE-optimal when >50% zeros
2. **Level shift** (Scenario 13): Distribution shift causes 15-21 holdout underperformances; James-Stein shrinkage addresses this
3. **Mixed history** (Scenario 5): Short-history series need fallback selection, not crashed pipelines
4. **Fill methods** (Scenario 28): Gap repair is critical for real data quality
5. **Naive comparison** (Scenario 29): Naive guard never triggers because backtest already selects models beating naive; holdout underperformance is distribution shift

## Running

```bash
uv run --extra dev pytest -q                                    # All 122 tests
uv run --extra dev pytest tests/test_real_scenarios.py -v       # 32 real-world scenarios
uv run --extra dev pytest tests/test_real_scenarios.py -k daily # Specific pattern
```
