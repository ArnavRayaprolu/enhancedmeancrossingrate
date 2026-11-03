# Revised MCR model: walk-forward results (protocol R1-2026-11)

Data: Alpaca US equities, 1-hour bars (09:00-15:00 ET), split and dividend adjusted, 2016-03-31 to 2026-10-07 out of sample. 58 walk-forward windows (90 formation days, 45 trading days). All parameters were fixed in mcr/config.py before any result was seen.

Universe per window: top 250 liquid common stocks chosen on formation data (median 250 names, 31,125 candidate pairs, 6 names per window that are now delisted). Median estimated one-way cost 3.1 bps.

## Headline (primary configuration, base costs)

| criterion | total_return | cagr | ann_vol | sharpe | sortino | max_drawdown | time_in_market | avg_exposure | turnover_pa | breakeven_arith_bps | breakeven_comp_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MCR | -0.478 | -0.061 | 0.084 | -0.704 | -0.999 | -0.541 | 1.000 | 0.974 | 66.181 |  |  |
| Cointegration | -0.318 | -0.036 | 0.085 | -0.392 | -0.560 | -0.465 | 1.000 | 0.959 | 66.034 | 0.741 | 0.191 |
| Distance | -0.315 | -0.036 | 0.039 | -0.914 | -1.250 | -0.343 | 1.000 | 0.973 | 65.251 |  |  |
| Correlation | -0.191 | -0.020 | 0.036 | -0.559 | -0.797 | -0.241 | 1.000 | 0.976 | 68.183 | 3.446 | 3.354 |
| HalfLife | -0.162 | -0.017 | 0.069 | -0.214 | -0.305 | -0.374 | 1.000 | 0.953 | 65.700 | 3.786 | 3.424 |
| MCR+Cointegration | -0.374 | -0.044 | 0.077 | -0.551 | -0.785 | -0.454 | 1.000 | 0.961 | 65.370 |  |  |
| MCR+Distance | -0.220 | -0.024 | 0.054 | -0.421 | -0.603 | -0.311 | 1.000 | 0.977 | 66.579 | 1.882 | 1.666 |
| MCR+Correlation | -0.313 | -0.036 | 0.057 | -0.614 | -0.869 | -0.355 | 1.000 | 0.981 | 66.951 | 0.985 | 0.746 |
| MCR+HalfLife | -0.463 | -0.058 | 0.074 | -0.779 | -1.103 | -0.561 | 1.000 | 0.971 | 65.906 |  |  |

## R1 - What MCR measures

MCR is the share of formation bars on which z = (MA_35 - MA_140)/SD_140 of the price ratio changes sign, i.e. the fast and slow moving averages of the ratio cross. It is not the ratio crossing its own mean. Fold-average Spearman correlations over a random sample of 50,000 candidate pairs per window:

| y | mean_rho | t | share_positive |
|---|---|---|---|
| ar1_z | -0.126 | -13.879 | 0.052 |
| coint_p | -0.154 | -36.912 | 0.000 |
| corr | 0.025 | 4.505 | 0.759 |
| cycles | 0.398 | 58.794 | 1.000 |
| distance | -0.243 | -27.530 | 0.000 |
| halflife | -0.377 | -73.289 | 0.000 |
| oos_mcr | -0.012 | -3.454 | 0.328 |
| oos_net | -0.005 | -0.898 | 0.466 |
| oos_trades | -0.007 | -1.727 | 0.431 |
| xr_coint | 0.137 | 45.394 | 1.000 |
| xr_ratio | 0.274 | 50.127 | 1.000 |

Every formation statistic vs the pair's own out-of-sample net return (oriented so positive = the statistic picks better pairs):

| x | mean_rho | t | share_positive |
|---|---|---|---|
| ar1_z | 0.010 | 1.234 | 0.517 |
| cycles | -0.003 | -0.565 | 0.466 |
| mcr | -0.005 | -0.898 | 0.466 |
| xr_coint | -0.005 | -1.317 | 0.431 |
| corr | -0.013 | -1.399 | 0.431 |
| xr_ratio | -0.011 | -1.851 | 0.397 |
| coint_p | -0.010 | -2.135 | 0.379 |
| distance | -0.023 | -2.152 | 0.397 |
| halflife | -0.019 | -2.869 | 0.397 |

## R2 / R9 - Specification search, data snooping and deflated Sharpe

Trials: 198 strategies (22 configurations x 9 criteria), every one reported in results/performance.csv.

| scope | benchmark | costs | spa_p | rc_p | spa_lower_p | K | T | block |
|---|---|---|---|---|---|---|---|---|
| primary_config | primary Cointegration | base | 0.640 | 0.640 | 0.504 | 5 | 18218 | 3.397 |
| all_configs | primary Cointegration | base | 0.538 | 0.538 | 0.399 | 110 | 18218 | 3.359 |

| label | config | criterion | costs | sharpe_ann | sr0_ann | dsr | psr | n_trials | T |
|---|---|---|---|---|---|---|---|---|---|
| primary champion | primary | MCR+Cointegration | base | -0.551 | 0.812 | 0.000 | 0.039 | 198 | 18218 |
| best trial | hedged_spread | MCR | base | 0.162 | 0.812 | 0.018 | 0.699 | 198 | 18218 |

Bootstrap 95% intervals (stationary bootstrap, Politis-White block):

| config | criterion | costs | sharpe_lo | sharpe_hi | ret_lo | ret_hi | block |
|---|---|---|---|---|---|---|---|
| primary | MCR | base | -1.343 | -0.110 | -0.699 | -0.121 | 1.294 |
| primary | Cointegration | base | -1.004 | 0.199 | -0.605 | 0.147 | 1.949 |
| primary | Distance | base | -1.502 | -0.296 | -0.468 | -0.120 | 2.707 |
| primary | Correlation | base | -1.105 | 0.026 | -0.337 | 0.003 | 1.000 |
| primary | HalfLife | base | -0.843 | 0.412 | -0.469 | 0.309 | 1.407 |
| primary | MCR+Cointegration | base | -1.175 | 0.078 | -0.620 | 0.032 | 1.000 |
| primary | MCR+Distance | base | -1.077 | 0.221 | -0.455 | 0.115 | 2.381 |
| primary | MCR+Correlation | base | -1.242 | 0.024 | -0.524 | -0.003 | 2.410 |
| primary | MCR+HalfLife | base | -1.416 | -0.193 | -0.668 | -0.160 | 1.376 |

## R3 - Diebold-Mariano tests (primary configuration)

d_t = n^A_t - n^B_t (net return per bar); loss = -net return; H0: E[d] = 0 vs two-sided. Columns: HAC lags = floor(4(T/100)^(2/9)) (nw), 2x, 4x, Andrews (1991), one day (7); the test on daily sums of d; stationary-bootstrap p (null imposed); Holm within family and across all.

| costs | family | A | B | mean_diff_pa | dm_nw | p_nw | p_nw2 | p_nw4 | p_andrews | p_7 | p_daily | p_boot | holm_family | holm_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | champion | MCR+Cointegration | Cointegration | -0.0099 | -0.4026 | 0.6873 | 0.6836 | 0.6811 | 0.6898 | 0.6889 | 0.6794 | 0.6837 | 1.0000 | 1.0000 |
| original | champion | MCR+Cointegration | Distance | -0.0018 | -0.0773 | 0.9384 | 0.9378 | 0.9378 | 0.9393 | 0.9388 | 0.9373 | 0.9309 | 1.0000 | 1.0000 |
| original | champion | MCR+Cointegration | Correlation | -0.0282 | -1.1693 | 0.2423 | 0.2362 | 0.2323 | 0.2563 | 0.2485 | 0.2291 | 0.2693 | 0.7269 | 1.0000 |
| original | hybrid_vs_component | MCR+Cointegration | Cointegration | -0.0099 | -0.4026 | 0.6873 | 0.6836 | 0.6811 | 0.6898 | 0.6889 | 0.6794 | 0.6837 | 0.9508 | 1.0000 |
| original | hybrid_vs_component | MCR+Distance | Distance | 0.0157 | 1.0008 | 0.3169 | 0.3110 | 0.2993 | 0.3330 | 0.3282 | 0.2908 | 0.3504 | 0.9508 | 1.0000 |
| original | hybrid_vs_component | MCR+Correlation | Correlation | -0.0167 | -0.9038 | 0.3661 | 0.3613 | 0.3449 | 0.3800 | 0.3726 | 0.3401 | 0.3664 | 0.9508 | 1.0000 |
| original | hybrid_vs_component | MCR+HalfLife | HalfLife | -0.0428 | -2.0585 | 0.0396 | 0.0388 | 0.0379 | 0.0408 | 0.0395 | 0.0379 | 0.0390 | 0.1582 | 0.4351 |
| original | mcr_vs_classic | MCR | Cointegration | -0.0260 | -0.9655 | 0.3343 | 0.3338 | 0.3346 | 0.3374 | 0.3364 | 0.3326 | 0.3483 | 0.6686 | 1.0000 |
| original | mcr_vs_classic | MCR | Distance | -0.0179 | -0.7242 | 0.4690 | 0.4633 | 0.4369 | 0.4798 | 0.4758 | 0.4319 | 0.4835 | 0.6686 | 1.0000 |
| original | mcr_vs_classic | MCR | Correlation | -0.0443 | -1.6616 | 0.0966 | 0.0899 | 0.0689 | 0.1007 | 0.0985 | 0.0655 | 0.1111 | 0.2898 | 0.8694 |
| original | mcr_vs_classic | MCR | HalfLife | -0.0460 | -1.7993 | 0.0720 | 0.0735 | 0.0692 | 0.0742 | 0.0726 | 0.0695 | 0.0821 | 0.2879 | 0.7198 |
| base | champion | MCR+Cointegration | Cointegration | -0.0090 | -0.3642 | 0.7157 | 0.7119 | 0.7089 | 0.7183 | 0.7173 | 0.7073 | 0.7197 | 1.0000 | 1.0000 |
| base | champion | MCR+Cointegration | Distance | -0.0066 | -0.2783 | 0.7807 | 0.7783 | 0.7779 | 0.7841 | 0.7821 | 0.7761 | 0.7768 | 1.0000 | 1.0000 |
| base | champion | MCR+Cointegration | Correlation | -0.0224 | -0.9300 | 0.3524 | 0.3455 | 0.3403 | 0.3672 | 0.3590 | 0.3364 | 0.3874 | 1.0000 | 1.0000 |
| base | hybrid_vs_component | MCR+Cointegration | Cointegration | -0.0090 | -0.3642 | 0.7157 | 0.7119 | 0.7089 | 0.7183 | 0.7173 | 0.7073 | 0.7197 | 1.0000 | 1.0000 |
| base | hybrid_vs_component | MCR+Distance | Distance | 0.0132 | 0.8416 | 0.4000 | 0.3930 | 0.3792 | 0.4157 | 0.4110 | 0.3694 | 0.4354 | 1.0000 | 1.0000 |
| base | hybrid_vs_component | MCR+Correlation | Correlation | -0.0148 | -0.8018 | 0.4227 | 0.4165 | 0.3976 | 0.4366 | 0.4294 | 0.3915 | 0.4264 | 1.0000 | 1.0000 |
| base | hybrid_vs_component | MCR+HalfLife | HalfLife | -0.0427 | -2.0563 | 0.0398 | 0.0388 | 0.0372 | 0.0412 | 0.0398 | 0.0369 | 0.0380 | 0.1591 | 0.4375 |
| base | mcr_vs_classic | MCR | Cointegration | -0.0260 | -0.9660 | 0.3341 | 0.3328 | 0.3327 | 0.3376 | 0.3366 | 0.3306 | 0.3493 | 0.6681 | 1.0000 |
| base | mcr_vs_classic | MCR | Distance | -0.0236 | -0.9531 | 0.3405 | 0.3339 | 0.3047 | 0.3525 | 0.3481 | 0.2990 | 0.3473 | 0.6681 | 1.0000 |
| base | mcr_vs_classic | MCR | Correlation | -0.0395 | -1.4797 | 0.1390 | 0.1306 | 0.1040 | 0.1442 | 0.1416 | 0.0994 | 0.1562 | 0.4169 | 1.0000 |
| base | mcr_vs_classic | MCR | HalfLife | -0.0446 | -1.7466 | 0.0807 | 0.0820 | 0.0767 | 0.0833 | 0.0816 | 0.0767 | 0.0881 | 0.3229 | 0.8072 |
| stress | champion | MCR+Cointegration | Cointegration | -0.0073 | -0.2964 | 0.7669 | 0.7630 | 0.7592 | 0.7695 | 0.7685 | 0.7575 | 0.7668 | 1.0000 | 1.0000 |
| stress | champion | MCR+Cointegration | Distance | -0.0182 | -0.7706 | 0.4410 | 0.4343 | 0.4308 | 0.4492 | 0.4445 | 0.4257 | 0.4434 | 1.0000 | 1.0000 |
| stress | champion | MCR+Cointegration | Correlation | -0.0159 | -0.6574 | 0.5109 | 0.5034 | 0.4962 | 0.5249 | 0.5174 | 0.4916 | 0.5465 | 1.0000 | 1.0000 |
| stress | hybrid_vs_component | MCR+Cointegration | Cointegration | -0.0073 | -0.2964 | 0.7669 | 0.7630 | 0.7592 | 0.7695 | 0.7685 | 0.7575 | 0.7668 | 1.0000 | 1.0000 |
| stress | hybrid_vs_component | MCR+Distance | Distance | 0.0088 | 0.5609 | 0.5749 | 0.5674 | 0.5523 | 0.5880 | 0.5842 | 0.5420 | 0.5926 | 1.0000 | 1.0000 |
| stress | hybrid_vs_component | MCR+Correlation | Correlation | -0.0123 | -0.6604 | 0.5090 | 0.5013 | 0.4798 | 0.5222 | 0.5157 | 0.4720 | 0.5375 | 1.0000 | 1.0000 |
| stress | hybrid_vs_component | MCR+HalfLife | HalfLife | -0.0435 | -2.0886 | 0.0368 | 0.0355 | 0.0327 | 0.0385 | 0.0370 | 0.0319 | 0.0390 | 0.1470 | 0.4043 |
| stress | mcr_vs_classic | MCR | Cointegration | -0.0255 | -0.9480 | 0.3431 | 0.3402 | 0.3378 | 0.3476 | 0.3464 | 0.3350 | 0.3594 | 0.4270 | 1.0000 |
| stress | mcr_vs_classic | MCR | Distance | -0.0364 | -1.4673 | 0.1423 | 0.1360 | 0.1112 | 0.1527 | 0.1490 | 0.1061 | 0.1431 | 0.4270 | 1.0000 |
| stress | mcr_vs_classic | MCR | Correlation | -0.0341 | -1.2754 | 0.2022 | 0.1914 | 0.1576 | 0.2089 | 0.2058 | 0.1513 | 0.2102 | 0.4270 | 1.0000 |
| stress | mcr_vs_classic | MCR | HalfLife | -0.0449 | -1.7533 | 0.0796 | 0.0800 | 0.0731 | 0.0826 | 0.0807 | 0.0723 | 0.0901 | 0.3183 | 0.7956 |

## R4 - Engine

Rules in mcr/engine.py (docstring = pseudocode). Effect of the rule changes: compare configs 'primary' and 'engine_original' below.

## R5 - Costs and break-even

| criterion | sharpe_base | sharpe_original | sharpe_stress | total_return_base | total_return_original | total_return_stress | breakeven_arith_bps | breakeven_comp_bps |
|---|---|---|---|---|---|---|---|---|
| MCR | -0.704 | -0.339 | -1.444 | -0.478 | -0.283 | -0.727 |  |  |
| Cointegration | -0.392 | -0.031 | -1.130 | -0.318 | -0.062 | -0.645 | 0.741 | 0.191 |
| Distance | -0.914 | -0.274 | -2.165 | -0.315 | -0.112 | -0.591 |  |  |
| Correlation | -0.559 | 0.443 | -2.413 | -0.191 | 0.169 | -0.600 | 3.446 | 3.354 |
| HalfLife | -0.214 | 0.252 | -1.115 | -0.162 | 0.168 | -0.561 | 3.786 | 3.424 |
| MCR+Cointegration | -0.551 | -0.163 | -1.345 | -0.374 | -0.148 | -0.669 |  |  |
| MCR+Distance | -0.421 | 0.093 | -1.420 | -0.220 | 0.037 | -0.555 | 1.882 | 1.666 |
| MCR+Correlation | -0.614 | -0.018 | -1.756 | -0.313 | -0.026 | -0.651 | 0.985 | 0.746 |
| MCR+HalfLife | -0.779 | -0.344 | -1.628 | -0.463 | -0.252 | -0.721 |  |  |

'original' = 1.5 bps per dollar traded (the paper's 3 bps round trip); 'base' = each stock's Abdi-Ranaldo half-spread + 0.5 bp fees + 0.5%/yr borrow on the short leg; 'stress' = 2x spread + square-root impact for a $10m book + 2%/yr borrow. Break-even is the one-way cost per dollar traded, by sum of returns and by compounded wealth.

## R6 - Sample, regimes, survivorship, missing data

| criterion | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 2016-19 expansion | 2020 covid | 2021 rally | 2022 bear | 2023-26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cointegration | 1.41 | -0.30 | 0.38 | -0.99 | -1.75 | -0.72 | 0.00 | -0.68 | -2.34 | 0.59 | 0.76 | -0.01 | -1.75 | -0.72 | 0.00 | -0.26 |
| Correlation | -1.26 | 0.79 | -1.68 | -0.05 | -1.49 | -1.93 | 0.48 | -1.62 | 0.54 | -1.26 | 1.30 | -0.47 | -1.49 | -1.93 | 0.48 | -0.22 |
| Distance | -0.22 | -2.22 | -0.75 | 0.38 | -1.58 | -1.23 | -0.51 | -0.80 | -0.39 | -1.09 | -1.36 | -0.66 | -1.58 | -1.23 | -0.51 | -0.93 |
| HalfLife | 1.36 | -0.73 | 1.75 | 1.44 | -1.69 | 0.21 | -0.48 | -0.67 | -0.21 | -0.39 | -0.49 | 0.99 | -1.69 | 0.21 | -0.48 | -0.42 |
| MCR | 1.73 | -0.74 | -1.12 | -1.39 | -2.48 | -0.54 | -0.05 | -0.88 | -1.69 | 0.43 | -0.30 | -0.60 | -2.48 | -0.54 | -0.05 | -0.52 |
| MCR+Cointegration | -0.85 | -0.41 | 0.08 | -0.99 | -1.77 | -0.65 | 0.01 | -0.37 | -1.06 | -0.34 | 0.08 | -0.51 | -1.77 | -0.65 | 0.01 | -0.39 |
| MCR+Correlation | -0.57 | 0.15 | -0.68 | -0.10 | -0.89 | 0.06 | -0.09 | -2.36 | -0.54 | -1.37 | -0.41 | -0.28 | -0.89 | 0.06 | -0.09 | -1.06 |
| MCR+Distance | 0.71 | -1.00 | -1.46 | -0.49 | -1.34 | -0.09 | -0.71 | -0.10 | 0.31 | 0.22 | 0.04 | -0.75 | -1.34 | -0.09 | -0.71 | 0.12 |
| MCR+HalfLife | -0.10 | -0.11 | 0.35 | -0.23 | -2.42 | -0.88 | -0.23 | -1.87 | -2.04 | -1.20 | 0.86 | -0.01 | -2.42 | -0.88 | -0.23 | -0.89 |

## R7 / R8 - Ratio vs hedged spread; one-at-a-time robustness (Sharpe, base costs)

| config | MCR | Cointegration | Distance | Correlation | HalfLife | MCR+Cointegration | MCR+Distance | MCR+Correlation | MCR+HalfLife | champion_rank | champion_minus_coint | mcr_minus_best_classic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complete_case | -0.61 | -0.52 | -0.74 | -0.80 | 0.01 | -0.60 | -0.34 | -0.43 | -0.55 | 6.00 | -0.08 | -0.62 |
| delay_1bar | -0.72 | -0.49 | -1.01 | -0.62 | -0.22 | -0.51 | -0.40 | -0.62 | -0.72 | 4.00 | -0.02 | -0.50 |
| engine_original | -0.72 | -0.53 | -1.08 | -0.63 | -0.27 | -0.54 | -0.47 | -0.62 | -0.79 | 4.00 | -0.01 | -0.45 |
| entry_0.75 | -0.60 | -0.51 | -1.18 | -0.88 | -0.26 | -0.43 | -0.52 | -0.77 | -0.76 | 2.00 | 0.08 | -0.34 |
| entry_1.25 | -0.57 | -0.52 | -0.83 | -0.23 | -0.27 | -0.40 | -0.08 | -0.42 | -0.50 | 4.00 | 0.12 | -0.34 |
| entry_1.50 | -0.45 | -0.21 | -0.29 | -0.05 | -0.21 | -0.22 | -0.07 | -0.28 | -0.46 | 5.00 | -0.01 | -0.40 |
| exit_0.00 | -0.74 | -0.47 | -0.95 | -0.46 | -0.06 | -0.51 | -0.27 | -0.64 | -0.67 | 5.00 | -0.04 | -0.68 |
| exit_0.50 | -0.68 | -0.35 | -1.02 | -0.56 | -0.21 | -0.54 | -0.52 | -0.74 | -0.75 | 4.00 | -0.20 | -0.47 |
| hedged_spread | 0.16 | -0.38 | -1.06 | -0.47 | -0.24 | -0.26 | -0.61 | -0.51 | -0.35 | 3.00 | 0.12 | 0.40 |
| ma_14_70 | -0.72 | -0.46 | -1.26 | -1.73 | -0.64 | -0.40 | -1.15 | -0.72 | -0.85 | 1.00 | 0.07 | -0.26 |
| ma_70_280 | -0.58 | -0.65 | -0.46 | -0.07 | -0.31 | -0.59 | -0.24 | -0.37 | -0.50 | 8.00 | 0.06 | -0.51 |
| original | -0.76 | -0.49 | -0.45 | -0.44 | -0.62 | -0.43 | -0.16 | -0.59 | -0.34 | 3.00 | 0.06 | -0.32 |
| pairs_10 | -0.43 | -0.31 | -1.36 | -0.85 | 0.04 | -0.28 | -0.55 | -0.47 | -0.28 | 2.00 | 0.03 | -0.47 |
| pairs_40 | -0.66 | -0.50 | -0.87 | -1.15 | -0.49 | -0.90 | -0.67 | -0.66 | -0.93 | 7.00 | -0.40 | -0.17 |
| pairs_5 | -0.58 | 0.03 | -1.13 | -0.52 | 0.03 | -0.52 | -0.44 | -0.14 | -0.60 | 5.00 | -0.54 | -0.61 |
| primary | -0.70 | -0.39 | -0.91 | -0.56 | -0.21 | -0.55 | -0.42 | -0.61 | -0.78 | 4.00 | -0.16 | -0.49 |
| sp500_sector | -0.29 | -0.37 | -1.12 | -0.77 | -1.07 | -0.58 | -0.53 | -0.79 | -0.87 | 4.00 | -0.21 | 0.08 |
| universe_125 | -0.42 | -0.39 | -1.18 | -0.83 | -0.03 | -0.79 | -0.68 | -0.39 | -0.56 | 7.00 | -0.40 | -0.39 |
| w_active_eq8 | -0.73 | -0.39 | -0.97 | -0.63 | -0.40 | -0.57 | -0.51 | -0.77 | -0.87 | 4.00 | -0.18 | -0.34 |
| w_equal | -0.69 | -0.30 | -0.79 | -0.24 | -0.06 | -0.41 | -0.19 | -0.36 | -0.62 | 6.00 | -0.10 | -0.63 |
| w_rank_fixed | -0.69 | -0.30 | -0.79 | -0.24 | -0.06 | -0.41 | -0.19 | -0.36 | -0.62 | 6.00 | -0.10 | -0.63 |
| windows_60_21 | -0.50 | -0.49 | -0.37 | -1.35 | -1.09 | -0.70 | -0.56 | -0.76 | -0.70 | 6.00 | -0.20 | -0.13 |

Note: the selected pairs are the top 20 of ~31,000 candidates, so their percentile scores are all within ~0.001 of 1. Rank weights (Eq. 8 and 'rank') are therefore almost equal weights; 'w_rank_fixed' and 'w_equal' coincide to two decimals.


## m4 / m7 - Composition and concentration of the books (primary)

| criterion | median_mcr | median_coint_p | median_halflife_bars | median_corr | median_cost_bps | time_in_trade | trades_per_pair | jaccard_with_champion |
|---|---|---|---|---|---|---|---|---|
| Cointegration | 0.012 | 0.001 | 27.342 | 0.283 | 2.770 | 0.621 | 2.368 | 0.068 |
| Correlation | 0.008 | 0.395 | 69.480 | 0.850 | 3.792 | 0.623 | 2.414 | 0.010 |
| Distance | 0.012 | 0.071 | 20.251 | 0.576 | 3.562 | 0.628 | 2.351 | 0.022 |
| HalfLife | 0.016 | 0.004 | 12.221 | 0.322 | 4.060 | 0.617 | 2.366 | 0.113 |
| MCR | 0.027 | 0.203 | 39.757 | 0.266 | 2.937 | 0.634 | 2.363 | 0.065 |
| MCR+Cointegration | 0.021 | 0.005 | 17.573 | 0.294 | 2.530 | 0.629 | 2.377 | 1.000 |
| MCR+Correlation | 0.020 | 0.187 | 38.098 | 0.650 | 3.257 | 0.637 | 2.372 | 0.041 |
| MCR+Distance | 0.020 | 0.065 | 21.640 | 0.365 | 3.600 | 0.637 | 2.373 | 0.096 |
| MCR+HalfLife | 0.022 | 0.015 | 15.708 | 0.299 | 3.603 | 0.639 | 2.376 | 0.289 |

| criterion | max_pair_share | hhi_pairs | effective_pairs | max_sector_share | top_pair_pnl_share | trades | pairs | same_sector | avg_exposure | time_invested |
|---|---|---|---|---|---|---|---|---|---|---|
| Cointegration | 0.075 | 0.055 | 18.301 | 0.226 | 0.184 | 47.362 | 20.000 | 0.168 | 0.959 | 1.000 |
| Correlation | 0.076 | 0.055 | 18.195 | 0.276 | 0.194 | 48.276 | 20.000 | 0.910 | 0.976 | 1.000 |
| Distance | 0.075 | 0.055 | 18.225 | 0.234 | 0.172 | 47.017 | 20.000 | 0.495 | 0.973 | 1.000 |
| HalfLife | 0.076 | 0.055 | 18.266 | 0.217 | 0.167 | 47.310 | 20.000 | 0.204 | 0.953 | 1.000 |
| MCR | 0.074 | 0.054 | 18.391 | 0.220 | 0.171 | 47.259 | 20.000 | 0.142 | 0.974 | 1.000 |
| MCR+Cointegration | 0.076 | 0.055 | 18.293 | 0.217 | 0.180 | 47.534 | 20.000 | 0.143 | 0.961 | 1.000 |
| MCR+Correlation | 0.074 | 0.055 | 18.314 | 0.288 | 0.189 | 47.431 | 20.000 | 0.568 | 0.981 | 1.000 |
| MCR+Distance | 0.074 | 0.055 | 18.356 | 0.218 | 0.164 | 47.466 | 20.000 | 0.222 | 0.977 | 1.000 |
| MCR+HalfLife | 0.075 | 0.054 | 18.369 | 0.219 | 0.172 | 47.517 | 20.000 | 0.149 | 0.971 | 1.000 |
