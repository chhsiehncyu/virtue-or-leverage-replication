# Replication package

Replication code for the paper "Virtue or leverage? Dual-purpose backward integration and contract choice in OTA-hotel channels".

## Authorship

The code in this repository was prepared by the author with the assistance of Claude (Anthropic), used as an auxiliary tool. The author reviewed, verified, and approved every file before it was uploaded and is solely responsible for the code and its output. The assistant is not an author of the code or of the paper.

Requirements: Python 3.11 or later with the packages in `requirements.txt`. All scripts are run from this directory; later scripts read the pickle files written by earlier ones. `bash run_all.sh` runs every step below in order and writes one log per script to the folder `logs`. The file `replication_log.txt` contains the output of a complete run, for comparison.

## Run order and what each script reproduces

| Step | Script | Reproduces |
|---|---|---|
| 1 | `model.py` | Demand system and the closed-form merchant and agency equilibria (Lemmas 1 and 2); prints the baseline values of Table 4 |
| 2 | `endog_r.py` | First-order condition of the platform-chosen commission; writes `focr.pkl` |
| 3 | `quartic.py` | Quartic first-order condition of Section 5.1; writes `quartic.pkl` |
| 4 | `feasibility_bounds.py` | Lemma 2: demand of the integrated hotel, the bound on the commission at which it vanishes, and the sign of its difference from the incumbent's bound |
| 5 | `proof_concavity.py`, `proof_concavity_L.py` | Lemma 3: coefficient signs of the second derivative in the incumbent's effective cost |
| 6 | `proof_monotone_commission.py` | Lemma 3: exact Bernstein-coefficient certificate that the optimal commission decreases in the intensity; writes `bernstein_certificate_H.txt` and `bernstein_certificate_L.txt` |
| 7 | `proof_welfare_rigid.py` | Proposition 6: welfare slope under a rigid commission and the closed-form conditions of Section 6.4 |
| 8 | `grid_evaluation.py` | Evaluation of every equilibrium object over the grids described below; writes `v2_flex.pkl` and `v2_rigid.pkl` |
| 9 | `profit_gain_flex.py` | Section 5.2: intensity at which the platform's profit peaks under the flexible commission, size of the gain, and the intensity by which profit is back below its zero-intensity level; writes `flex_gain.pkl` |
| 10 | `observations.py` | All numerical observations of Sections 5 and 6 and the support table below |
| 11 | `baseline_numbers.py` | Baseline numbers quoted in Sections 5 to 7 and 10 (commissions, market shares, profit peak, welfare thresholds, tier-switch ranges, commission cap) |
| 12 | `check_reviewers.py`, `rigid_tables.py` | Baseline thresholds, Table 3, and Table 6; `rigid_tables.py` prints four decimals so that the two-decimal entries of Tables 3 and 6 can be rounded from its output |
| 13 | `tables_4_5.py` | Tables 4 and 5, printed with the same number of decimals as in the paper |
| 14 | `nonuniform2.py` | Table A1 (merchant equilibrium under the density f(theta) = 2 theta) |
| 15 | `extensions_check.py`, `continuous_quality_flex.py` | Numbers reported in Section 8 |
| 16 | `figures.py` | Figures 1 to 7, written to the folder `figures` |

Steps 5 to 7 use exact symbolic arithmetic. The two certificate files are included in this package, so the polynomial and every Bernstein coefficient used in the proof of Lemma 3 can be inspected without running the scripts; each file states the substitution, the degrees, and the sign pattern that the proof uses. Steps 8 to 10 take a few minutes on a laptop.

## Grids

Flexible commission (Section 5): h from 1.10 to 2.00 in steps of 0.05; l from 0.50 to 0.95 in steps of 0.05; c from 0.10 in steps of 0.025 up to 2/(1 + h), with one further point 0.005 below the bound; the intensity from 0 to 1 in steps of 0.02. This gives 5,510 parameter points, each evaluated at 51 intensities. At every point the optimal commission is the root of the quartic below the rate at which the incumbent's demand vanishes, and both hotels sell at that commission.

Rigid commission (Section 6): the same h, l, and intensities; c from 0.10 in steps of 0.025 up to min{4/(2h + 1), 2/(1 + l)}, with one further point 0.005 below the bound; r in {0.10, 0.15, 0.20, 0.25, 0.30}. This gives 34,610 combinations. A combination enters the high-end statistics if r < h[4 - c(2h + 1)]/2 and r < h/(h + 1) (31,799 combinations) and the economy statistics if r < [2 - c(1 + l)]/(2 - cl) (32,674 combinations); the tier-switch statistics require both (30,974 combinations).

The profit gain under the flexible commission (row F4 below) is located on a finer set of intensities, from 0 to 0.25 in steps of 0.0005, by `profit_gain_flex.py`.

## Support for each numerical observation

Support is the share of admissible points at which the observation holds at every intensity. The last column locates the exceptions by parameter values; the manuscript states each observation with the same region and exceptions.

| Id | Observation | Admissible points | Support | Exceptions |
|---|---|---|---|---|
| F1 | Optimal commission strictly decreases in the intensity (proved in Lemma 3) | 5,510 | 100.00% | none |
| F2 | Incumbent margin and profit strictly increase in the intensity, both tiers | 5,510 | 100.00% | none |
| F3 | Total welfare strictly increases in the intensity, both tiers | 5,510 | 100.00% | none |
| F4 | Platform profit peaks at an intensity of at most 0.065 with a gain of at most 0.083%, falls thereafter, and is below its zero-intensity level from an intensity of at most 0.126 | 5,510 | 100.00% | none |
| F5 | Optimal commission at zero intensity is at least 0.4 | 5,510 | 95.64% | only where c > 0.8 |
| F6 | Tier under flexible agency equals merchant tier at every intensity | 5,510 | 96.77% | only where the two merchant tiers are within 10% in objective value |
| F7 | Agency tier does not change with the intensity | 5,510 | 97.37% | same set as F6; switches in both directions |
| F8 | Platform objective and profit higher under agency at every intensity | 5,510 | 100.00% | none |
| F9 | Consumer surplus higher under merchant at every intensity (preferred tiers) | 5,510 | 100.00% | none |
| F10 | Incumbent profit higher under merchant at every intensity (preferred tiers) | 5,510 | 97.35% | only where the formats lead to different tiers; within a tier 100% |
| F11 | Consumer-surplus gap between formats widens in the intensity (preferred tiers) | 5,510 | 97.39% | only where the agency tier changes with the intensity |
| F12 | High-end: total-welfare ranking of formats crosses at most once, below 0.5 | 5,510 | 100.00% | none |
| F13 | Economy: total-welfare ranking of formats crosses at most once | 5,510 | 98.97% | c <= 0.15 with l >= 0.9 |
| R1 | Tier switch (high-end to economy): positioning gap crosses zero exactly once where sign conditions hold | 2,018 | 100.00% | none |
| R2 | Tier switch (high-end to economy): coverage larger under economy integration at every intensity | 2,018 | 100.00% | none |
| R3 | Consumer-surplus gap between formats crosses zero exactly once (high-end) | 31,799 | 99.97% | h = 1.1, r = 0.3, c at its upper bound |
| R4 | Consumer-surplus gap between formats crosses zero exactly once (economy) | 32,674 | 100.00% | none |
| R5 | Platform prefers merchant at zero intensity (high-end) | 31,799 | 96.03% | r >= 0.25 and c > 0.65 |
| R6 | Platform prefers merchant at zero intensity (economy) | 32,674 | 99.38% | r >= 0.25 and c > 0.8 |
| R7 | Platform prefers agency at some intensity (high-end) | 31,799 | 99.34% | c <= 0.2 with h <= 1.2 |
| R8 | Platform prefers agency at some intensity (economy) | 32,674 | 99.97% | l = 0.95, c = 0.95, r = 0.1 |
| R9 | Welfare decline from peak to parity below 2% (both tiers) | 64,473 | 100.00% | none |
| R10 | Reverse tier switch (economy to high-end): positioning gap crosses zero exactly once where it is negative at zero intensity and positive at parity | 2,017 | 100.00% | none |
| R11 | Reverse tier switch: consumer surplus at parity higher under high-end integration | 2,017 | 99.21% | only at the upper end of the quality-cost range of the switch, c >= 0.65 with r >= 0.15 |
| R12 | Reverse tier switch: coverage at parity larger under economy integration | 2,017 | 95.34% | h <= 1.15 with c <= 0.225 |
