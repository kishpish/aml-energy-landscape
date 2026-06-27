# Phase 5 — Dynamical Characterization summary

- **n_states_total**: 79
- **states with computed biophysical fingerprints**: 46
- **rare states (from Phase 3)**: 5
- **states observed longitudinally**: 77
- **LSC subtypes discovered**: 2

## LSC subtypes

| lsc_subtype_id   |   n_cells |   n_real |   n_patients | dominant_basin   | dominant_state_id   |   G1_frac |   S_frac |   G2M_frac |   cycling_frac | phase_tag   |   LSC17_mean |   LSC6_mean |   D_trace_mean |
|:-----------------|----------:|---------:|-------------:|:-----------------|:--------------------|----------:|---------:|-----------:|---------------:|:------------|-------------:|------------:|---------------:|
| LSC_sub_0        |      1681 |     1681 |           17 | A1               | A1_L1_0             |     0.767 |    0.123 |      0.109 |          0.232 | mixed       |        0.309 |       0.645 |          2.28  |
| LSC_sub_1        |       180 |      180 |           13 | A1               | A1_L1_4             |     0.061 |    0.272 |      0.667 |          0.939 | cycling     |        0.31  |       0.592 |          2.733 |

## Top 5 most-persistent states (median log2 PR, post-treatment)

| state_id   | basin   |   persistence_log2_PR_median |   persistence_frac_enriched |
|:-----------|:--------|-----------------------------:|----------------------------:|
| A1_L1_16   | A1      |                      2.6716  |                       0.778 |
| A3_L1_7    | A3      |                      2.49225 |                       0.9   |
| A1_L1_1    | A1      |                      2.4526  |                       0.818 |
| A3_L1_1    | A3      |                      2.0844  |                       0.727 |
| A3_L1_5    | A3      |                      2.0093  |                       1     |

## Top 5 deepest-basin states

| state_id   | basin   |   basin_depth |   D_trace_mean |   committor_to_mature_monocyte |
|:-----------|:--------|--------------:|---------------:|-------------------------------:|
| A1_L1_2    | A1      |        14.104 |         14.925 |                          0.505 |
| A1_L1_11   | A1      |        13.853 |         16.439 |                          0.33  |
| A1_L1_13   | A1      |        11.958 |         14.9   |                          0.361 |
| A1_L1_12   | A1      |        11.784 |         16.895 |                          0.525 |
| A0_L1_13   | A0      |        10.386 |         15.225 |                          0.509 |

## Top 5 largest states

| state_id        | basin   |   n_cells_total | top_vangalen_type   |   malignant_frac |
|:----------------|:--------|----------------:|:--------------------|-----------------:|
| A1_basin_edge_0 | A1      |            7758 | Prog-like           |            0.082 |
| A0_L1_2         | A0      |            5994 | Mono-like           |            0.573 |
| A1_L1_1         | A1      |            5340 | T                   |            0.003 |
| A1_L1_0         | A1      |            3492 | Prog-like           |            0.601 |
| A1_L1_5         | A1      |            2814 | NK                  |            0.002 |
