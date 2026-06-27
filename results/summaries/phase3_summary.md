# Phase 3 — Empirical Landscape Reconstruction summary

- **n_cells**: 38,193
- **attractors**: 4
- **saddles (from Hessian inspection)**: 2
- **basin counts**: {'A1': 22487, 'A0': 11368, 'A3': 2859, 'A2': 1479}
- **rare-category counts**: {'core': 35941, 'basin_edge': 1910, 'high_U': 342}
- **rare states discovered**: 5
- **barrier heights**: median 21.01, range [5.34, 24.27]

## Critical points

| id   | kind      |      phi |   score_norm |   min_eig |   max_eig |   n_negative_eigs |
|:-----|:----------|---------:|-------------:|----------:|----------:|------------------:|
| A0   | saddle    | -27.1415 |     16.2021  | -0.340965 |  202.807  |                 1 |
| A1   | attractor | -27.7759 |      4.96887 |  0.178993 |  305.763  |                 0 |
| A2   | attractor | -29.4845 |      5.01871 |  0.224511 |  253.65   |                 0 |
| A3   | attractor | -12.398  |     10.406   |  0.364369 |   13.37   |                 0 |
| A4   | saddle    | -14.3464 |      9.71244 | -9.84309  |   18.9354 |                 1 |
| A5   | attractor |  -9.9193 |     12.1816  |  0.242864 |   17.7546 |                 0 |

## Barriers (string method)

| from   | to   |   barrier_i_to_j |   barrier_j_to_i |   phi_max |   saddle_node |
|:-------|:-----|-----------------:|-----------------:|----------:|--------------:|
| A0     | A1   |          24.2677 |         25.9763  |  -3.50815 |             1 |
| A0     | A2   |          21.6971 |          6.31915 |  -6.07881 |             9 |
| A0     | A3   |          17.8566 |          0       |  -9.9193  |            19 |
| A1     | A2   |          20.4775 |          3.391   |  -9.00696 |             5 |
| A1     | A3   |          21.5337 |          1.96854 |  -7.95077 |            12 |
| A2     | A3   |           5.3389 |          2.86024 |  -7.05906 |            12 |

## Rare states catalog

| state_id        |   n_cells | basin   | category   |   mean_phi |   mean_basin_margin | top_vangalen_type   |   top_vangalen_frac | top_atlas_type   |   top_atlas_frac |
|:----------------|----------:|:--------|:-----------|-----------:|--------------------:|:--------------------|--------------------:|:-----------------|-----------------:|
| A1_basin_edge_0 |      1293 | A1      | basin_edge |  -13.8816  |          0.0425382  | Prog-like           |            0.227378 | HSPC             |         0.540603 |
| A1_basin_edge_3 |        12 | A1      | basin_edge |  -10.8861  |          0.0529259  | B                   |            1        | B                |         0.583333 |
| A1_basin_edge_4 |         9 | A1      | basin_edge |   -5.97184 |          0.052901   | HSC-like            |            0.444444 | nan              |         0.888889 |
| A1_basin_edge_2 |         6 | A1      | basin_edge |   -1.13157 |          0.0141862  | lateEry             |            1        | nan              |         0.666667 |
| A1_basin_edge_1 |         6 | A1      | basin_edge |  -11.1423  |          0.00701884 | lateEry             |            1        | nan              |         1        |

## Basin × Van Galen cell type crosstab

| basin   |   B |   CTL |   GMP |   GMP-like |   HSC |   HSC-like |   Mono |   Mono-like |   NK |   Plasma |   ProB |   ProMono |   ProMono-like |   Prog |   Prog-like |    T |   cDC |   cDC-like |   earlyEry |   lateEry |   nan |   pDC |
|:--------|----:|------:|------:|-----------:|------:|-----------:|-------:|------------:|-----:|---------:|-------:|----------:|---------------:|-------:|------------:|-----:|------:|-----------:|-----------:|----------:|------:|------:|
| A0      |  64 |    12 |   147 |        146 |     6 |         14 |   2685 |        2514 |   15 |     1094 |      1 |      1063 |           1314 |     13 |          22 |   44 |   792 |       1334 |          8 |        52 |     1 |    27 |
| A1      | 388 |  1166 |   628 |       1361 |  1392 |       1578 |     38 |          22 | 1748 |       36 |    245 |        54 |             72 |   1389 |        2975 | 6158 |    41 |        607 |        865 |       899 |   667 |   158 |
| A2      |  16 |     4 |    81 |        189 |   113 |        182 |      3 |           0 |    6 |        1 |     24 |        12 |             10 |    151 |         411 |   34 |     3 |         63 |        106 |        23 |    30 |    17 |
| A3      |  52 |   152 |    58 |        121 |   187 |        156 |      7 |           1 |  194 |        3 |     26 |         9 |              7 |    135 |         283 |  848 |     5 |         56 |        130 |       327 |    82 |    20 |
