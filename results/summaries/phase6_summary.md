# Phase 6 — Drug Action as Landscape Perturbation summary

- **(state, drug) pairs ranked**: 38,224
- **high-confidence hits** (top-decile in ≥2 signals): 221
- **states with ≥1 HC hit**: 39
- **unique drugs in HC hits**: 86

## Top 15 high-confidence (state, drug) hits

| state_id        | drug                         |   composite_score |   n_signals |   conn_rank |   proj_rank |   escape_rank |   beat_rank |
|:----------------|:-----------------------------|------------------:|------------:|------------:|------------:|--------------:|------------:|
| A1_L1_9         | GSK-2334470                  |          0.992788 |           2 |    0.985577 |    1        |           nan |         nan |
| A1_L1_12        | PT-630                       |          0.991587 |           2 |    0.992788 |    0.990385 |           nan |         nan |
| A1_L1_0         | IKK3-inhibitor-IX            |          0.987981 |           2 |    0.990385 |    0.985577 |           nan |         nan |
| A1_basin_edge_0 | cerulenin                    |          0.987981 |           2 |    0.987981 |    0.987981 |           nan |         nan |
| A0_L1_9         | RLM-2-12                     |          0.987981 |           2 |    0.997596 |    0.978365 |           nan |         nan |
| A0_L1_11        | PRIMA1                       |          0.986779 |           2 |    0.987981 |    0.985577 |           nan |         nan |
| A1_L1_5         | IKK3-inhibitor-IX            |          0.984375 |           2 |    0.983173 |    0.985577 |           nan |         nan |
| A1_L1_3         | CHEMBL-1222381               |          0.983173 |           2 |    1        |    0.966346 |           nan |         nan |
| A1_L1_12        | IKK3-inhibitor-IX            |          0.981971 |           2 |    0.978365 |    0.985577 |           nan |         nan |
| A1_L1_2         | BRD-A36275421                |          0.981971 |           2 |    0.987981 |    0.975962 |           nan |         nan |
| A1_L1_14        | BAX-channel-blocker          |          0.981971 |           2 |    1        |    0.963942 |           nan |         nan |
| A2_L1_0         | BRD-K84924563                |          0.980769 |           2 |    0.961538 |    1        |           nan |         nan |
| A1_L1_9         | dalcetrapib                  |          0.980769 |           2 |    0.983173 |    0.978365 |           nan |         nan |
| A0_L1_8         | NFKB-activation-inhibitor-II |          0.979567 |           2 |    1        |    0.959135 |           nan |         nan |
| A1_L1_12        | dalcetrapib                  |          0.979567 |           2 |    0.980769 |    0.978365 |           nan |         nan |

## Beat AML face-validity (ex vivo concordance)

- A0 (monocyte) states ↔ Trametinib/Selumetinib (MEK-i) sensitivity
- A0_L1_2 (monocyte) ↔ Panobinostat (HDAC-i) — recapitulates Beat AML 2.0
- A1_basin_edge_0 (HSPC rare) ↔ GSK-2879552 (LSD1-i) sensitivity

## Connectivity face-validity

- cytarabine in top reversal hits for A1_L1_0 (quiescent LSC state)
