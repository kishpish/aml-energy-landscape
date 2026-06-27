# Phase 7 — Multi-Level Validation summary

- **states**: 79
- **confidence tiers**: {'HIGH': 34, 'EXPLORATORY': 33, 'MODERATE': 12}

## Cross-dataset reproduction (AML scAtlas, 748k cells)

- exact cell-type match: 20/40
- broad-lineage match: 34/40
- significant enrichment (p<0.05): 40/40

## Functional concordance (LINCS predictions vs Beat AML ex vivo)

- drugs in both resources: 21
- mean concordance rate: 0.079
- background rate: 0.086
- enrichment over background: 0.92× (near 1.0 = connectivity does NOT strongly predict ex vivo cytotoxicity in the 21 overlapping drugs — honest null)

## Robustness to synthetic cells

- real-vs-augmented marker overlap: 9.57/10
- basin agreement Phase3↔Phase5: 100%
- all LSC subtypes real-only: True

## HIGH-confidence states

| state_id   | basin   | top_vangalen_type   | top_scoring_atlas_ct   | best_pathway_term                                                                     |   n_real |
|:-----------|:--------|:--------------------|:-----------------------|:--------------------------------------------------------------------------------------|---------:|
| A0_L1_2    | A0      | Mono-like           | CD16+ Mono             | Neutrophil Degranulation R-HSA-6798695                                                |     5994 |
| A1_L1_1    | A1      | T                   | T                      | Coronavirus disease                                                                   |     5340 |
| A1_L1_0    | A1      | Prog-like           | HSPC                   | Myc Targets V1                                                                        |     3492 |
| A1_L1_5    | A1      | NK                  | NK                     | Immunoregulatory Interactions Between A Lymphoid And A non-Lymphoid Cell R-HSA-198933 |     2814 |
| A0_L1_6    | A0      | ProMono             | ProMono                | Neutrophil Degranulation R-HSA-6798695                                                |     2679 |
| A1_L1_3    | A1      | Prog-like           | HSPC                   | PD-1 Signaling R-HSA-389948                                                           |     2126 |
| A0_L1_11   | A0      | Plasma              | Plasma                 | Protein processing in endoplasmic reticulum                                           |     1148 |
| A1_L1_10   | A1      | T                   | T                      | TNF-alpha Signaling via NF-kB                                                         |     1104 |
| A0_L1_8    | A0      | cDC-like            | cDC                    | Immune System R-HSA-168256                                                            |     1055 |
| A1_L1_7    | A1      | earlyEry            | Erythroid              | heme Metabolism                                                                       |      810 |
| A1_L1_9    | A1      | GMP-like            | CMP                    | ATF6 (ATF6-alpha) Activates Chaperones R-HSA-381033                                   |      782 |
| A3_L1_1    | A3      | T                   | T                      | Coronavirus disease                                                                   |      759 |
| A1_L1_12   | A1      | earlyEry            | MEP                    | Hemostasis R-HSA-109582                                                               |      546 |
| A3_L1_0    | A3      | Prog-like           | HSPC                   | Cellular Responses To Stimuli R-HSA-8953897                                           |      450 |
| A2_L1_0    | A2      | Prog-like           | HSPC                   | Myc Targets V1                                                                        |      415 |
| A3_L1_5    | A3      | NK                  | NK                     | Immunoregulatory Interactions Between A Lymphoid And A non-Lymphoid Cell R-HSA-198933 |      352 |
| A1_L1_8    | A1      | cDC-like            | pDC                    | Immune System R-HSA-168256                                                            |      343 |
| A3_L1_7    | A3      | lateEry             | Erythroid              | heme Metabolism                                                                       |      330 |
| A1_L1_14   | A1      | B                   | B                      | Interferon Gamma Response                                                             |      313 |
| A1_L1_13   | A1      | lateEry             | Erythroid              | heme Metabolism                                                                       |      311 |
| A1_L1_16   | A1      | T                   | ProB                   | G2-M Checkpoint                                                                       |      301 |
| A3_L1_3    | A3      | HSC-like            | HSPC                   | Graft-versus-host disease                                                             |      197 |
| A1_L1_6    | A1      | GMP-like            | ProMono                | E2F Targets                                                                           |      196 |
| A3_L1_10   | A3      | T                   | T                      | FCERI Mediated Ca+2 Mobilization R-HSA-2871809                                        |      152 |
| A0_L1_4    | A0      | cDC-like            | ProMono                | E2F Targets                                                                           |      133 |
| A0_L1_9    | A0      | ProMono-like        | ProMono                | Neutrophil Degranulation R-HSA-6798695                                                |      131 |
| A2_L1_12   | A2      | earlyEry            | MEP                    | Signaling By Interleukins R-HSA-449147                                                |      113 |
| A2_L1_9    | A2      | GMP-like            | CMP                    | ATF6 (ATF6-alpha) Activates Chaperones R-HSA-381033                                   |      109 |
| A3_L1_13   | A3      | lateEry             | Erythroid              | heme Metabolism                                                                       |       96 |
| A3_L1_9    | A3      | GMP-like            | CMP                    | Neutrophil Degranulation R-HSA-6798695                                                |       86 |
| A1_L1_2    | A1      | ProMono-like        | CD16+ Mono             | Neutrophil Degranulation R-HSA-6798695                                                |       79 |
| A0_L1_13   | A0      | lateEry             | Erythroid              | heme Metabolism                                                                       |       75 |
| A2_L1_7    | A2      | earlyEry            | Erythroid              | heme Metabolism                                                                       |       66 |
| A3_L1_8    | A3      | cDC-like            | pDC                    | Immune System R-HSA-168256                                                            |       54 |
