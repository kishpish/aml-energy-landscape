# Phase 8 — Survival Validation + Biomarker Panels summary

- **TCGA-LAML patients scored**: 151
- **states survival-tested**: 46

## Survival (TCGA-LAML, OS)

- univariate log-rank q<0.05: 0 (nominal p<0.05: 2)
- multivariate Cox q<0.05: 0
- nominal adverse (Cox p<0.05, HR>1): ['A1_L1_8', 'A3_L1_8']
- max ΔC-index (state adds over age+FLT3+NPM1): 0.0186

## Diagnostic panel (malignant vs normal, single-cell)

- **25 genes, 5-fold CV-AUC = 0.9535** (target ≥ 0.95 ✓)

## Prognostic panel (TCGA-LAML OS, LASSO-Cox)

- **10 genes, C-index = 0.775**
- LSC17 C-index (same cohort) = 0.588
- prognostic panel beats LSC17: **True** (in-sample; needs external validation)
