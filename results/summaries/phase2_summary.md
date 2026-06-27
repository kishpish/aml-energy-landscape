# Phase 2 — final deliverable summary

- **n_cells**: 38,193
- **n_genes**: 19,441
- **n_patients (primary cohort)**: 21
- **n_samples (primary cohort)**: 41
- **disease_state**: {'AML_diagnosis': 15582, 'AML_residual_or_relapse': 14946, 'healthy_donor': 7665}
- **primary_malignant_call**: {'normal': 22871, 'malignant': 13438, 'normal-like': 1744, 'putative_malignant': 140}
- **cell_cycle_phase**: {'G1': 23740, 'G2M': 10089, 'S': 4364}
- **Leiden r=1.0 clusters**: 18
- **patients_with_longitudinal_samples**: ['AML314', 'AML328', 'AML329', 'AML371', 'AML420B', 'AML475', 'AML556', 'AML707B', 'AML722B', 'AML870', 'AML997', 'BM5']

## obsm keys present
- `X_scVI`: shape (38193, 30)
- `X_scVI_d30`: shape (38193, 30)
- `X_umap`: shape (38193, 2)
- `score_aucell`: shape (38193, 50)

## obs columns present (required)
- `sample_id`
- `patient_id`
- `timepoint`
- `treatment_status`
- `disease_state`
- `n_genes_by_counts`
- `total_counts`
- `pct_counts_mt`
- `pct_counts_ribo`
- `S_score`
- `G2M_score`
- `phase`
- `celltypist_label`
- `celltypist_majority`
- `cnv_score`
- `cnv_proxy_call`
- `primary_malignant_call`
- `D_local_trace`
- `D_local_anisotropy`
- `LSC17_score`
- `LSC6_score`
- `LSC_surface_score`

## obs columns present (optional)
- `Author Cell Type`
