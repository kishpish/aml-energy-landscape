"""Phase 5.4 — LSC subtyping (quiescent vs cycling vs primed).

Define high-confidence LSC candidates by convergent evidence:
  Criterion 1: LSC17 score in top 10% (Ng et al. 2016)
  Criterion 2: LSC6 score in top 10% (Elsayed et al. 2020)
  Criterion 3: LSC_surface_score in top 10% (CD34+ KIT+ IL3RA+ CD38-)
  Criterion 4: VanGalen_CellType in {HSC-like, Prog-like} AND
              VanGalen_malignant_call == 'malignant'

A cell is "high-confidence LSC" if it meets ≥3 of the 4 criteria.

Sub-cluster the LSC pool by:
  - cell-cycle phase  (Phase 2.11 score_genes_cell_cycle output)
  - basin assignment
  - Leiden at resolution 0.3 within the LSC subset

Then characterize each LSC subtype with:
  - cell cycle composition (G1/S/G2M fractions)
  - D_trace_mean (transcriptional noise)
  - basin_depth (from Phase 5.2 fingerprint)
  - top markers vs other LSCs (Wilcoxon)

Output:
  outputs/phase5/lsc_subtypes.csv
  outputs/phase5/lsc_subtype_markers.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
AUG = ROOT / "data/augmented"
OUTP5 = ROOT / "outputs/phase5"

print("[p5_04] loading state-level AnnData (augmented + state_id)…")
a = sc.read_h5ad(AUG / "van_galen_phase5_states.h5ad")
is_syn = a.obs["is_synthetic"].astype(str).isin(["True", "1", "1.0"]).values
print(f"[p5_04] {a.shape}, real={(~is_syn).sum()}")

# --- LSC criteria ---
# Cast LSC score columns to float (they may have been str-coerced during save)
for col in ["LSC17_score", "LSC6_score", "LSC_surface_score", "D_local_trace"]:
    if col in a.obs.columns:
        a.obs[col] = pd.to_numeric(a.obs[col], errors="coerce")

lsc17 = a.obs["LSC17_score"].values.astype(np.float32)
lsc6 = a.obs["LSC6_score"].values.astype(np.float32) if "LSC6_score" in a.obs.columns else None
lsc_surf = a.obs["LSC_surface_score"].values.astype(np.float32)

# Compute thresholds on REAL cells only
real_mask = ~is_syn
thr_lsc17 = np.percentile(lsc17[real_mask], 90)
thr_lsc6 = np.percentile(lsc6[real_mask], 90) if lsc6 is not None else None
thr_surf = np.percentile(lsc_surf[real_mask], 90)
print(f"[p5_04] LSC thresholds (P90 over real cells):")
print(f"        LSC17 >= {thr_lsc17:.3f}")
print(f"        LSC6  >= {thr_lsc6:.3f}")
print(f"        LSC_surf >= {thr_surf:.3f}")

crit1 = lsc17 >= thr_lsc17
crit2 = lsc6 >= thr_lsc6 if lsc6 is not None else np.zeros_like(crit1)
crit3 = lsc_surf >= thr_surf
# Criterion 4: malignant + HSC-like/Prog-like
vg_type = a.obs["VanGalen_CellType"].astype(str)
mal_call = a.obs["VanGalen_malignant_call"].astype(str) if "VanGalen_malignant_call" in a.obs.columns else None
crit4 = (vg_type.isin(["HSC-like", "Prog-like"])).values
if mal_call is not None:
    crit4 = crit4 & (mal_call == "malignant").values

n_criteria = crit1.astype(int) + crit2.astype(int) + crit3.astype(int) + crit4.astype(int)
hc_lsc = (n_criteria >= 3)
print(f"[p5_04] high-confidence LSC (>=3 criteria): {hc_lsc.sum()} cells "
      f"(real: {(hc_lsc & real_mask).sum()}, synth: {(hc_lsc & ~real_mask).sum()})")

a.obs["LSC_n_criteria"] = n_criteria.astype("int8")
a.obs["LSC_high_confidence"] = hc_lsc.astype("uint8")

# --- Sub-cluster LSC pool ---
print("[p5_04] sub-clustering high-confidence LSCs…")
lsc_ad = a[hc_lsc].copy()
print(f"[p5_04] LSC subset: {lsc_ad.shape}")
sc.pp.neighbors(lsc_ad, use_rep="X_scVI", n_neighbors=15)
sc.tl.leiden(lsc_ad, resolution=0.3, key_added="lsc_subtype_leiden")
n_sub = lsc_ad.obs["lsc_subtype_leiden"].nunique()
print(f"[p5_04] {n_sub} LSC sub-clusters at resolution 0.3")
sub_counts = lsc_ad.obs["lsc_subtype_leiden"].value_counts()
print(f"        sub-cluster sizes: {sub_counts.to_dict()}")

# Per-subtype characterization
records = []
for sub_id in sorted(sub_counts.index, key=lambda x: int(x)):
    mask = (lsc_ad.obs["lsc_subtype_leiden"] == sub_id)
    n = int(mask.sum())
    n_real = int(((~is_syn)[hc_lsc] & mask).sum())
    n_synth = n - n_real
    phase_counts = lsc_ad.obs.loc[mask, "phase"].value_counts(normalize=True).to_dict()
    g1_frac = round(float(phase_counts.get("G1", 0)), 3)
    s_frac  = round(float(phase_counts.get("S", 0)), 3)
    g2m_frac = round(float(phase_counts.get("G2M", 0)), 3)
    basin = lsc_ad.obs.loc[mask, "basin"].astype(str).mode().iloc[0]
    state_id = lsc_ad.obs.loc[mask, "state_id"].astype(str).mode().iloc[0]
    d_trace = (lsc_ad.obs.loc[mask, "D_local_trace"].mean()
                if "D_local_trace" in lsc_ad.obs.columns else np.nan)
    lsc17_mean = float(lsc_ad.obs.loc[mask, "LSC17_score"].mean())
    lsc6_mean = float(lsc_ad.obs.loc[mask, "LSC6_score"].mean()) if "LSC6_score" in lsc_ad.obs.columns else np.nan
    # patient diversity
    n_patients = int(lsc_ad.obs.loc[mask, "patient_id"].nunique())

    # Tag the subtype by phase pattern: quiescent (high G1, low S+G2M) vs cycling (high S+G2M)
    cycling_frac = s_frac + g2m_frac
    if cycling_frac < 0.20:
        tag = "quiescent"
    elif cycling_frac > 0.50:
        tag = "cycling"
    else:
        tag = "mixed"

    records.append({
        "lsc_subtype_id": f"LSC_sub_{sub_id}",
        "n_cells": n, "n_real": n_real, "n_synth": n_synth,
        "n_patients": n_patients,
        "dominant_basin": basin,
        "dominant_state_id": state_id,
        "G1_frac": g1_frac, "S_frac": s_frac, "G2M_frac": g2m_frac,
        "cycling_frac": round(cycling_frac, 3),
        "phase_tag": tag,
        "LSC17_mean": round(lsc17_mean, 3),
        "LSC6_mean": round(lsc6_mean, 3),
        "D_trace_mean": round(float(d_trace), 3) if not np.isnan(d_trace) else None,
    })

df = pd.DataFrame(records)
df.to_csv(OUTP5 / "lsc_subtypes.csv", index=False)
print(f"\n[p5_04] LSC subtype catalog:")
print(df.to_string(index=False))
print(f"\n[p5_04] wrote {OUTP5 / 'lsc_subtypes.csv'}")

# --- Subtype-specific markers (Wilcoxon vs other LSCs) ---
print("\n[p5_04] LSC subtype markers (Wilcoxon)…")
sc.tl.rank_genes_groups(lsc_ad, "lsc_subtype_leiden", method="wilcoxon",
                         n_genes=15, use_raw=False)
mk = sc.get.rank_genes_groups_df(lsc_ad, group=None)
mk_top5 = (mk.groupby("group")
              .apply(lambda x: ",".join(x.head(5)["names"].tolist()))
              .reset_index()
              .rename(columns={0: "top5_markers"}))
df_full = df.merge(mk_top5, left_on=df["lsc_subtype_id"].str.replace("LSC_sub_", ""),
                   right_on="group", how="left").drop(columns=["group"])
df_full.to_csv(OUTP5 / "lsc_subtype_markers.csv", index=False)
print(df_full[["lsc_subtype_id", "phase_tag", "dominant_basin", "top5_markers"]].to_string(index=False))

# Save AnnData with LSC labels
a.write_h5ad(AUG / "van_galen_phase5_lsc.h5ad", compression="gzip")
print(f"\n[p5_04] wrote {AUG / 'van_galen_phase5_lsc.h5ad'}")
