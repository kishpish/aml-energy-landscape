"""Phase 5.1 — define cell states for the dynamical-fingerprint analysis.

Definition (top-down): a "cell state" is a population of cells that share
both a (a) biological identity and (b) a position in landscape topology.
We construct it from three nested levels:

  L0  basin              — coarse: 4 attractors {A0, A1, A2, A3}
  L1  sub-cluster        — within each basin, Leiden at resolution 1.0
                            on the augmented X_scVI
  L2  rare-state overlay — the Phase 3 rare-state ids (basin_edge subsets)

A `state_id` is composed as:
  - "{basin}_L1_{subcluster}"                    for ordinary states
  - "{rare_state_id}"                            for the Phase 3 rare states
The rare-state overlay takes precedence over L1 subclusters where they
overlap.

The dynamical-fingerprint analysis (5.2) then operates per state_id, using
real cells only for biological marker DE and synthetic cells for
statistical power on the augmented states.

Outputs:
  data/augmented/van_galen_phase5_states.h5ad   (adds obs['state_id'])
  outputs/phase5/state_definitions.csv          one row per state
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
OUTP5.mkdir(exist_ok=True, parents=True)

print("[p5_01] loading augmented AnnData…")
a = sc.read_h5ad(AUG / "van_galen_phase4_augmented.h5ad")
print(f"[p5_01] {a.shape}  is_synthetic counts: {a.obs['is_synthetic'].value_counts().to_dict()}")

# --- Sub-clustering with Leiden inside each basin ---
print("[p5_01] running Leiden at resolution 1.0 (already has neighbors)…")
sc.tl.leiden(a, resolution=1.0, key_added="leiden_r1")
print(f"[p5_01] Leiden r=1.0 → {a.obs['leiden_r1'].nunique()} clusters")

# Compose state_id: basin × Leiden, then overlay rare states
state_id = np.array([f"{b}_L1_{l}" for b, l in zip(
    a.obs["basin"].astype(str), a.obs["leiden_r1"].astype(str))], dtype=object)

# Overlay rare-state ids where they exist (priority)
src = a.obs["source_state"].astype(str).values
real_rare = a.obs["rare_state_id"].astype(str).values if "rare_state_id" in a.obs.columns else np.array([""]*a.n_obs)
is_syn_b = a.obs["is_synthetic"].astype(str).isin(["True", "1", "1.0"]).values
for i in range(a.n_obs):
    if is_syn_b[i] and src[i] not in ("", "nan"):
        state_id[i] = src[i]
    elif (not is_syn_b[i]) and real_rare[i] not in ("", "nan"):
        state_id[i] = real_rare[i]

a.obs["state_id"] = pd.Categorical(state_id)
state_counts = a.obs["state_id"].value_counts()
print(f"[p5_01] {len(state_counts)} states defined")
print(state_counts.head(25))

# --- Build state definitions table ---
records = []
for sid, n in state_counts.items():
    mask = a.obs["state_id"].values == sid
    is_syn = a.obs.loc[mask, "is_synthetic"]
    # Normalize: accept bool, uint8(0/1), or str("True"/"False"/"0"/"1")
    is_syn_b = is_syn.astype(str).isin(["True", "1", "1.0"])
    n_real = int((~is_syn_b).sum())
    n_syn  = int(is_syn_b.sum())

    # basin assignment
    basins = a.obs.loc[mask, "basin"].astype(str)
    basin_top = basins.mode().iloc[0] if len(basins) else ""

    # cell-type composition
    if "VanGalen_CellType" in a.obs.columns:
        vg = a.obs.loc[mask, "VanGalen_CellType"].astype(str)
        vg_top = vg.mode().iloc[0] if len(vg) else "nan"
        vg_top_frac = (vg == vg_top).mean()
    else:
        vg_top, vg_top_frac = "nan", 0.0
    if "Author Cell Type" in a.obs.columns:
        at = a.obs.loc[mask, "Author Cell Type"].astype(str)
        at_top = at.mode().iloc[0] if len(at) else "nan"
        at_top_frac = (at == at_top).mean()
    else:
        at_top, at_top_frac = "nan", 0.0

    # malignant fraction
    if "primary_malignant_call" in a.obs.columns:
        mal = a.obs.loc[mask, "primary_malignant_call"].astype(str)
        mal_frac = (mal.isin(["malignant", "putative_malignant"])).mean()
    else:
        mal_frac = np.nan

    # patient diversity
    if "patient_id" in a.obs.columns:
        n_patients = a.obs.loc[mask, "patient_id"].astype(str).nunique()
    else:
        n_patients = -1

    # is this a Phase-3 rare state?
    is_rare = sid.startswith("A") and "basin_edge" in sid

    records.append({
        "state_id": sid,
        "basin": basin_top,
        "n_cells_total": int(n),
        "n_real": n_real,
        "n_synth": n_syn,
        "augmentation_factor": (n / n_real if n_real > 0 else np.nan),
        "top_vangalen_type": vg_top,
        "top_vangalen_frac": round(float(vg_top_frac), 3),
        "top_atlas_type": at_top,
        "top_atlas_frac": round(float(at_top_frac), 3),
        "malignant_frac": round(float(mal_frac), 3) if not np.isnan(mal_frac) else None,
        "n_patients": int(n_patients),
        "is_rare_state": is_rare,
    })

df = pd.DataFrame(records).sort_values("n_cells_total", ascending=False).reset_index(drop=True)
df.to_csv(OUTP5 / "state_definitions.csv", index=False)
print(f"\n[p5_01] wrote {OUTP5 / 'state_definitions.csv'} ({len(df)} states)")
print(df.head(25).to_string(index=False))

# --- Save augmented AnnData with state_id ---
a.write_h5ad(AUG / "van_galen_phase5_states.h5ad", compression="gzip")
print(f"[p5_01] wrote augmented + state_id AnnData ({(AUG / 'van_galen_phase5_states.h5ad').stat().st_size/1e6:.1f} MB)")
