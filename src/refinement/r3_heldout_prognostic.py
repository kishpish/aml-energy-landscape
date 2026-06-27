"""R3 — honest out-of-sample validation of the prognostic gene panel.

The Phase 8 C-index (0.775) was computed in-sample (fit and scored on the same
patients) → optimistic. Here we do proper nested evaluation:

  * 5-fold cross-validation: in each fold, run the FULL pipeline (univariate
    Cox screen → LASSO-Cox feature selection → fit) on the TRAIN split only,
    then score the held-out TEST split. Report mean test C-index.
  * Same nested protocol for LSC17 (fit a single-covariate Cox on train, score
    test) so the comparison is apples-to-apples and equally out-of-sample.
  * Repeat 5x with different seeds; report mean ± SD.

This removes the in-sample leak and the feature-selection leak (the most common
cause of inflated genomic-signature C-indices in the literature).

Output: outputs/phase8/prognostic_heldout.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
TCGA = ROOT / "data/raw/tcga_laml"
OUTP5 = ROOT / "outputs/phase5"
OUTP7 = ROOT / "outputs/phase7"
OUTP8 = ROOT / "outputs/phase8"

# --- Ensembl→symbol + TPM (reuse Phase 8 logic) ---
atlas = sc.read_h5ad(ROOT / "data/raw/scatlas/AML_scAtlas.h5ad", backed="r")
ens_to_sym = {}
col = "feature_id" if "feature_id" in atlas.var.columns else None
src = atlas.var[col].astype(str) if col else atlas.var_names.astype(str)
for eid, sym in zip(src, atlas.var["feature_name"].astype(str)):
    ens_to_sym[eid.split(".")[0]] = sym
del atlas

tpm = pd.read_csv(TCGA / "TCGA-LAML.star_tpm.tsv.gz", sep="\t", index_col=0)
tpm.index = [ens_to_sym.get(str(e).split(".")[0], None) for e in tpm.index]
tpm = tpm[~tpm.index.isna()]; tpm = tpm[~tpm.index.duplicated(keep="first")]
tpm.columns = [c[:12] for c in tpm.columns]
tpm = tpm.loc[:, ~tpm.columns.duplicated()]

surv = pd.read_csv(TCGA / "TCGA-LAML.survival.tsv.gz", sep="\t")
surv["patient"] = surv["_PATIENT"]; surv = surv.drop_duplicates("patient").set_index("patient")
common = [p for p in tpm.columns if p in surv.index]
T = surv.loc[common, "OS.time"].values.astype(float)
E = surv.loc[common, "OS"].values.astype(int)
ok = (~np.isnan(T)) & (T > 0)
common = list(np.array(common)[ok]); T = T[ok]; E = E[ok]
print(f"[r3] patients: {len(common)}")

# candidate genes = union of top-15 markers across HIGH states
cat = pd.read_csv(OUTP7 / "validated_state_catalog.csv")
high = cat[cat["confidence_tier"] == "HIGH"]["state_id"].tolist()
de = pd.read_csv(OUTP5 / "de_per_state_real.csv")
cand = set()
for sid in high:
    g = de[(de["group"]==sid)&(de["logfoldchanges"]>0)].nlargest(15,"logfoldchanges")["names"].tolist()
    cand |= set(g)
cand = [g for g in sorted(cand) if g in tpm.index]
X_all = np.log1p(tpm.loc[cand, common].T.values)   # patients × genes
print(f"[r3] candidate genes: {len(cand)}")

# LSC17 score per patient
sigs = json.load(open(ROOT / "data/raw/msigdb/AML_signatures.json"))
lsc17 = [g for g in sigs["LSC17_Ng2016"] if g in tpm.index]
lsc_score_all = np.log1p(tpm.loc[lsc17, common].T.values).mean(1)
print(f"[r3] LSC17 genes present: {len(lsc17)}")

def nested_eval(seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    panel_cidx, lsc_cidx = [], []
    for tr, te in kf.split(X_all):
        Xtr, Xte = X_all[tr], X_all[te]
        Ttr, Tte = T[tr], T[te]; Etr, Ete = E[tr], E[te]
        if Ete.sum() < 3 or Etr.sum() < 5:
            continue
        # --- panel: univariate screen on TRAIN, LASSO-Cox on TRAIN, score TEST ---
        keep_g = []
        for j in range(Xtr.shape[1]):
            x = Xtr[:, j]
            d = pd.DataFrame({"t": Ttr, "e": Etr, "x": (x-x.mean())/(x.std()+1e-9)})
            try:
                c = CoxPHFitter(penalizer=0.1).fit(d, "t", "e")
                if c.summary.loc["x","p"] < 0.10:
                    keep_g.append(j)
            except Exception:
                pass
        if len(keep_g) < 3:
            keep_g = list(range(min(15, Xtr.shape[1])))
        sc_ = StandardScaler().fit(Xtr[:, keep_g])
        Xtr_s = sc_.transform(Xtr[:, keep_g]); Xte_s = sc_.transform(Xte[:, keep_g])
        try:
            net = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=10000)
            net.fit(Xtr_s, Surv.from_arrays(Etr.astype(bool), Ttr))
            # pick alpha with ~10 nonzero on train
            nnz = (np.abs(net.coef_) > 1e-6).sum(0)
            col_ = int(np.argmin(np.abs(nnz - 10)))
            risk_te = Xte_s @ net.coef_[:, col_]
            ci = concordance_index(Tte, -risk_te, Ete)   # higher risk → shorter survival
            panel_cidx.append(ci)
        except Exception:
            pass
        # --- LSC17 (single covariate), out-of-sample ---
        ls_tr = lsc_score_all[tr]; ls_te = lsc_score_all[te]
        d = pd.DataFrame({"t": Ttr, "e": Etr, "x": (ls_tr-ls_tr.mean())/(ls_tr.std()+1e-9)})
        try:
            c = CoxPHFitter(penalizer=0.1).fit(d, "t", "e")
            beta = c.params_["x"]
            risk_te = ((ls_te-ls_tr.mean())/(ls_tr.std()+1e-9)) * beta
            lsc_cidx.append(concordance_index(Tte, -risk_te, Ete))
        except Exception:
            pass
    return np.mean(panel_cidx) if panel_cidx else np.nan, np.mean(lsc_cidx) if lsc_cidx else np.nan

panel_scores, lsc_scores = [], []
for seed in range(5):
    p, l = nested_eval(seed)
    panel_scores.append(p); lsc_scores.append(l)
    print(f"[r3] seed {seed}: panel C-index={p:.3f}  LSC17 C-index={l:.3f}")

res = {
    "n_patients": len(common),
    "n_events": int(E.sum()),
    "panel_heldout_cindex_mean": round(float(np.nanmean(panel_scores)), 3),
    "panel_heldout_cindex_sd": round(float(np.nanstd(panel_scores)), 3),
    "lsc17_heldout_cindex_mean": round(float(np.nanmean(lsc_scores)), 3),
    "lsc17_heldout_cindex_sd": round(float(np.nanstd(lsc_scores)), 3),
    "protocol": "5-fold nested CV (feature selection inside train fold), 5 seeds",
    "note": "Out-of-sample; replaces the in-sample 0.775 from Phase 8.",
}
res["panel_beats_lsc17_heldout"] = res["panel_heldout_cindex_mean"] > res["lsc17_heldout_cindex_mean"]
with open(OUTP8 / "prognostic_heldout.json", "w") as f:
    json.dump(res, f, indent=2)
print("\n[r3]   HELD-OUT RESULT  ")
print(json.dumps(res, indent=2))
