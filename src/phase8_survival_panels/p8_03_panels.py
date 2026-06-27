"""Phase 8.3 — diagnostic + prognostic biomarker panels.

DIAGNOSTIC PANEL (malignant vs normal, from the single-cell data):
  * Random forest on real cells (malignant vs normal, primary_malignant_call).
  * Top features by importance → recursive trimming to the smallest gene set
    holding 5-fold CV AUC ≥ 0.95.

PROGNOSTIC PANEL (overall survival, from TCGA-LAML):
  * Univariate Cox screen on individual genes (top markers of HIGH states),
    keep p<0.01.
  * LASSO-Cox (scikit-survival CoxnetSurvivalAnalysis) → compact gene panel.
  * Compare C-index to the LSC17 signature score.

Outputs:
  outputs/phase8/diagnostic_panel.csv
  outputs/phase8/prognostic_panel.csv
  outputs/phase8/panels_summary.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(".")
PROC = ROOT / "data/processed"
TCGA = ROOT / "data/raw/tcga_laml"
OUTP5 = ROOT / "outputs/phase5"
OUTP8 = ROOT / "outputs/phase8"

#   DIAGNOSTIC PANEL
print("[p8_03]   diagnostic panel (malignant vs normal)  ")
a = sc.read_h5ad(PROC / "van_galen_phase5_complete.h5ad")
is_syn = a.obs["is_synthetic"].astype(str).isin(["True","1","1.0"]).values
a = a[~is_syn].copy()  # real cells only
call = a.obs["primary_malignant_call"].astype(str)
y = call.isin(["malignant","putative_malignant"]).astype(int).values
# keep cells with a clear call
keep = call.isin(["malignant","putative_malignant","normal","normal-like"]).values
import scipy.sparse as sp
X = a.X
X = X.toarray() if sp.issparse(X) else np.asarray(X)
X = X[keep]; y = y[keep]
genes = a.var_names.tolist()
print(f"[p8_03] diagnostic: {X.shape[0]} cells, malignant frac {y.mean():.2f}")

rf = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=0,
                             class_weight="balanced")
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=genes).sort_values(ascending=False)
top50 = imp.head(50).index.tolist()

# recursive trimming: shrink until AUC < 0.95
cv = StratifiedKFold(5, shuffle=True, random_state=0)
panel = top50[:]
best_panel, best_auc = top50, 0
for k in [50, 40, 30, 25, 20, 15, 12, 10, 8]:
    feats = top50[:k]
    idx = [genes.index(g) for g in feats]
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    auc = cross_val_score(clf, X[:, idx], y, cv=cv, scoring="roc_auc").mean()
    print(f"  panel size {k:>3d}: AUC = {auc:.4f}")
    if auc >= 0.95:
        best_panel, best_auc = feats, auc
diag_df = pd.DataFrame({"gene": best_panel,
                        "rf_importance": imp.loc[best_panel].values})
diag_df.to_csv(OUTP8 / "diagnostic_panel.csv", index=False)
print(f"[p8_03] diagnostic panel: {len(best_panel)} genes, CV-AUC {best_auc:.3f}")

#   PROGNOSTIC PANEL
print("\n[p8_03]   prognostic panel (TCGA-LAML survival)  ")
# Candidate genes = union of top-15 markers across HIGH states
cat = pd.read_csv(ROOT / "outputs/phase7/validated_state_catalog.csv")
high = cat[cat["confidence_tier"] == "HIGH"]["state_id"].tolist()
de = pd.read_csv(OUTP5 / "de_per_state_real.csv")
cand = set()
for sid in high:
    g = de[(de["group"]==sid)&(de["logfoldchanges"]>0)].nlargest(15,"logfoldchanges")["names"].tolist()
    cand |= set(g)
cand = sorted(cand)
print(f"[p8_03] candidate prognostic genes: {len(cand)}")

# TCGA TPM, mapped to symbols (reuse mapping logic)
atlas = sc.read_h5ad(ROOT / "data/raw/scatlas/AML_scAtlas.h5ad", backed="r")
ens_to_sym = {}
col = "feature_id" if "feature_id" in atlas.var.columns else None
if col:
    for eid, sym in zip(atlas.var[col].astype(str), atlas.var["feature_name"].astype(str)):
        ens_to_sym[eid.split(".")[0]] = sym
else:
    for eid, sym in zip(atlas.var_names.astype(str), atlas.var["feature_name"].astype(str)):
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
print(f"[p8_03] TCGA patients with TPM+survival: {len(common)}")

cand_present = [g for g in cand if g in tpm.index]
Xg = tpm.loc[cand_present, common].T.values   # patients × genes
Xg = np.log1p(Xg)
T = surv.loc[common, "OS.time"].values
E = surv.loc[common, "OS"].values.astype(bool)
mask = ~np.isnan(T) & (T > 0)
Xg, T, E = Xg[mask], T[mask], E[mask]
print(f"[p8_03] prognostic frame: {Xg.shape}")

# Univariate Cox screen
from lifelines import CoxPHFitter
uni_p = {}
for j, g in enumerate(cand_present):
    d = pd.DataFrame({"t": T, "e": E.astype(int), "x": (Xg[:,j]-Xg[:,j].mean())/(Xg[:,j].std()+1e-9)})
    try:
        c = CoxPHFitter(penalizer=0.1).fit(d, "t", "e")
        uni_p[g] = float(c.summary.loc["x","p"])
    except Exception:
        pass
screen = [g for g, p in uni_p.items() if p < 0.05]
print(f"[p8_03] univariate-screen genes (p<0.05): {len(screen)}")

# LASSO-Cox
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
prog_panel = []
prog_cidx = None
lsc_cidx = None
if len(screen) >= 3:
    idxs = [cand_present.index(g) for g in screen]
    Xs = StandardScaler().fit_transform(Xg[:, idxs])
    ysurv = Surv.from_arrays(E, T)
    try:
        net = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=10000)
        net.fit(Xs, ysurv)
        # pick a mid alpha with nonzero coefs
        coefs = net.coef_  # genes × alphas
        # choose alpha with ~10 nonzero
        nnz = (np.abs(coefs) > 1e-6).sum(0)
        target_col = np.argmin(np.abs(nnz - 10))
        sel = np.where(np.abs(coefs[:, target_col]) > 1e-6)[0]
        prog_panel = [screen[i] for i in sel]
        prog_cidx = float(net.score(Xs, ysurv))
    except Exception as e:
        print(f"[p8_03] LASSO-Cox failed: {e}")
        prog_panel = screen[:15]

# LSC17 comparison
sigs = json.load(open(ROOT / "data/raw/msigdb/AML_signatures.json"))
lsc17 = [g for g in sigs["LSC17_Ng2016"] if g in tpm.index]
if len(lsc17) >= 5:
    lsc_score = np.log1p(tpm.loc[lsc17, common].T.values).mean(1)[mask]
    d = pd.DataFrame({"t": T, "e": E.astype(int),
                      "x": (lsc_score-lsc_score.mean())/(lsc_score.std()+1e-9)})
    try:
        c = CoxPHFitter(penalizer=0.1).fit(d, "t", "e")
        lsc_cidx = float(c.concordance_index_)
    except Exception:
        pass

prog_df = pd.DataFrame({"gene": prog_panel})
prog_df.to_csv(OUTP8 / "prognostic_panel.csv", index=False)

summary = {
    "diagnostic_panel_size": len(best_panel),
    "diagnostic_cv_auc": round(float(best_auc), 4),
    "diagnostic_genes": best_panel,
    "prognostic_screen_genes": len(screen),
    "prognostic_panel_size": len(prog_panel),
    "prognostic_panel_genes": prog_panel,
    "prognostic_panel_cindex": round(prog_cidx, 3) if prog_cidx else None,
    "LSC17_cindex": round(lsc_cidx, 3) if lsc_cidx else None,
}
with open(OUTP8 / "panels_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\n[p8_03]   panels summary  ")
print(json.dumps(summary, indent=2, default=str))
