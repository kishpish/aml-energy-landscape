"""Download Human Cell Atlas bone marrow (healthy) cells via CellxGene Census."""
import cellxgene_census
import scanpy as sc
import sys

OUT = "./data/raw/hca_bm/hca_bone_marrow.h5ad"

# Pin to a specific Census release for reproducibility — latest stable
CENSUS_VERSION = "stable"

print(f"[hca_bm] opening Census release='{CENSUS_VERSION}'")
with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
    print("[hca_bm] querying bone marrow / normal / Homo sapiens ...")
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter=(
            'tissue_general == "bone marrow" '
            'and disease == "normal" '
            'and is_primary_data == True'
        ),
        column_names={
            "obs": [
                "soma_joinid", "dataset_id", "assay", "cell_type",
                "donor_id", "sex", "development_stage",
                "tissue", "suspension_type", "is_primary_data",
            ],
        },
    )

print(f"[hca_bm] downloaded AnnData: {adata}")
print(f"[hca_bm] writing {OUT}")
adata.write_h5ad(OUT, compression="gzip")
print("[hca_bm] done.")
