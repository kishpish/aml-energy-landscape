
## Repository layout note

The committed `results/` directory holds a snapshot of the key numeric outputs
used in the manuscript, and `models/` holds the small trained models. When the
pipeline scripts are executed they recreate the full working tree at the
repository root (`data/processed/`, `data/augmented/`, `outputs/`, `logs/`, and
`models/score_based/`); those runtime directories are excluded from version
control via `.gitignore`. Run all scripts from the repository root (the runner
scripts in `src/pipelines/` change into it automatically).
