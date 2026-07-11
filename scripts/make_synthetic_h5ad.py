"""
make_synthetic_h5ad.py — builds a fake but correctly-shaped preprocessed h5ad
so `src/inference.py --h5ad ...` has something real to run against.

Runs the actual run_pipeline() from src/preprocess.py on synthetic raw counts,
then re-attaches subject_id/cell_type_id and writes out an AnnData matching
what Predictor.predict_h5ad() expects (scaled expression, exactly
model.input_dim=2000 features to match the existing checkpoints).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "data"))

from constants import ALL_SMOKE_MARKERS
from data.transforms import qc_filter, normalize, smoke_aware_hvg, annotate_cell_types
from data.labellers import add_malignancy_labels

N_SUBJECTS, CELLS_PER_SUBJECT, N_GENES, N_HVGS = 8, 80, 2500, 2000

np.random.seed(0)
N = N_SUBJECTS * CELLS_PER_SUBJECT

genes = [f"GENE_{i}" for i in range(N_GENES)]
genes[:10] = [f"MT-{i}" for i in range(10)]
for i, mk in enumerate(ALL_SMOKE_MARKERS[:8]):
    genes[100 + i] = mk

raw = np.random.negative_binomial(5, 0.7, (N, N_GENES)).astype("float32")
obs = pd.DataFrame({
    "donor_id":        [f"patient_{i // CELLS_PER_SUBJECT:03d}" for i in range(N)],
    "subject_id":      [f"patient_{i // CELLS_PER_SUBJECT:03d}" for i in range(N)],
    "smoke_type":      np.random.randint(0, 6, N),
    "smoke_type_name": "mixed",
    "data_modality":   "scrna",
    "is_pseudo_bulk":  False,
    "malignancy":      0.0,
    "cell_type_id":    0,
}, index=[f"c{i}" for i in range(N)])

adata = ad.AnnData(X=sp.csr_matrix(raw), obs=obs, var=pd.DataFrame(index=genes))

adata = normalize(qc_filter(adata))
adata = smoke_aware_hvg(adata, n_hvgs=N_HVGS, batch_key=None)
adata = annotate_cell_types(adata)
adata = add_malignancy_labels(adata)

out_path = Path(__file__).parents[1] / "data" / "raw" / "patient_synthetic.h5ad"
out_path.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(out_path)

print(f"wrote {adata.n_obs:,} cells x {adata.n_vars} genes -> {out_path}")
print(f"subjects: {sorted(adata.obs['subject_id'].unique())}")
