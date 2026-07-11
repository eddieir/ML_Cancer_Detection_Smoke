#!/usr/bin/env bash
# setup_project.sh
# Run from inside ML_Cancer_Detection_Smoke/
# Builds the full project structure and moves existing flat files to correct locations.

set -e
ROOT="$(pwd)"

echo "=== Building project structure in: $ROOT ==="

# ─── Directories ──────────────────────────────────────────────────────────────
mkdir -p src/data
mkdir -p configs
mkdir -p data/raw data/processed
mkdir -p checkpoints
mkdir -p notebooks
mkdir -p tests

# ─── Python packages ──────────────────────────────────────────────────────────
touch src/__init__.py
touch src/data/__init__.py
touch tests/__init__.py

# ─── Move flat files to correct locations ─────────────────────────────────────
for f in constants.py loaders.py transforms.py labellers.py assembly.py; do
    [ -f "$ROOT/$f" ] && mv "$ROOT/$f" "$ROOT/src/data/$f" && echo "  moved $f → src/data/$f"
done

[ -f "$ROOT/preprocess.py" ] && mv "$ROOT/preprocess.py" "$ROOT/src/preprocess.py" \
    && echo "  moved preprocess.py → src/preprocess.py"

# constants.py belongs at src/ level (used by model.py too — not just data/)
[ -f "$ROOT/src/data/constants.py" ] && mv "$ROOT/src/data/constants.py" "$ROOT/src/constants.py" \
    && echo "  moved constants.py → src/constants.py (shared by all src modules)"

# ─── Placeholder src files (future steps) ─────────────────────────────────────
for f in model.py train.py evaluate.py inference.py; do
    [ ! -f "src/$f" ] && printf '# TODO: Step %s\n' "$f" > "src/$f" \
        && echo "  created placeholder src/$f"
done

# ─── Notebook stubs ───────────────────────────────────────────────────────────
for nb in "01_data_download" "02_preprocessing" "03_training" "04_evaluation"; do
    [ ! -f "notebooks/${nb}.ipynb" ] && echo '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}' \
        > "notebooks/${nb}.ipynb" && echo "  created notebooks/${nb}.ipynb"
done

# ─── Test stubs ───────────────────────────────────────────────────────────────
for t in test_loaders test_transforms test_labellers test_assembly test_model test_pipeline; do
    [ ! -f "tests/${t}.py" ] && printf '# %s\nimport pytest\n' "$t" > "tests/${t}.py" \
        && echo "  created tests/${t}.py"
done

# ─── Config ───────────────────────────────────────────────────────────────────
cat > configs/default.yaml << 'YAML'
data:
  scrna_sources: []
  microarray_sources: []
  loiselle_path: null
  gse288003_path: null
  nlst_csv: null
  nlst_outcomes_csv: null
  tumor_barcodes: null
  n_hvgs: 2000
  out_dir: data/processed

model:
  input_dim: 2000
  embedding_dim: 256
  num_smoke_types: 6
  num_cell_types: 4
  dropout: 0.3
  attention_dim: 128

train:
  phase1_epochs: 15
  phase1_lr: 1.0e-3
  phase1_batch_size: 512
  phase2_epochs: 12
  phase2_lr: 5.0e-4
  phase3_epochs: 8
  phase3_lr: 1.0e-4
  patience: 5
  grad_clip: 1.0
  checkpoint_dir: checkpoints
YAML
echo "  created configs/default.yaml"

# ─── requirements.txt ─────────────────────────────────────────────────────────
cat > requirements.txt << 'REQ'
torch>=2.2.0
scanpy>=1.9.8
anndata>=0.10.0
harmonypy>=0.0.9
celltypist>=1.6.0
pybiomart>=0.2.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.13.0
pyyaml>=6.0
REQ
echo "  created requirements.txt"

# ─── .gitignore ───────────────────────────────────────────────────────────────
cat > .gitignore << 'GIT'
# data — never commit raw or processed genomics data
data/raw/
data/processed/

# model checkpoints
checkpoints/

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/
.eggs/

# environments
.venv/
env/
.env

# notebooks
.ipynb_checkpoints/

# OS
.DS_Store
GIT
echo "  created .gitignore"

# ─── Final tree ───────────────────────────────────────────────────────────────
echo ""
echo "=== Done. Final structure: ==="
find . -not -path './.git/*' -not -path './__pycache__/*' \
       -not -path './src/__pycache__/*' \
       -not -path './src/data/__pycache__/*' \
       -not -name '*.pyc' \
    | sort | sed 's|[^/]*/|  |g; s|  \([^ ]\)|── \1|'