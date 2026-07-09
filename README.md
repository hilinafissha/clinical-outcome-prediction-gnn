# Clinical Outcome Prediction in ICU Patients Using Relational Graph Neural Networks
A deep learning framework for predicting clinical outcomes in ICU patients by applying relational graph neural networks (HeteroGraphSAGE) to multi-table clinical databases. Built on the [RelBench](https://relbench.stanford.edu/) framework for relational deep learning.

## Overview
Most clinical prediction models treat patient data as a flat feature vector, manually engineering features from each table. This project takes a different approach in which the raw relational structure of the database (patients, labs, medications, diagnoses, vitals) is modelled directly as a heterogeneous graph, and a GNN learns across table relationships automatically.

Five prediction tasks are implemented:

| Task | Dataset | Type | Key Metric |
|------|---------|------|------------|
| AKI Detection (binary) | eICU | Binary classification | AUROC 0.7399 |
| AKI Severity Staging | eICU | Ordinal (Stage ≥1 / ≥2 / ≥3) | AUROC 0.7133 / 0.7339 / 0.7718 |
| Vasopressor Initiation (6h window) | eICU | Binary classification | AUROC 0.8323 |
| ICD Code Prediction (top-50) | MIMIC-IV | Multi-label classification | Macro-AUROC 0.8407 |
| Principal Diagnosis Prediction (top-50) | MIMIC-IV | Multi-class classification | Top-1 accuracy 0.2592 |

## Datasets
This project uses two credentialed clinical datasets:
- **eICU Collaborative Research Database** — 200,000+ ICU admissions across 208 US hospitals
- **MIMIC-IV** — ICU admissions from Beth Israel Deaconess Medical Center

Both datasets require credentialed access through [PhysioNet](https://physionet.org/). You must complete the required training and data use agreement before downloading.
- eICU: https://physionet.org/content/eicu-crd/2.0/
- MIMIC-IV: https://physionet.org/content/mimiciv/3.1/

## Model Architecture
- **Encoder**: HeteroEncoder (per-table feature encoding)
- **Temporal Encoder**: HeteroTemporalEncoder (time-aware node embeddings)
- **GNN**: HeteroGraphSAGE (message passing across relational edges)
- **Head**: MLP for task-specific prediction
- **Text embeddings**: GloVe 6B 300d via SentenceTransformer

## Setup

```bash
pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install torch-geometric
pip install relbench[full]
pip install sentence-transformers
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

> **Note:** A CUDA-capable GPU is required. The notebook was developed and tested on a Kaggle T4 instance.

## Usage
1. Obtain access to eICU and MIMIC-IV from PhysioNet
2. Update the `base_dir` paths in the notebook to point to your local data
3. Run the notebook end to end: `notebook/clinical_outcome_prediction.ipynb`

Each task section is self-contained — you can run them independently by executing the setup cells first.

## Project Structure
