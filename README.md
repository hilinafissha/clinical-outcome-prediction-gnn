# Clinical Outcome Prediction in ICU Patients Using Relational Graph Neural Networks

A deep learning framework for predicting clinical outcomes in ICU patients by applying relational graph neural networks (HeteroGraphSAGE) to multi-table clinical databases. Built on the [RelBench](https://relbench.stanford.edu/) framework for relational deep learning.

## Overview

Most clinical prediction models treat patient data as a flat feature vector, manually engineering features from each table. This project takes a different approach — the raw relational structure of the database (patients, labs, medications, diagnoses, vitals) is modelled directly as a heterogeneous graph, and a GNN learns across table relationships automatically.

Three prediction tasks are implemented:

| Task | Dataset | Type | AUROC |
|------|---------|------|-------|
| AKI Detection (binary) | eICU | Binary classification | — |
| AKI Severity Staging | eICU | Ordinal (Stage ≥1, ≥2, ≥3) | — |
| Vasopressor Initiation (6h window) | eICU | Binary classification | 0.807 |
| Vasopressor Initiation (8h window) | eICU | Binary classification | 0.823 |
| ICD Code Prediction | MIMIC-IV | Multi-label classification | — |
| Principal Diagnosis Prediction | MIMIC-IV | Multi-class classification | — |

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
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric
pip install relbench[full]
pip install sentence-transformers
pip install pandas numpy scikit-learn duckdb tqdm
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

1. Obtain access to eICU and MIMIC-IV from PhysioNet
2. Update the `base_dir` paths in the notebook to point to your local data
3. Run the notebook end to end: `notebook/clinical_outcome_prediction.ipynb`

Each task section is self-contained — you can run them independently by executing the setup cells first.

## Project Structure

```
clinical-outcome-prediction-gnn/
├── notebook/
│   └── clinical_outcome_prediction.ipynb
├── results/
│   ├── aki/
│   ├── vasopressor/
│   └── icd/
├── data/
│   └── README.md
├── requirements.txt
└── .gitignore
```
## Author

Hilina Fissha Woreta
MSc Student — University of Bologna
