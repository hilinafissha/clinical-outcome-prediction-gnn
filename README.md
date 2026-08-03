# Clinical Outcome Prediction in ICU Patients Using Relational Graph Neural Networks

A deep learning framework for predicting clinical outcomes in ICU patients by applying relational graph neural networks (HeteroGraphSAGE) to multi-table clinical databases. Built on the [RelBench](https://relbench.stanford.edu/) framework for relational deep learning.

## Overview

Most clinical prediction models treat patient data as a flat feature vector, manually engineering features from each table. This project takes a different approach in which the raw relational structure of the database (patients, labs, medications, diagnoses, vitals) is modelled directly as a heterogeneous graph, and a GNN learns across table relationships automatically.

Five prediction tasks are implemented. Each task has an explicit observation window: the model only sees data recorded before its prediction time.

| Task | Dataset | Observation window | Type | Key Metric |
|------|---------|--------------------|------|------------|
| AKI Onset Prediction (binary) | eICU | First 12h, predicts next 48h | Binary classification | AUROC 0.7399 |
| AKI Severity Staging | eICU | First 12h, predicts next 48h | Ordinal (Stage ≥1 / ≥2 / ≥3) | AUROC 0.7133 / 0.7339 / 0.7718 |
| Vasopressor Initiation | eICU | First 6h, predicts next 12h | Binary classification | AUROC 0.8323 |
| ICD Code Prediction (top-50) | MIMIC-IV | At admission | Multi-label classification | Macro-AUROC 0.8407 |
| ICD Code Prediction (top-50) | MIMIC-IV | Full stay, to discharge | Multi-label classification | Macro-AUROC 0.8632 |
| Principal Diagnosis Prediction (top-50) | MIMIC-IV | At admission | Multi-class classification | Top-1 accuracy 0.2592 |

The two ICD rows are the same pipeline run with different observation windows. The admission-time model predicts discharge codes from demographics, registration fields and prior hospitalizations only; extending the window to discharge gives the model the full in-stay record and improves macro-AUPRC.

## Ablations and Experiments

Four additional experiments probe how the models behave under changed conditions.

**Label space scaling (ICD, multi-label).** Widening the label set keeps ranking performance stable but steadily hurts precision, as expected when rarer codes with fewer positives are added.

| Label space | Macro-AUROC | Macro-AUPRC |
|-------------|------------:|------------:|
| Top-50      | 0.8407      | 0.2708      |
| Top-100     | 0.8429      | 0.2082      |
| Top-200     | 0.8466      | 0.1610      |

**Principal diagnosis coverage trade-off (single-label).** Expanding the label space nearly doubles the share of admissions the model can make a prediction for, at the cost of accuracy.

| Label space | Admissions covered | Coverage | Top-1 accuracy | Top-5 accuracy |
|-------------|-------------------:|---------:|---------------:|---------------:|
| Top-50      | 17,898             | 23.1%    | 0.2592         | 0.6607         |
| Top-100     | 24,111             | 31.1%    | 0.1971         | 0.5139         |
| Top-200     | 32,035             | 41.3%    | 0.1651         | 0.4132         |

**APACHE table removed (AKI).** The APACHE severity score is computed from the patient's worst values across the first 24 hours, which extends beyond the model's 12-hour observation window. Removing `apachePatientResult` from the graph quantifies its contribution: test AUROC drops from 0.7399 to 0.7205 and AUPRC from 0.3385 to 0.3159.

**Observation window (ICD).** Re-running the ICD pipeline with the observation window extended from admission to discharge, reported in the two ICD rows above.

## Datasets

This project uses two credentialed clinical datasets:

- **eICU Collaborative Research Database v2.0** — 200,000+ ICU admissions across 208 US hospitals
- **MIMIC-IV v3.1** — 300,000+ hospital admissions from Beth Israel Deaconess Medical Center

Both datasets require credentialed access through [PhysioNet](https://physionet.org/). You must complete the required training and sign the data use agreement before downloading.

- eICU: https://physionet.org/content/eicu-crd/2.0/
- MIMIC-IV: https://physionet.org/content/mimiciv/3.1/

**Plan ahead for access.** In my case the process took roughly two weeks end to end: two to three days for the required training, and the rest waiting for the credentialing review and the supervisor reference to be processed. The data files cannot be redistributed, so anyone reproducing this work needs their own credentialed access.

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

> **Note:** A CUDA-capable GPU is strongly recommended. The first setup cell detects the runtime and falls back to CPU wheels automatically, but training on CPU is impractically slow for the full cohorts. Developed and tested on a Kaggle T4 instance and on Google Colab.

## Usage

1. Obtain access to eICU and MIMIC-IV from PhysioNet and download the CSV files (decompress the `.csv.gz` archives).
2. **On Kaggle:** attach both datasets to the notebook — the configuration cell detects their paths automatically. **Elsewhere:** set `EICU_DIR` and `MIMIC_DIR` in that cell to your local data folders. These are the only paths that need changing.
3. Run the notebook end to end: `notebook/clinical_outcome_prediction.ipynb`

Each task section is self-contained — you can run them independently after executing the setup and configuration cells at the top.

## Project Structure

```
clinical-outcome-prediction-gnn/
├── notebook/
│   └── clinical_outcome_prediction.ipynb
├── presentation/
│   └── clinical_outcome_prediction.pptx
├── README.md
├── requirements.txt
└── .gitignore
```
