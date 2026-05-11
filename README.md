# WUT-TrajData

WUT-TrajData is an AIS-based vessel trajectory dataset and benchmark for maritime trajectory prediction.

This repository provides:

- AIS trajectory preprocessing pipeline
- Benchmark organization
- Baseline implementations
- Adapted external trajectory prediction models
- Evaluation tools
- Visualization utilities

---

# Overview

Automatic Identification System (AIS) data provide large-scale vessel movement records and are widely used in:

- maritime traffic analysis
- vessel behavior understanding
- trajectory prediction
- autonomous navigation research

WUT-TrajData is designed to support systematic evaluation of vessel trajectory prediction methods under representative maritime scenarios.

The dataset currently contains two representative maritime regions:

| Region | Scenario |
|---|---|
| CSJ | Open-sea corridor |
| ZS | Port and anchorage area |

The benchmark includes multiple categories of trajectory prediction models:

- Physics-based methods
- Classical machine learning methods
- Basic deep learning models
- Transformer-based methods
- Interaction-aware models
- Generative models

---

# Repository Structure

```text
WUT-TrajData/
├── data/                  # Dataset description and sample data
├── preprocess/            # AIS cleaning, segmentation, resampling
├── benchmark/             # Benchmark models
│   ├── simple_baselines/  # Physics-based and classical ML baselines
│   ├── basic_dl/          # LSTM, GRU, BiLSTM, Transformer
│   ├── external_models/   # Adapted open-source models
│   └── adapters/          # Data converters for external models
├── evaluation/            # Evaluation metrics and result aggregation
├── configs/               # Dataset and benchmark configuration files
├── scripts/               # Running scripts
├── visualization/         # Visualization tools
├── outputs/               # Generated outputs (ignored by Git)
└── docs/                  # Additional documentation
```

---

# Benchmark Organization

## 1. Simple Baselines

`benchmark/simple_baselines/`

Contains physics-based and classical machine learning baselines.

Examples:

- Constant Velocity (CV)
- Constant Acceleration (CA)
- Kalman Filter (KF)
- Extended Kalman Filter (EKF)
- KNN
- SVR
- HMM

These models are implemented under a relatively unified script structure.

---

## 2. Basic Deep Learning Models

`benchmark/basic_dl/`

Contains basic deep learning models implemented under a unified framework.

Examples:

- LSTM
- GRU
- BiLSTM
- Transformer

---

## 3. External Models

`benchmark/external_models/`

Contains adapted open-source trajectory prediction models.

Examples:

- LED
- TUTR
- Social-LSTM
- Social-GAN
- PECNet
- ST-MGT
- MSRL
- TrAISformer

For these models, the original project structures are preserved as much as possible.

WUT-TrajData-specific modifications, scripts, and data conversion tools are provided separately for each model.

---

# Data

The full AIS dataset is not directly included in this repository due to data size.

Please refer to:

```text
data/README.md
```

for:

- dataset format
- sample data
- access instructions

Expected local data structure:

```text
data/
├── raw/
├── processed/
└── samples/
```

The `raw/` and `processed/` folders are ignored by Git.

---

# Lightweight Entry Point

This repository provides a lightweight entry point:

```bash
python run.py --list
```

List all available benchmark models.

Show information for a specific model:

```bash
python run.py --group simple --model cv
python run.py --group basic_dl --model lstm
python run.py --group external --model led
```

Due to the heterogeneous structures of adapted external models, detailed training and evaluation instructions are provided in each model folder.

---

# Evaluation Metrics

The evaluation module supports commonly used trajectory prediction metrics, including:

- ADE
- FDE
- minADE
- minFDE
- DTW
- Hausdorff Distance
- Miss Rate

Detailed metric definitions and result formats will be provided in:

```text
evaluation/README.md
docs/benchmark_protocol.md
```

---

# License

To be determined.

---

# Acknowledgement

Some benchmark models are adapted from existing open-source implementations.

We thank the original authors for their contributions.