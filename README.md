# WUT-TrajData

WUT-TrajData is an AIS-based vessel trajectory dataset and benchmark for maritime trajectory prediction.

This repository provides the dataset description, preprocessing pipeline, benchmark protocols, baseline implementations, adapted external models, and evaluation tools for vessel trajectory prediction.

## Overview

WUT-TrajData contains AIS vessel trajectories from two representative maritime scenarios:

- **CSJ**: open-sea corridor scenario
- **ZS**: port and anchorage scenario

The benchmark includes physics-based models, classical machine learning models, basic deep learning models, Transformer-based models, interaction-aware models, and generative models.

## Repository Structure

```text
WUT-TrajData/
├── data/                  # Dataset description and sample data
├── preprocess/            # AIS cleaning, segmentation, and resampling
├── benchmark/             # Benchmark models
│   ├── simple_baselines/  # Physics-based and classical ML baselines
│   ├── basic_dl/          # LSTM, GRU, BiLSTM, Transformer
│   ├── external_models/   # Adapted open-source models
│   └── adapters/          # Data format converters for external models
├── evaluation/            # Evaluation metrics and result aggregation
├── configs/               # Dataset and benchmark configuration files
├── scripts/               # Running scripts
├── visualization/         # Visualization tools
├── outputs/               # Generated outputs, ignored by Git
└── docs/                  # Additional documentation
