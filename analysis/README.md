# Dataset Analysis

This folder contains the dataset analysis code for WUT-TrajData.

The main script `ais_evaluate_final.py` provides an OpenTraj-inspired analysis pipeline adapted for AIS-based vessel trajectories.

The analysis includes trajectory statistics and complexity-related metrics, such as:

- speed statistics
- acceleration statistics
- trajectory efficiency
- angular deviation
- local density
- CPA-related metrics
- entropy-based complexity

Generated results should be saved under `analysis/outputs/`, which is ignored by Git.