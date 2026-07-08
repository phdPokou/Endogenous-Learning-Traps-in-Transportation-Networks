# Endogenous Learning Traps in Transportation Networks

This repository contains the empirical replication code for the paper:

> **Endogenous Learning Traps in Transportation Networks**

The code reproduces the Porto taxi analysis reported in the paper, including:

- construction and validation of the trajectory sample;
- route-family and repeated-history construction;
- measurement of selective-feedback incompleteness;
- identification of supported better-alternative opportunities;
- construction of the primary and strong empirical learning-trap diagnostics;
- event-time and within-history analyses;
- decision-level inference with taxi-clustered resampling;
- information-completion benchmarks;
- threshold-sensitivity analysis; and
- the tables and figures reported in the manuscript.

The repository is intended to reproduce the empirical analysis of the paper. It does not estimate the latent beliefs, observational-equivalence sets, or behavioral-support sets appearing in the theoretical model.

---

## Data

The empirical analysis uses the publicly available **Taxi Service Trajectory — Prediction Challenge, ECML PKDD 2015** dataset from the UCI Machine Learning Repository.

**Official download page:**

https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015

The dataset contains trajectories recorded for the Porto taxi fleet over one year.

## 🚀 Running the Pipeline

To reproduce the core empirical inference results, bootstrap confidence intervals, and decision-level post-opportunity gradients reported in the paper, execute the main execution script from your terminal:

```bash
python porto_learning_traps_v12_final_inference.py \
    --data train.csv.zip \
    --out Results_Porto_V12 \
    --seeds 30


