# KIAD — Knowledge-Enhanced Intrinsic-Adaptive Diffusion for Out-of-Town Trajectory Recommendation

A knowledge-enhanced intrinsic-adaptive diffusion framework for out-of-town trajectory recommendation.

## Overview

KIAD is a unified knowledge-enhanced intrinsic-adaptive diffusion framework for **Out-of-Town Trajectory Recommendation**. Given a user's home-city historical check-in sequence `Hu`, a POI attribute knowledge graph `Gk`, and a conditional travel query `qu`, the model autoregressively generates a complete POI visit trajectory in the target city.

KIAD jointly addresses three core challenges: cross-city preference drift, spatio-temporal behavioral evolution, and generative decision uncertainty. By integrating knowledge graph semantics, spatio-temporal behavioral modeling, and conditional diffusion-based generation, KIAD achieves significant improvements over the SPOT-Trip baseline on both Foursquare and Yelp datasets.

### Three Core Modules

| Module | Name | Function |
|--------|------|----------|
| **KIAM** | Knowledge-Aware Intrinsic Attention Module | Extracts user long-term intrinsic preferences `p_intrinsic` from home-city check-in sequences and POI knowledge graphs, including relation-aware neighbor attention, identity-KG gated fusion, sequence-level adaptive aggregation, and Source-Label InfoNCE contrastive learning |
| **STAPM** | Spatio-Temporal Adaptive Preference Module | Models dynamic preference evolution during out-of-town travel, producing adaptive state sequence `H_adaptive`, including adaptive spatio-temporal tensor construction, temporal/spatial feature encoding, Bi-GRU sequence modeling, Stiefel Graph Fourier Transform (SGFT) spectral refinement, and time-aware gated evolution |
| **CP-Diff** | Conditional Preference Diffusion Module | Generates multi-modal next-step intent vectors `g_diff` conditioned on intrinsic preferences, adaptive states, and time deltas via VP-SDE forward diffusion and reverse denoising, fused with deterministic intent through adaptive gating |

## Datasets

Two real-world datasets are supported (following the SPOT-Trip preprocessing pipeline):

- **Foursquare** — 3,007 users, 23,884 POIs, 21 regions, 47,768 KG triples
- **Yelp** — 4,417 users, 29,930 POIs, 214 regions, 353,918 KG triples

## Project Structure

```
KIAD_Code/
├── main.py                # Entry point: two-stage training, evaluation, random hyperparameter search
├── model.py               # KIAD main model (coordinates KIAM / STAPM / CP-Diff modules)
├── dataset.py             # Data loader: OD-pair-aware partitioning, dynamic max-length scan, multi-source feature construction
├── kiam.py                # KIAM: Knowledge-Aware Intrinsic Attention Module (gated fusion + InfoNCE)
├── stapm.py               # STAPM: Spatio-Temporal Adaptive Preference Module (Bi-GRU + Stiefel GCN + gated evolution)
├── cpdiff.py              # CP-Diff: Conditional Preference Diffusion Module (VP-SDE + adaptive gated fusion)
├── transe.py              # TransE / TransH knowledge graph embedding pre-training
├── new_metrics.py         # Evaluation metrics: F1, PairsF1, Full-F1, Full-PairsF1, adjacent repetition rate
├── utils.py               # Utilities: seed setting, top-np sampling, Metrics class
├── config_Foursquare.py   # Foursquare dataset configuration
├── config_Yelp.py         # Yelp dataset configuration
├── Foursquare/            # Foursquare dataset
│   ├── home.txt           # Home-city check-in sequences
│   ├── oot.txt            # Out-of-town check-in sequences
│   ├── travel.txt         # Travel information (OD pairs and cities)
│   ├── kg.txt             # POI attribute knowledge graph triples
│   ├── poi_id.pkl         # POI string ID → integer ID mapping
│   ├── poi_coord.pkl      # POI latitude/longitude coordinates
│   ├── region_poi.pkl     # Region → POI set mapping
│   └── city_tz_mapping.pkl
└── Yelp/                  # Yelp dataset (same structure)
```

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, GPU auto-detection)

### Install Dependencies

```bash
pip install torch numpy tqdm pytz
```

## Usage

```bash
# Train with Foursquare dataset
python main.py --config config_Foursquare

# Train with Yelp dataset
python main.py --config config_Yelp
```

## Training Pipeline

KIAD employs a **two-stage training pipeline**:

1. **Stage 1 — KG Pre-training:** TransE/TransH pre-training on POI attribute knowledge graph triples to obtain entity and relation semantic embeddings
2. **Stage 2 — End-to-End Joint Training:** Transfer KG embeddings to POI embedding layers and optimize the full framework end-to-end

**Grouped Multi-Objective Joint Loss:**

```
L_train = λ_rec · L_rec + λ_diff · L_diff + L_intrinsic
```

where `L_rec` is the cross-entropy recommendation loss, `L_diff` is the diffusion noise reconstruction loss (MSE), and `L_intrinsic` includes contrastive learning loss `L_cl` and preference alignment loss `L_mse`. Auxiliary regularization terms (KG loss, ranking loss, gating loss) are attributed to their corresponding branches.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **F1** | Intermediate trajectory POI matching F1 (excluding start/end points) |
| **PairsF1** | Intermediate trajectory POI pairwise order consistency F1 |
| **Full-F1** | Full trajectory F1 (including start/end points) |
| **Full-PairsF1** | Full trajectory pairwise order consistency F1 |

Model selection is based on a weighted objective score: **45% Full-F1 + 55% Full-PairsF1**. All metrics are reported as averages over three independent runs.

## Key Features

- **Multi-Source Embedding:** POI representation uses `e_poi = e_id + e_cat + e_region` additive multi-source embeddings to mitigate cold-start and sparsity issues
- **Knowledge-Aware Gated Fusion:** KIAM adaptively fuses POI identity embeddings with KG neighbor semantics via a gating mechanism that automatically suppresses noisy relations
- **Source-Label InfoNCE:** Treats home-city and out-of-town check-ins as distinct source labels, constructing contrastive learning objectives to enhance the discriminability of intrinsic preferences
- **Stiefel Manifold Spectral Refinement:** STAPM constructs an adaptive sequence-level graph and enhances structural signals via Stiefel Graph Fourier Transform (SGFT)
- **Time-Aware Gated Evolution:** `h_i = g_i ⊙ h_refined,i + (1 - g_i) ⊙ h_{i-1}`, maintaining temporal sensitivity across both high-frequency short-hop and low-frequency long-hop scenarios
- **VP-SDE Conditional Diffusion:** CP-Diff adopts Variance-Preserving Stochastic Differential Equations with a conditional denoising network (3-layer MLP + sinusoidal time embeddings) to learn multi-modal intent distributions
- **Adaptive Gated Fusion (Inference):** `f_i = i_i + α_i ⊙ g_diff,i`, where `α_i = σ(GateMLP([i_i ∥ g_diff,i]))` learns the optimal fusion ratio between diffusion output and deterministic intent
- **Target Region Masking:** Non-target city/region POI logits are set to -∞ during both training and inference to enforce cross-city constraints
- **OD-Pair-Aware Data Split:** Stratified 80/10/10 split by Origin-Destination pairs to ensure generalization to unseen OD pairs
- **Ablation Framework:** All three core modules can be independently ablated (output zeroed) or replaced with SPOT-Trip counterparts, enabling comprehensive module effectiveness validation

## Configuration Parameters

| Parameter | Foursquare | Yelp | Description |
|-----------|-----------|------|-------------|
| batch_size | 64 | 128 | Batch size |
| lr | 0.0008 | 0.0008 | Learning rate |
| embed_dim | 256 | 128 | Embedding dimension d |
| hidden_dim | 512 | 1024 | MLP hidden dimension |
| max_seq_len | 21 | 20 | Home-city sequence max length M |
| oot_seq_len | 10 | 12 | Out-of-town sequence length N |
| diff_steps | 100 | 100 | Diffusion steps T_d |
| beta_start | 0.0001 | 0.0001 | Noise schedule start value |
| beta_end | 0.02 | 0.02 | Noise schedule end value |
| stiefel_k | 64 | 64 | Stiefel manifold projection dimension |
| λ_rec | 2.0 | 2.1 | Reconstruction loss weight |
| λ_diff | 1.0 | 1.0 | Diffusion loss weight |
| epochs | 500 | 500 | Maximum training epochs |
| patience | 10 | 10 | Early stopping patience |

## Citation

```bibtex
@article{zhu2025kiad,
  title   = {KIAD: Knowledge-Enhanced Intrinsic-Adaptive Diffusion for
             Out-of-Town Trajectory Recommendation},
  author  = {Jiabao Zhu and Xiangle Lei and Sijie Ruan and Shuliang Wang},
  journal = {},
  year    = {2025}
}
```

## License

This project is for academic research purposes only.
