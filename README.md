# BioDCASE 2026 — Cross-Domain Mosquito Species Classification (CD-MSC)

Competition entry for the [BioDCASE 2026 Challenge](https://arxiv.org/abs/2603.20118).

## Contribution

Anant Gupta reviewed related research, designed the spectrogram structure, developed the approach and contributed code. The Git history also records contributions under the name Aditya Gupta; commit metadata alone does not establish each person's complete contribution.

## Challenge
Classify 9 mosquito species across 5 recording domains, optimizing for **unseen-domain generalization** (`BA_unseen`).

The figures below are challenge baseline figures, **not** this repository's final competition score.

| Metric | Baseline (MTRCNN) |
|--------|------------------|
| BA_seen | 0.8806 |
| BA_unseen | 0.1751 |
| DSG | 0.7055 |

## Files

| File | Description |
|------|-------------|
| `config.py` | Centralized configuration (paths, preprocessing, training) |
| `domain_sampler.py` | Domain-Balanced Sampler, Dataset, LODO splits, SpecAugment |
| `train.py` | Training pipeline with MTRCNN + domain-adversarial option |
| `BioDCASE2026_EDA.ipynb` | Comprehensive EDA notebook |

## Processing overview

```text
Audio recordings + species/domain labels
  -> waveform loading and preprocessing
  -> log-mel spectrograms
  -> domain-balanced sampling and optional SpecAugment
  -> MTRCNN training, optionally with domain-adversarial learning
  -> held-out-domain evaluation
```

The intended hold-out-domain workflow is described in the training code. The repository does not contain a verified final competition score, trained weights, or a self-contained copy of the challenge data. Results and reproducibility depend on the dataset, environment, configuration and evaluation split.

## Quick Start (GCP)

```bash
# Install dependencies
pip install librosa torch torchaudio scikit-learn tqdm seaborn matplotlib pandas

# Unzip dataset (if not already done)
cd ~ && unzip Development_data.zip

# Test the domain-balanced sampler
python domain_sampler.py

# Train: LODO hold-out D1, domain-balanced + adversarial
python train.py --test_domain 0 --strategy equal --domain_adversarial

# Advanced: species×domain stratified sampling
python train.py --test_domain 0 --species_stratified --domain_adversarial
```

## Key Innovation: Domain-Balanced Sampling
D5 has ~265k clips vs D1 with ~200. Without balancing, the model memorizes domain artifacts instead of species cues. Our sampler forces equal domain representation per mini-batch.

## Dataset
- 271,380 clips | 60.66 hours | 9 species | 5 domains
- Format: `S_<speciesID>_D_<domainID>_<clipIndex>.wav`
- [Zenodo Record 19095788](https://zenodo.org/records/19095788)
