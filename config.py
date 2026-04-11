"""
BioDCASE 2026 — CD-MSC Configuration Module
=============================================
Centralised settings for preprocessing, model, and training.
Matches the official baseline (MTRCNN, arXiv 2603.20118).
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Paths (GCP Instance)
# ═══════════════════════════════════════════════════════════════════
DATA_ROOT = Path.home() / "Development_data" / "Development_data"
OUTPUT_DIR = Path.home() / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# ═══════════════════════════════════════════════════════════════════
# Species & Domain Mappings
# ═══════════════════════════════════════════════════════════════════
NUM_SPECIES = 9
NUM_DOMAINS = 5

SPECIES_MAP = {
    1: "Ae. aegypti",
    2: "Ae. albopictus",
    3: "Cx. quinquefasciatus",
    4: "An. gambiae",
    5: "An. arabiensis",
    6: "An. dirus",
    7: "Cx. pipiens",
    8: "An. minimus",
    9: "An. stephensi",
}

DOMAIN_MAP = {i: f"D{i}" for i in range(1, NUM_DOMAINS + 1)}

# ═══════════════════════════════════════════════════════════════════
# Audio Preprocessing (matches baseline exactly)
# ═══════════════════════════════════════════════════════════════════
TARGET_SR = 8000          # Resample to 8 kHz
N_FFT = 512               # FFT window size
HOP_LENGTH = 80           # 10 ms hop at 8 kHz
N_MELS = 64               # Number of mel filter banks
F_MIN = 0                 # Min frequency
F_MAX = 4000              # Max frequency (Nyquist at 8 kHz)
CLIP_DURATION_S = 1.28    # Baseline uses 1.28s clips
CLIP_SAMPLES = int(CLIP_DURATION_S * TARGET_SR)  # 10240 samples

# ═══════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealingWarmRestarts"
SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2

# Domain-Balanced Sampler settings
SAMPLES_PER_DOMAIN_PER_EPOCH = None  # None = auto-balance to smallest domain
DOMAIN_BALANCE_STRATEGY = "equal"    # "equal" | "sqrt" | "log"

# ═══════════════════════════════════════════════════════════════════
# Augmentation
# ═══════════════════════════════════════════════════════════════════
AUG_PITCH_SHIFT_RANGE = (-2, 2)      # semitones
AUG_TIME_STRETCH_RANGE = (0.8, 1.2)  # rate factor
AUG_NOISE_SNR_DB = (5, 20)           # signal-to-noise ratio (dB)
AUG_SPEC_FREQ_MASK = 8               # SpecAugment: max freq bands to mask
AUG_SPEC_TIME_MASK = 16              # SpecAugment: max time steps to mask
AUG_MIXUP_ALPHA = 0.3                # Mixup alpha parameter

# ═══════════════════════════════════════════════════════════════════
# Model (MTRCNN Baseline Reference)
# ═══════════════════════════════════════════════════════════════════
MODEL_NAME = "MTRCNN"
MTRCNN_CHANNELS = [32, 64, 128]
MTRCNN_KERNEL_SIZES = [(3, 3), (5, 5), (7, 7)]  # Three parallel branches
MTRCNN_DILATION_RATES = [1, 2, 4]               # Hybrid dilated convolution

# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════
# Baseline results for reference
BASELINE_BA_SEEN = 0.8806
BASELINE_BA_UNSEEN = 0.1751
BASELINE_DSG = 0.7055

# LODO splits used in baseline
LODO_FOLDS = {
    "fold1": {"test_domain": 1, "train_domains": [2, 3, 4, 5]},
    "fold2": {"test_domain": 2, "train_domains": [1, 3, 4, 5]},
    "fold3": {"test_domain": 3, "train_domains": [1, 2, 4, 5]},
    "fold4": {"test_domain": 4, "train_domains": [1, 2, 3, 5]},
    "fold5": {"test_domain": 5, "train_domains": [1, 2, 3, 4]},
}

# Random seed
SEED = 42
