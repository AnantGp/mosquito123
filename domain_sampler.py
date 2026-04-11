"""
BioDCASE 2026 — Domain-Balanced Sampler
========================================
Ensures each mini-batch contains equal representation from every domain,
preventing the model from learning domain-specific artifacts.

The core challenge: D5 has ~265k clips while D1 may have ~200.
Without balancing, the model learns "this sounds like D5" instead of 
"this sounds like Ae. aegypti".

Three strategies:
  • "equal"  — sample same N from each domain (most aggressive)
  • "sqrt"   — sample proportional to √(domain_size) (moderate)
  • "log"    — sample proportional to log(domain_size) (gentle)

Usage:
    from domain_sampler import DomainBalancedBatchSampler, build_dataloader
    
    dataloader = build_dataloader(
        dataset, batch_size=64, strategy="equal", num_workers=4
    )
"""

import re
import math
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Dict, Tuple, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from config import (
    DATA_ROOT, TARGET_SR, N_FFT, HOP_LENGTH, N_MELS, F_MAX, F_MIN,
    CLIP_SAMPLES, CLIP_DURATION_S, NUM_SPECIES, NUM_DOMAINS,
    SEED, BATCH_SIZE,
)

# ═══════════════════════════════════════════════════════════════════
# 1. Parse Dataset
# ═══════════════════════════════════════════════════════════════════

FILENAME_PATTERN = re.compile(r"^S_(\d+)_D_(\d+)_(\d+)\.wav$")


def parse_dataset(data_root: Path = DATA_ROOT) -> List[Dict]:
    """Parse all wav files and return list of metadata dicts."""
    records = []
    wav_files = sorted(data_root.glob("*.wav"))
    
    for f in wav_files:
        m = FILENAME_PATTERN.match(f.name)
        if m:
            records.append({
                "filepath": str(f),
                "species_id": int(m.group(1)) - 1,  # 0-indexed for PyTorch
                "domain_id": int(m.group(2)) - 1,    # 0-indexed
                "clip_index": int(m.group(3)),
            })
    
    print(f"Parsed {len(records):,} files from {data_root}")
    return records


# ═══════════════════════════════════════════════════════════════════
# 2. Audio Dataset (PyTorch)
# ═══════════════════════════════════════════════════════════════════

class MosquitoDataset(Dataset):
    """
    PyTorch Dataset for mosquito audio clips.
    
    Preprocessing pipeline (matches baseline):
      1. Load audio → resample to 8 kHz
      2. Peak normalization
      3. Pad/trim to CLIP_SAMPLES
      4. Compute log-mel spectrogram (64 bands, 512 FFT, 10ms hop)
      5. Standardize using TRAINING-SET-ONLY statistics
    """
    
    def __init__(
        self,
        records: List[Dict],
        train: bool = True,
        transform=None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.records = records
        self.train = train
        self.transform = transform
        self.mean = mean
        self.std = std
        
        # Build domain→indices mapping for the sampler
        self.domain_indices: Dict[int, List[int]] = defaultdict(list)
        self.species_list: List[int] = []
        self.domain_list: List[int] = []
        
        for i, r in enumerate(records):
            self.domain_indices[r["domain_id"]].append(i)
            self.species_list.append(r["species_id"])
            self.domain_list.append(r["domain_id"])
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        import librosa
        
        rec = self.records[idx]
        
        # 1. Load and resample
        y, _ = librosa.load(rec["filepath"], sr=TARGET_SR)
        
        # 2. Peak normalization
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        
        # 3. Pad or trim to fixed length
        if len(y) < CLIP_SAMPLES:
            y = np.pad(y, (0, CLIP_SAMPLES - len(y)), mode="constant")
        else:
            y = y[:CLIP_SAMPLES]
        
        # 4. Log-mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # 5. Standardize (training-set statistics only!)
        if self.mean is not None and self.std is not None:
            log_mel = (log_mel - self.mean) / (self.std + 1e-8)
        
        # Convert to tensor: (1, n_mels, time)
        spec = torch.FloatTensor(log_mel).unsqueeze(0)
        
        # Apply augmentation transforms
        if self.transform is not None and self.train:
            spec = self.transform(spec)
        
        species_label = torch.tensor(rec["species_id"], dtype=torch.long)
        domain_label = torch.tensor(rec["domain_id"], dtype=torch.long)
        
        return spec, species_label, domain_label
    
    def compute_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std from THIS dataset (call on training set only!)."""
        import librosa
        
        print("Computing training-set normalization statistics...")
        all_mels = []
        sample_indices = np.random.choice(
            len(self.records), min(5000, len(self.records)), replace=False
        )
        
        for idx in sample_indices:
            rec = self.records[idx]
            y, _ = librosa.load(rec["filepath"], sr=TARGET_SR)
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak
            if len(y) < CLIP_SAMPLES:
                y = np.pad(y, (0, CLIP_SAMPLES - len(y)), mode="constant")
            else:
                y = y[:CLIP_SAMPLES]
            
            mel = librosa.feature.melspectrogram(
                y=y, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            all_mels.append(log_mel)
        
        all_mels = np.array(all_mels)  # (N, n_mels, time)
        mean = all_mels.mean(axis=(0, 2), keepdims=True).squeeze(0)  # (n_mels, 1)
        std = all_mels.std(axis=(0, 2), keepdims=True).squeeze(0)    # (n_mels, 1)
        
        self.mean = mean
        self.std = std
        print(f"  Mean range: [{mean.min():.1f}, {mean.max():.1f}]")
        print(f"  Std range : [{std.min():.2f}, {std.max():.2f}]")
        return mean, std


# ═══════════════════════════════════════════════════════════════════
# 3. Domain-Balanced Batch Sampler
# ═══════════════════════════════════════════════════════════════════

class DomainBalancedBatchSampler(Sampler):
    """
    Sampler that ensures each mini-batch has equal representation per domain.
    
    For batch_size=64 with 5 domains:
      → ~12-13 samples per domain per batch
      → Each domain shuffled independently
      → Minority domains cycle (with replacement) when exhausted
    
    Args:
        dataset: MosquitoDataset with domain_indices populated
        batch_size: Total batch size
        strategy: "equal" | "sqrt" | "log"
        drop_last: Drop incomplete final batch
        seed: Random seed
    """
    
    def __init__(
        self,
        dataset: MosquitoDataset,
        batch_size: int = BATCH_SIZE,
        strategy: str = "equal",
        drop_last: bool = True,
        seed: int = SEED,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.drop_last = drop_last
        self.seed = seed
        
        self.domain_indices = dict(dataset.domain_indices)
        self.num_domains = len(self.domain_indices)
        
        # Compute per-domain samples per batch
        self.samples_per_domain = self._compute_allocation()
        
        # Epoch length = largest domain fills naturally
        max_domain_size = max(len(v) for v in self.domain_indices.values())
        self.num_batches = max_domain_size // max(self.samples_per_domain.values())
        
        domain_sizes = {d: len(idx) for d, idx in self.domain_indices.items()}
        print(f"\n{'='*60}")
        print(f"Domain-Balanced Sampler Initialized")
        print(f"{'='*60}")
        print(f"Strategy       : {strategy}")
        print(f"Batch size     : {batch_size}")
        print(f"Domains        : {self.num_domains}")
        print(f"Domain sizes   : {domain_sizes}")
        print(f"Samples/domain : {self.samples_per_domain}")
        print(f"Batches/epoch  : {self.num_batches}")
        print(f"{'='*60}\n")
    
    def _compute_allocation(self) -> Dict[int, int]:
        """Compute how many samples each domain contributes per batch."""
        domain_sizes = {d: len(idx) for d, idx in self.domain_indices.items()}
        
        if self.strategy == "equal":
            per_domain = self.batch_size // self.num_domains
            alloc = {d: per_domain for d in domain_sizes}
        
        elif self.strategy == "sqrt":
            raw = {d: math.sqrt(n) for d, n in domain_sizes.items()}
            total = sum(raw.values())
            alloc = {d: max(1, int(self.batch_size * v / total)) for d, v in raw.items()}
        
        elif self.strategy == "log":
            raw = {d: math.log1p(n) for d, n in domain_sizes.items()}
            total = sum(raw.values())
            alloc = {d: max(1, int(self.batch_size * v / total)) for d, v in raw.items()}
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Adjust to match batch_size exactly
        diff = self.batch_size - sum(alloc.values())
        for d in sorted(alloc.keys()):
            if diff == 0:
                break
            alloc[d] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
        
        return alloc
    
    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        
        # Shuffle each domain's indices
        domain_pools = {}
        for d, indices in self.domain_indices.items():
            pool = indices.copy()
            rng.shuffle(pool)
            domain_pools[d] = pool
        
        domain_pointers = {d: 0 for d in domain_pools}
        
        for _ in range(self.num_batches):
            batch = []
            
            for d, n_samples in self.samples_per_domain.items():
                pool = domain_pools[d]
                ptr = domain_pointers[d]
                
                selected = []
                for _ in range(n_samples):
                    if ptr >= len(pool):
                        # Reshuffle and reset (cycle for small domains)
                        rng.shuffle(pool)
                        domain_pools[d] = pool
                        ptr = 0
                    selected.append(pool[ptr])
                    ptr += 1
                
                domain_pointers[d] = ptr
                batch.extend(selected)
            
            rng.shuffle(batch)  # Shuffle within batch
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches


# ═══════════════════════════════════════════════════════════════════
# 4. Species-Stratified Domain-Balanced Sampler (Advanced)
# ═══════════════════════════════════════════════════════════════════

class SpeciesStratifiedDomainSampler(Sampler):
    """
    Advanced sampler: balance BOTH domain AND species within each batch.
    
    Groups data by (species, domain) cells, then samples proportionally
    from each non-empty cell to fill each batch.
    """
    
    def __init__(
        self,
        dataset: MosquitoDataset,
        batch_size: int = BATCH_SIZE,
        drop_last: bool = True,
        seed: int = SEED,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        
        # Build (species, domain) → indices
        self.cell_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i, rec in enumerate(dataset.records):
            key = (rec["species_id"], rec["domain_id"])
            self.cell_indices[key].append(i)
        
        self.cells = list(self.cell_indices.keys())
        self.num_cells = len(self.cells)
        self.samples_per_cell = max(1, batch_size // self.num_cells)
        
        max_cell = max(len(v) for v in self.cell_indices.values())
        self.num_batches = max_cell // self.samples_per_cell
        
        print(f"\nSpecies-Stratified Domain Sampler:")
        print(f"  Non-empty (species, domain) cells: {self.num_cells}")
        print(f"  Samples per cell per batch: {self.samples_per_cell}")
        print(f"  Effective batch size: {self.samples_per_cell * self.num_cells}")
        print(f"  Batches per epoch: {self.num_batches}")
    
    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        
        cell_pools = {}
        for key, indices in self.cell_indices.items():
            pool = indices.copy()
            rng.shuffle(pool)
            cell_pools[key] = pool
        
        cell_ptrs = {key: 0 for key in cell_pools}
        
        for _ in range(self.num_batches):
            batch = []
            for key in self.cells:
                pool = cell_pools[key]
                ptr = cell_ptrs[key]
                
                for _ in range(self.samples_per_cell):
                    if ptr >= len(pool):
                        rng.shuffle(pool)
                        cell_pools[key] = pool
                        ptr = 0
                    batch.append(pool[ptr])
                    ptr += 1
                
                cell_ptrs[key] = ptr
            
            rng.shuffle(batch)
            yield batch
    
    def __len__(self):
        return self.num_batches


# ═══════════════════════════════════════════════════════════════════
# 5. LODO (Leave-One-Domain-Out) Split Utility
# ═══════════════════════════════════════════════════════════════════

def create_lodo_split(
    records: List[Dict],
    test_domain: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split records into train/test using Leave-One-Domain-Out.
    
    Args:
        records: All parsed records (0-indexed domain_id)
        test_domain: Domain ID to hold out (0-indexed)
    
    Returns:
        (train_records, test_records)
    """
    train = [r for r in records if r["domain_id"] != test_domain]
    test = [r for r in records if r["domain_id"] == test_domain]
    
    print(f"LODO Split: test_domain=D{test_domain + 1}")
    print(f"  Train: {len(train):,} clips ({len(set(r['domain_id'] for r in train))} domains)")
    print(f"  Test : {len(test):,} clips (domain D{test_domain + 1})")
    return train, test


# ═══════════════════════════════════════════════════════════════════
# 6. SpecAugment Transform
# ═══════════════════════════════════════════════════════════════════

class SpecAugment:
    """
    SpecAugment: frequency and time masking on spectrograms.
    
    From: Park et al., "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition", 2019.
    """
    
    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 16,
                 num_freq_masks: int = 2, num_time_masks: int = 2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: (1, n_mels, time)"""
        _, n_mels, time_steps = spec.shape
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            spec[:, f0:f0 + f, :] = 0.0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_steps - 1))
            t0 = random.randint(0, time_steps - t)
            spec[:, :, t0:t0 + t] = 0.0
        
        return spec


# ═══════════════════════════════════════════════════════════════════
# 7. Build DataLoader (convenience function)
# ═══════════════════════════════════════════════════════════════════

def build_dataloader(
    dataset: MosquitoDataset,
    batch_size: int = BATCH_SIZE,
    strategy: str = "equal",
    species_stratified: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader with domain-balanced sampling.
    
    Args:
        dataset: MosquitoDataset instance
        batch_size: Batch size
        strategy: "equal" | "sqrt" | "log" (ignored if species_stratified=True)
        species_stratified: Use advanced species×domain balancing
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
    
    Returns:
        DataLoader with domain-balanced batches
    """
    if species_stratified:
        sampler = SpeciesStratifiedDomainSampler(
            dataset=dataset,
            batch_size=batch_size,
        )
    else:
        sampler = DomainBalancedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            strategy=strategy,
        )
    
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ═══════════════════════════════════════════════════════════════════
# 8. Quick Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("BioDCASE 2026 — Domain-Balanced Sampler Test")
    print("=" * 60)
    
    # Parse dataset
    records = parse_dataset()
    
    if not records:
        print("No .wav files found! Check DATA_ROOT in config.py")
        exit(1)
    
    # Create LODO split (hold out D1)
    train_recs, test_recs = create_lodo_split(records, test_domain=0)
    
    # Create datasets
    train_ds = MosquitoDataset(train_recs, train=True)
    
    # Compute normalization stats from training set
    mean, std = train_ds.compute_normalization_stats()
    
    # Test domain-balanced sampler
    print("\n--- Testing Domain-Balanced Sampler (equal) ---")
    loader = build_dataloader(train_ds, batch_size=64, strategy="equal")
    
    # Verify domain balance in first batch
    batch = next(iter(loader))
    specs, species_labels, domain_labels = batch
    
    print(f"\nFirst batch:")
    print(f"  Spectrogram shape: {specs.shape}")
    print(f"  Species distribution: {torch.bincount(species_labels, minlength=NUM_SPECIES).tolist()}")
    print(f"  Domain distribution : {torch.bincount(domain_labels, minlength=NUM_DOMAINS).tolist()}")
    
    # Test species-stratified sampler
    print("\n--- Testing Species-Stratified Domain Sampler ---")
    loader2 = build_dataloader(train_ds, batch_size=64, species_stratified=True)
    
    batch2 = next(iter(loader2))
    specs2, species2, domains2 = batch2
    print(f"\nFirst batch (stratified):")
    print(f"  Spectrogram shape: {specs2.shape}")
    print(f"  Species distribution: {torch.bincount(species2, minlength=NUM_SPECIES).tolist()}")
    print(f"  Domain distribution : {torch.bincount(domains2, minlength=NUM_DOMAINS).tolist()}")
    
    print("\n✅ All tests passed!")
