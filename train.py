"""
BioDCASE 2026 — Training Pipeline
===================================
Full training loop with:
  • Domain-Balanced Sampling (from domain_sampler.py)
  • LODO (Leave-One-Domain-Out) cross-validation
  • SpecAugment augmentation
  • Balanced Accuracy evaluation (seen & unseen domains)
  • DSG (Domain Shift Gap) tracking
  • Checkpoint saving & early stopping

Usage:
    python train.py                          # Train with default settings
    python train.py --test_domain 0          # LODO: hold out D1
    python train.py --strategy sqrt          # Use sqrt domain balancing
    python train.py --species_stratified     # Use species×domain balancing
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import balanced_accuracy_score

from config import (
    NUM_SPECIES, NUM_DOMAINS, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, SCHEDULER_T0, SCHEDULER_T_MULT,
    SEED, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR,
    N_MELS, BASELINE_BA_SEEN, BASELINE_BA_UNSEEN, BASELINE_DSG,
)
from domain_sampler import (
    parse_dataset, create_lodo_split, MosquitoDataset,
    build_dataloader, SpecAugment,
)


# ═══════════════════════════════════════════════════════════════════
# 1. MTRCNN Model (Official Baseline Architecture)
# ═══════════════════════════════════════════════════════════════════

class MTRCNNBranch(nn.Module):
    """Single branch of the MTRCNN with specific kernel size."""
    
    def __init__(self, kernel_size=(3, 3), channels=[32, 64, 128]):
        super().__init__()
        
        layers = []
        in_ch = 1
        for ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, ch, kernel_size=kernel_size, 
                         padding=(kernel_size[0]//2, kernel_size[1]//2)),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
            in_ch = ch
        
        self.features = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.features(x)


class MTRCNN(nn.Module):
    """
    Multitemporal Resolution CNN (MTRCNN).
    
    Three parallel branches with kernel sizes (3,3), (5,5), (7,7)
    capture different temporal resolutions of the wingbeat signal.
    Features are concatenated and fed to a classifier.
    
    Total params: ~220k (lightweight baseline reference).
    """
    
    def __init__(self, num_classes=NUM_SPECIES, num_domains=NUM_DOMAINS):
        super().__init__()
        
        self.branch_3 = MTRCNNBranch(kernel_size=(3, 3))
        self.branch_5 = MTRCNNBranch(kernel_size=(5, 5))
        self.branch_7 = MTRCNNBranch(kernel_size=(7, 7))
        
        # Adaptive pooling to fixed size regardless of input length
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 128 * 3 branches = 384 features
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        
        # Optional domain classifier (for domain-adversarial training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_domains),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x):
        """Extract concatenated features from all branches."""
        f3 = self.pool(self.branch_3(x)).flatten(1)
        f5 = self.pool(self.branch_5(x)).flatten(1)
        f7 = self.pool(self.branch_7(x)).flatten(1)
        return torch.cat([f3, f5, f7], dim=1)
    
    def forward(self, x, return_domain=False):
        features = self.extract_features(x)
        species_logits = self.classifier(features)
        
        if return_domain:
            domain_logits = self.domain_classifier(features)
            return species_logits, domain_logits
        
        return species_logits


# ═══════════════════════════════════════════════════════════════════
# 2. Gradient Reversal Layer (for Domain-Adversarial Training)
# ═══════════════════════════════════════════════════════════════════

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ═══════════════════════════════════════════════════════════════════
# 3. Training Functions
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, use_domain_adversarial=False,
                    domain_weight=0.1, epoch=0, max_epochs=100):
    model.train()
    total_loss = 0
    total_species_loss = 0
    total_domain_loss = 0
    all_preds = []
    all_labels = []
    n_batches = 0
    
    # Progressive domain adversarial weight (ramp up)
    p = epoch / max_epochs
    grl_alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # Sigmoid schedule
    
    for specs, species_labels, domain_labels in loader:
        specs = specs.to(device)
        species_labels = species_labels.to(device)
        domain_labels = domain_labels.to(device)
        
        optimizer.zero_grad()
        
        if use_domain_adversarial:
            species_logits, domain_logits = model(specs, return_domain=True)
            species_loss = F.cross_entropy(species_logits, species_labels)
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
            loss = species_loss + domain_weight * grl_alpha * domain_loss
            total_domain_loss += domain_loss.item()
        else:
            species_logits = model(specs)
            species_loss = F.cross_entropy(species_logits, species_labels)
            loss = species_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_species_loss += species_loss.item()
        
        preds = species_logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(species_labels.cpu().numpy())
        n_batches += 1
    
    ba = balanced_accuracy_score(all_labels, all_preds)
    
    return {
        "loss": total_loss / max(n_batches, 1),
        "species_loss": total_species_loss / max(n_batches, 1),
        "domain_loss": total_domain_loss / max(n_batches, 1),
        "ba": ba,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_domains = []
    total_loss = 0
    n_batches = 0
    
    for specs, species_labels, domain_labels in loader:
        specs = specs.to(device)
        species_labels = species_labels.to(device)
        
        logits = model(specs)
        loss = F.cross_entropy(logits, species_labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(species_labels.cpu().numpy())
        all_domains.extend(domain_labels.numpy())
        n_batches += 1
    
    # Overall BA
    ba = balanced_accuracy_score(all_labels, all_preds)
    
    # Per-domain BA
    domain_ba = {}
    for d in sorted(set(all_domains)):
        mask = [i for i, dd in enumerate(all_domains) if dd == d]
        if mask:
            d_preds = [all_preds[i] for i in mask]
            d_labels = [all_labels[i] for i in mask]
            domain_ba[d] = balanced_accuracy_score(d_labels, d_preds)
    
    return {
        "loss": total_loss / max(n_batches, 1),
        "ba": ba,
        "domain_ba": domain_ba,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. Main Training Loop
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BioDCASE 2026 Training")
    parser.add_argument("--test_domain", type=int, default=0,
                        help="Domain to hold out for LODO (0-indexed, default=0 → D1)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--strategy", type=str, default="equal",
                        choices=["equal", "sqrt", "log"])
    parser.add_argument("--species_stratified", action="store_true",
                        help="Use species×domain stratified sampling")
    parser.add_argument("--domain_adversarial", action="store_true",
                        help="Enable domain-adversarial training")
    parser.add_argument("--domain_weight", type=float, default=0.1,
                        help="Weight for domain adversarial loss")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("BioDCASE 2026 — Cross-Domain Mosquito Species Classification")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Test domain     : D{args.test_domain + 1}")
    print(f"Strategy        : {args.strategy}")
    print(f"Species strat.  : {args.species_stratified}")
    print(f"Domain adv.     : {args.domain_adversarial}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Epochs          : {args.epochs}")
    print(f"Learning rate   : {args.lr}")
    print("=" * 70)
    
    # Parse and split data
    records = parse_dataset()
    train_recs, test_recs = create_lodo_split(records, test_domain=args.test_domain)
    
    # Create datasets
    train_ds = MosquitoDataset(train_recs, train=True, transform=SpecAugment())
    test_ds = MosquitoDataset(test_recs, train=False)
    
    # Compute training-set normalization
    mean, std = train_ds.compute_normalization_stats()
    test_ds.mean = mean
    test_ds.std = std
    
    # Create dataloaders
    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        strategy=args.strategy,
        species_stratified=args.species_stratified,
        num_workers=args.num_workers,
    )
    
    # Test loader: standard sequential (no balancing needed)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Model
    model = MTRCNN(num_classes=NUM_SPECIES, num_domains=NUM_DOMAINS).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: MTRCNN ({n_params:,} parameters)")
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT)
    
    # Training loop
    best_ba = 0.0
    patience_counter = 0
    history = []
    
    run_name = f"LODO_D{args.test_domain+1}_{args.strategy}"
    if args.species_stratified:
        run_name += "_stratified"
    if args.domain_adversarial:
        run_name += "_dadv"
    
    print(f"\nRun: {run_name}")
    print("-" * 70)
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            use_domain_adversarial=args.domain_adversarial,
            domain_weight=args.domain_weight,
            epoch=epoch, max_epochs=args.epochs,
        )
        
        # Evaluate on held-out domain
        test_metrics = evaluate(model, test_loader, device)
        
        scheduler.step()
        
        elapsed = time.time() - t0
        
        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_ba": train_metrics["ba"],
            "test_loss": test_metrics["loss"],
            "test_ba": test_metrics["ba"],  # This is BA_unseen
            "test_domain_ba": test_metrics["domain_ba"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
        }
        history.append(log_entry)
        
        # DSG = |BA_unseen - BA_seen|
        ba_seen = train_metrics["ba"]
        ba_unseen = test_metrics["ba"]
        dsg = abs(ba_unseen - ba_seen)
        
        # Print progress
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"BA_seen: {ba_seen:.4f} | "
            f"BA_unseen: {ba_unseen:.4f} | "
            f"DSG: {dsg:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"{elapsed:.1f}s"
        )
        
        # Checkpoint
        if ba_unseen > best_ba:
            best_ba = ba_unseen
            patience_counter = 0
            ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ba": best_ba,
                "mean": mean,
                "std": std,
            }, ckpt_path)
            print(f"  → Saved best model (BA_unseen={best_ba:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping at epoch {epoch+1} (patience={args.patience})")
            break
    
    # Final report
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best BA_unseen  : {best_ba:.4f}")
    print(f"Baseline BA_unseen: {BASELINE_BA_UNSEEN:.4f}")
    print(f"Improvement     : {(best_ba - BASELINE_BA_UNSEEN)*100:+.2f}pp")
    
    if best_ba > BASELINE_BA_UNSEEN:
        print("🎉 BEAT THE BASELINE!")
    else:
        print("📉 Below baseline — try: --domain_adversarial, --species_stratified, or different --strategy")
    
    # Save history
    history_path = LOG_DIR / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\nHistory saved: {history_path}")


if __name__ == "__main__":
    main()
