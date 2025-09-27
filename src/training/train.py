"""Training entrypoint for MNIST digit classifier."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.mnist import DataConfig, create_loaders
from src.models.cnn import MNISTNet


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


@dataclass
class TrainingSummary:
    epochs: List[EpochLog]
    test_accuracy: float
    test_loss: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "epochs": [asdict(epoch) for epoch in self.epochs],
            "test_accuracy": self.test_accuracy,
            "test_loss": self.test_loss,
        }


@dataclass
class TrainingConfig:
    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.1
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts_dir: Path = Path("artifacts")

    @classmethod
    def from_json(cls, path: Path) -> "TrainingConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            epochs=data.get("epochs", cls.epochs),
            batch_size=data.get("batch_size", cls.batch_size),
            learning_rate=data.get("learning_rate", cls.learning_rate),
            weight_decay=data.get("weight_decay", cls.weight_decay),
            val_split=data.get("val_split", cls.val_split),
            num_workers=data.get("num_workers", cls.num_workers),
            seed=data.get("seed", cls.seed),
            device=data.get("device", cls.device),
            artifacts_dir=Path(data.get("artifacts_dir", cls.artifacts_dir)),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    iterator = tqdm(dataloader, desc=desc, leave=False)
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if train_mode:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == targets).sum().item()
        running_total += batch_size

    return running_loss / running_total, running_correct / running_total


def train(config: TrainingConfig) -> TrainingSummary:
    device = torch.device(config.device)

    set_seed(config.seed)

    data_cfg = DataConfig(
        data_dir=config.artifacts_dir / "data",
        batch_size=config.batch_size,
        val_split=config.val_split,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    train_loader, val_loader, test_loader = create_loaders(data_cfg)

    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5, verbose=True)

    best_val_acc = 0.0
    best_state: Dict[str, torch.Tensor] | None = None
    history: List[EpochLog] = []

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, desc="Train")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, desc="Val")
        scheduler.step(val_acc)

        history.append(
            EpochLog(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"New best validation accuracy: {val_acc:.4f}")

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    model_path = config.artifacts_dir / "mnist_cnn.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_path)

    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, desc="Test")
    print(f"Test accuracy: {test_acc:.4f}")

    summary = TrainingSummary(history, test_accuracy=test_acc, test_loss=test_loss)
    metrics_path = config.artifacts_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST CNN")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    base = TrainingConfig.from_json(args.config) if args.config else TrainingConfig()

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.device is not None:
        overrides["device"] = args.device
    if args.artifacts_dir is not None:
        overrides["artifacts_dir"] = args.artifacts_dir

    return TrainingConfig(
        epochs=overrides.get("epochs", base.epochs),
        batch_size=overrides.get("batch_size", base.batch_size),
        learning_rate=overrides.get("learning_rate", base.learning_rate),
        weight_decay=overrides.get("weight_decay", base.weight_decay),
        val_split=base.val_split,
        num_workers=base.num_workers,
        seed=base.seed,
        device=overrides.get("device", base.device),
        artifacts_dir=Path(overrides.get("artifacts_dir", base.artifacts_dir)),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    summary = train(config)
    print(json.dumps(summary.as_dict(), indent=2))


if __name__ == "__main__":
    main()
