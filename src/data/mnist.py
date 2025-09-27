"""MNIST dataset utilities."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

os.environ.setdefault("TORCHVISION_DISABLE_ONNX", "1")

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    data_dir: Path
    batch_size: int = 128
    val_split: float = 0.1
    num_workers: int = 2
    seed: int = 42


def _transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def create_loaders(config: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    transform = _transforms()

    train_dataset = datasets.MNIST(
        root=str(config.data_dir),
        train=True,
        transform=transform,
        download=True,
    )

    val_size = int(len(train_dataset) * config.val_split)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    test_dataset = datasets.MNIST(
        root=str(config.data_dir),
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
