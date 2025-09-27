"""CLI utility for running MNIST digit predictions on user images."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

os.environ.setdefault("TORCHVISION_DISABLE_ONNX", "1")

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.models.cnn import MNISTNet

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def build_transform(size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def preprocess_pil_image(image: Image.Image, invert: bool, size: Tuple[int, int]) -> torch.Tensor:
    if invert:
        image = ImageOps.invert(image.convert("L"))
    else:
        image = image.convert("L")

    transform = build_transform(size)
    tensor = transform(image)
    return tensor.unsqueeze(0)


def load_model(model_path: Path, device: torch.device) -> MNISTNet:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Run `python -m src.training.train` first."
        )
    state_dict = torch.load(model_path, map_location=device)
    model = MNISTNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path: Path, invert: bool, size: Tuple[int, int]) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found at {image_path}")

    with Image.open(image_path) as image:
        return preprocess_pil_image(image, invert=invert, size=size)


def predict_digit(model: MNISTNet, tensor: torch.Tensor, device: torch.device) -> Tuple[int, torch.Tensor]:
    with torch.no_grad():
        logits = model(tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)
        prediction = int(probabilities.argmax(dim=1).item())
    return prediction, probabilities.squeeze(0).cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MNIST digit from an image")
    parser.add_argument("image", type=Path, help="Path to the digit image (PNG/JPG)")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/mnist_cnn.pt"),
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert image colors (use if digit is dark on white background)",
    )
    parser.add_argument(
        "--show-probabilities",
        action="store_true",
        help="Print class probabilities alongside the predicted digit",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(28, 28),
        metavar=("HEIGHT", "WIDTH"),
        help="Resize dimensions applied before inference (default: 28 28)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    tensor = preprocess_image(args.image, invert=args.invert, size=tuple(args.size))
    model = load_model(args.model_path, device)
    digit, probabilities = predict_digit(model, tensor, device)

    print(f"Predicted digit: {digit}")
    if args.show_probabilities:
        for idx, prob in enumerate(probabilities.tolist()):
            print(f"  {idx}: {prob:.4f}")


if __name__ == "__main__":
    main()
