"""Flask app exposing a simple MNIST prediction interface."""
from __future__ import annotations

import base64
import io
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List

from flask import Flask, flash, jsonify, render_template, request
from PIL import Image
import torch

from src.inference.predict import load_model, predict_digit, preprocess_pil_image


@dataclass
class PredictionResult:
    prediction: int
    probabilities: List[tuple[int, float]]


def _analyze_image(
    image_bytes: bytes,
    invert: bool,
    model: torch.nn.Module,
    device: torch.device,
    target_size: tuple[int, int],
) -> dict:
    if not image_bytes:
        raise ValueError("Empty file uploaded")

    image = Image.open(io.BytesIO(image_bytes))
    preview_size = image.size

    tensor = preprocess_pil_image(image, invert=invert, size=target_size)
    prediction, probs_tensor = predict_digit(model, tensor, device)
    probabilities = [(idx, float(prob)) for idx, prob in enumerate(probs_tensor.tolist())]
    result = PredictionResult(prediction=prediction, probabilities=probabilities)

    buffered = io.BytesIO()
    image.copy().resize((180, 180)).save(buffered, format="PNG")
    preview_src = base64.b64encode(buffered.getvalue()).decode("utf-8")
    stored_image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "result": result,
        "preview_src": preview_src,
        "preview_size": preview_size,
        "stored_image_b64": stored_image_b64,
    }


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY="mnist-secret-key",
        MODEL_PATH=Path("artifacts/mnist_cnn.pt"),
        DEVICE="cuda" if torch.cuda.is_available() else "cpu",
        TARGET_SIZE=(28, 28),
    )

    if config:
        app.config.update(config)

    device = torch.device(app.config["DEVICE"])
    model = load_model(Path(app.config["MODEL_PATH"]), device)

    try:
        with torch.no_grad():
            height, width = app.config["TARGET_SIZE"]
            dummy_input = torch.zeros((1, 1, height, width), device=device)
            model(dummy_input)
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Model warm-up failed: %s", exc)

    @app.route("/", methods=["GET", "POST"])
    def index():
        invert = bool(request.form.get("invert"))
        result = None
        preview_src = None
        preview_size = app.config["TARGET_SIZE"]
        stored_image_b64 = request.form.get("existing_image")
        current_year = datetime.now().year

        if request.method == "POST":
            file = request.files.get("image")
            image_bytes = b""

            if file and file.filename:
                image_bytes = file.read()
            elif stored_image_b64:
                try:
                    image_bytes = base64.b64decode(stored_image_b64)
                except Exception as exc:  # noqa: BLE001
                    flash(f"Failed to decode existing image: {exc}")
            else:
                flash("Please choose an image file to upload.")
                return render_template(
                    "index.html",
                    result=None,
                    preview_src=None,
                    invert=invert,
                    preview_size=preview_size,
                    existing_image=stored_image_b64,
                    current_year=current_year,
                )

            try:
                analysis = _analyze_image(
                    image_bytes=image_bytes,
                    invert=invert,
                    model=model,
                    device=device,
                    target_size=app.config["TARGET_SIZE"],
                )
                result = analysis["result"]
                preview_src = analysis["preview_src"]
                preview_size = analysis["preview_size"]
                stored_image_b64 = analysis["stored_image_b64"]
            except Exception as exc:  # noqa: BLE001
                flash(f"Failed to process image: {exc}")

        return render_template(
            "index.html",
            result=result,
            preview_src=preview_src,
            invert=invert,
            preview_size=preview_size,
            existing_image=stored_image_b64,
            current_year=current_year,
        )

    @app.post("/predict")
    def predict_api():
        invert_value = request.form.get("invert", "")
        invert = invert_value.lower() in {"1", "true", "on", "yes"}
        stored_image_b64 = request.form.get("existing_image")
        file = request.files.get("image")
        image_bytes = b""

        if file and file.filename:
            image_bytes = file.read()
        elif stored_image_b64:
            try:
                image_bytes = base64.b64decode(stored_image_b64)
            except Exception as exc:  # noqa: BLE001
                return jsonify({"error": f"Failed to decode existing image: {exc}"}), 400
        else:
            return jsonify({"error": "Please choose an image file to upload."}), 400

        try:
            analysis = _analyze_image(
                image_bytes=image_bytes,
                invert=invert,
                model=model,
                device=device,
                target_size=app.config["TARGET_SIZE"],
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Failed to process image: {exc}"}), 400

        result: PredictionResult = analysis["result"]
        probabilities_payload = [
            {"digit": digit, "probability": prob} for digit, prob in result.probabilities
        ]

        return jsonify(
            {
                "prediction": result.prediction,
                "probabilities": probabilities_payload,
                "preview": {
                    "dataUrl": f"data:image/png;base64,{analysis['preview_src']}",
                    "width": analysis["preview_size"][0],
                    "height": analysis["preview_size"][1],
                },
                "existingImage": analysis["stored_image_b64"],
                "invert": invert,
            }
        )

    return app


if __name__ == "__main__":
    # Allows running via `python -m app.main`
    app = create_app()
    app.run(debug=True, port=5000)
