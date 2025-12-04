# MNIST Digit Recognition

This project trains a convolutional neural network (CNN) on the MNIST handwritten digit dataset. The goal is to preprocess the images, train a classifier for digits 0–9, and evaluate its accuracy.

## Repo Layout

```
src/
├── data/         # Dataset utilities (downloads & preprocessing)
├── models/       # Model definitions
└── training/     # Training scripts / entrypoints
artifacts/        # Created after training (data cache, weights, metrics)
```

## Quickstart

1. Install dependencies (Python 3.11 recommended):

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Train the model (downloads MNIST automatically, saves artifacts under `artifacts/`):

   ```powershell
   python -m src.training.train
   ```

3. Inspect results:
   - `artifacts/mnist_cnn.pt`: best validation checkpoint (PyTorch state dict).
   - `artifacts/metrics.json`: training/validation history and final test metrics.

The current configuration (6 epochs, Adam optimizer, dropout regularization) achieves ≈99.2 % accuracy on the MNIST test set.

## Predict on your own digit

1. Prepare your image:

   - 28×28 pixels works best; non-square images are resized automatically.
   - Use a grayscale PNG/JPG. If your digit is dark on a white background, pass `--invert`.

2. Run the inference helper:

   ```powershell
   python -m src.inference.predict path\to\digit.png --show-probabilities
   ```

   Add `--model-path` if you saved the checkpoint elsewhere. The script loads `artifacts/mnist_cnn.pt` by default and prints the predicted digit (optionally class probabilities).

## Optional web interface

Spin up a lightweight UI for drag-and-drop uploads:

```powershell
python -m app.main
```

Then open http://127.0.0.1:5000/ in your browser. The backend preprocesses the image to 28×28 grayscale, runs the trained model, and displays both the predicted digit and the per-class probabilities. Use the “Invert colors” toggle if your digit is drawn dark-on-light.
