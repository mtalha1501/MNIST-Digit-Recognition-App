document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("analyze-form");
  if (!form) {
    return;
  }

  const analyzeButton = document.getElementById("analyze-button");
  const existingImageField = document.getElementById("existing-image-field");
  const invertCheckbox = document.getElementById("invert");
  const previewCard = document.getElementById("preview-card");
  const previewPlaceholder = document.getElementById("preview-placeholder");
  const previewImage = document.getElementById("preview-image");
  const previewSize = document.getElementById("preview-size");
  const previewInvert = document.getElementById("preview-invert");
  const resultCard = document.getElementById("result-card");
  const resultDigit = document.getElementById("result-digit");
  const probabilitiesList = document.getElementById("probabilities-list");
  const errorBox = document.getElementById("prediction-error");
  const fileInput = document.getElementById("image");

  const setButtonState = (isLoading) => {
    if (!analyzeButton) return;
    analyzeButton.disabled = isLoading;
    analyzeButton.textContent = isLoading ? "Analyzing…" : "Analyze Digit";
  };

  const showError = (message) => {
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  };

  const clearError = () => {
    if (!errorBox) return;
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  };

  const updatePreview = (payload) => {
    if (!payload.preview) return;
    const { dataUrl, width, height } = payload.preview;
    if (previewCard) {
      previewCard.classList.remove("hidden");
    }
    if (previewPlaceholder) {
      previewPlaceholder.classList.add("hidden");
    }
    if (previewImage && dataUrl) {
      previewImage.src = dataUrl;
    }
    if (previewSize) {
      const widthText = width ? width : "?";
      const heightText = height ? height : "?";
      previewSize.textContent = `${widthText} × ${heightText}`;
    }
    if (previewInvert) {
      previewInvert.textContent = payload.invert ? "Yes" : "No";
    }
  };

  const updateResult = (payload) => {
    if (payload.prediction === undefined) return;
    if (resultCard) {
      resultCard.classList.remove("hidden");
    }
    if (resultDigit) {
      resultDigit.textContent = payload.prediction;
    }
    if (probabilitiesList && Array.isArray(payload.probabilities)) {
      probabilitiesList.innerHTML = "";
      payload.probabilities.forEach((item) => {
        const span = document.createElement("span");
        span.dataset.digit = String(item.digit);
        const probability = Number(item.probability);
        span.textContent = `${item.digit} ▸ ${probability.toFixed(4)}`;
        probabilitiesList.appendChild(span);
      });
    }
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!analyzeButton) {
      return;
    }

    clearError();
    setButtonState(true);

    const formData = new FormData(form);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      let payload;
      try {
        payload = await response.json();
      } catch (jsonError) {
        throw new Error("Prediction failed: invalid server response.");
      }

      if (!response.ok) {
        throw new Error(payload.error || "Prediction failed.");
      }

      if (existingImageField && payload.existingImage) {
        existingImageField.value = payload.existingImage;
      }

      if (invertCheckbox) {
        invertCheckbox.checked = Boolean(payload.invert);
      }

      updatePreview(payload);
      updateResult(payload);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Prediction failed.");
    } finally {
      setButtonState(false);
    }
  });

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      if (existingImageField) {
        existingImageField.value = "";
      }
    });
  }
});
