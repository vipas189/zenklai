// Tab switching functionality
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    // Remove active class from all tabs and tab contents
    document
      .querySelectorAll(".tab")
      .forEach((t) => t.classList.remove("active"));
    document
      .querySelectorAll(".tab-content")
      .forEach((content) => content.classList.remove("active"));

    // Add active class to clicked tab and corresponding content
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
  });
});

// Model selection change handler
document.getElementById("model-type").addEventListener("change", function () {
  // Hide all settings sections
  document
    .querySelectorAll('.settings-section[id$="-settings"]')
    .forEach((section) => {
      section.style.display = "none";
    });

  // Show the selected model's settings
  const selectedModel = this.value;
  if (selectedModel === "cnn") {
    document.getElementById("cnn-settings").style.display = "block";
  } else if (selectedModel === "rnn") {
    document.getElementById("rnn-settings").style.display = "block";
  } else if (selectedModel === "svm") {
    document.getElementById("svm-settings").style.display = "block";
  } else if (selectedModel === "random_forest") {
    document.getElementById("rf-settings").style.display = "block";
  } else if (selectedModel === "knn") {
    document.getElementById("knn-settings").style.display = "block";
  }
});

// Start training button handler
document
  .getElementById("start-training-btn")
  .addEventListener("click", function () {
    const trainingConsole = document.getElementById("training-console");
    const loader = document.getElementById("training-loader");

    // Check if files are uploaded
    const trainImages = document.getElementById("train-images").files;
    const trainCsv = document.getElementById("train-csv").files;

    if (trainImages.length === 0 || trainCsv.length === 0) {
      trainingConsole.innerHTML +=
        '<p class="error">Error: Please upload both training images (ZIP) and CSV file</p>';
      return;
    }

    // Show loader
    loader.style.display = "block";

    // Get selected model and its hyperparameters
    const selectedModel = document.getElementById("model-type").value;
    let hyperparams = {};

    if (selectedModel === "cnn") {
      hyperparams = {
        epochs: document.getElementById("cnn-epochs").value,
        batchSize: document.getElementById("cnn-batch-size").value,
        learningRate: document.getElementById("cnn-learning-rate").value,
        optimizer: document.getElementById("cnn-optimizer").value,
        dropout: document.getElementById("cnn-dropout").value,
        convLayers: document.getElementById("cnn-conv-layers").value,
      };
    } else if (selectedModel === "rnn") {
      hyperparams = {
        epochs: document.getElementById("rnn-epochs").value,
        batchSize: document.getElementById("rnn-batch-size").value,
        learningRate: document.getElementById("rnn-learning-rate").value,
        lstmCells: document.getElementById("rnn-lstm-cells").value,
        layers: document.getElementById("rnn-layers").value,
      };
    } else if (selectedModel === "svm") {
      hyperparams = {
        C: document.getElementById("svm-c").value,
        kernel: document.getElementById("svm-kernel").value,
        gamma: document.getElementById("svm-gamma").value,
      };
    } else if (selectedModel === "random_forest") {
      hyperparams = {
        nEstimators: document.getElementById("rf-n-estimators").value,
        maxDepth: document.getElementById("rf-max-depth").value,
        minSamplesSplit: document.getElementById("rf-min-samples-split").value,
      };
    } else if (selectedModel === "knn") {
      hyperparams = {
        nNeighbors: document.getElementById("knn-n-neighbors").value,
        weights: document.getElementById("knn-weights").value,
        algorithm: document.getElementById("knn-algorithm").value,
      };
    }

    // Clear console and add initial messages
    trainingConsole.innerHTML = "<p>== Training Console ==</p>";
    trainingConsole.innerHTML +=
      '<p class="info">Starting training process...</p>';
    trainingConsole.innerHTML += `<p>Selected model: ${selectedModel.toUpperCase()}</p>`;
    trainingConsole.innerHTML += "<p>Hyperparameters:</p>";
    for (const [key, value] of Object.entries(hyperparams)) {
      trainingConsole.innerHTML += `<p>- ${key}: ${value}</p>`;
    }

    // Simulate training process
    simulateTraining(selectedModel, hyperparams);
  });

// Simulate training process with console output
function simulateTraining(model, hyperparams) {
  const trainingConsole = document.getElementById("training-console");
  const loader = document.getElementById("training-loader");

  // Simulate loading data
  setTimeout(() => {
    trainingConsole.innerHTML +=
      '<p class="info">Loading and preprocessing training data...</p>';
    trainingConsole.scrollTop = trainingConsole.scrollHeight;

    setTimeout(() => {
      trainingConsole.innerHTML +=
        '<p class="success">Training data loaded successfully (39,209 images)</p>';
      trainingConsole.innerHTML +=
        "<p>Extracting features and preparing batches...</p>";
      trainingConsole.scrollTop = trainingConsole.scrollHeight;

      setTimeout(() => {
        trainingConsole.innerHTML +=
          '<p class="success">Data preparation complete</p>';
        trainingConsole.innerHTML +=
          '<p class="info">Starting model training...</p>';
        trainingConsole.scrollTop = trainingConsole.scrollHeight;

        // Simulate epochs
        let currentEpoch = 1;
        const totalEpochs =
          model === "cnn"
            ? parseInt(hyperparams.epochs)
            : model === "rnn"
            ? parseInt(hyperparams.epochs)
            : 5;

        const epochInterval = setInterval(() => {
          if (currentEpoch <= totalEpochs) {
            const accuracy = (70 + (currentEpoch / totalEpochs) * 25).toFixed(
              2
            );
            const loss = (1 - (currentEpoch / totalEpochs) * 0.8).toFixed(4);

            trainingConsole.innerHTML += `<p>Epoch ${currentEpoch}/${totalEpochs} - accuracy: ${accuracy}% - loss: ${loss}</p>`;
            trainingConsole.scrollTop = trainingConsole.scrollHeight;
            currentEpoch++;
          } else {
            clearInterval(epochInterval);

            // Training completed
            trainingConsole.innerHTML +=
              '<p class="success">Training completed!</p>';
            trainingConsole.innerHTML +=
              "<p>Evaluating model on validation set...</p>";
            trainingConsole.scrollTop = trainingConsole.scrollHeight;

            setTimeout(() => {
              const finalAccuracy = (92 + Math.random() * 5).toFixed(2);
              const finalLoss = (0.1 + Math.random() * 0.05).toFixed(4);
              const precision = (91 + Math.random() * 5).toFixed(2);
              const recall = (91 + Math.random() * 5).toFixed(2);
              const f1 = (91 + Math.random() * 5).toFixed(2);

              trainingConsole.innerHTML +=
                '<p class="info">Final evaluation metrics:</p>';
              trainingConsole.innerHTML += `<p>- Accuracy: ${finalAccuracy}%</p>`;
              trainingConsole.innerHTML += `<p>- Loss: ${finalLoss}</p>`;
              trainingConsole.innerHTML += `<p>- Precision: ${precision}%</p>`;
              trainingConsole.innerHTML += `<p>- Recall: ${recall}%</p>`;
              trainingConsole.innerHTML += `<p>- F1 Score: ${f1}%</p>`;
              trainingConsole.innerHTML +=
                '<p class="success">Model saved successfully!</p>';
              trainingConsole.scrollTop = trainingConsole.scrollHeight;

              // Hide loader
              loader.style.display = "none";
            }, 2000);
          }
        }, 1000);
      }, 1500);
    }, 2000);
  }, 1000);
}

// Test image preview
document
  .getElementById("test-image")
  .addEventListener("change", function (event) {
    const preview = document.getElementById("test-image-preview");
    const file = event.target.files[0];

    if (file) {
      preview.style.display = "block";
      preview.src = URL.createObjectURL(file);
    } else {
      preview.style.display = "none";
    }
  });

// Run test button handler
document.getElementById("run-test-btn").addEventListener("click", function () {
  const testImage = document.getElementById("test-image").files;
  const testResult = document.getElementById("test-result");
  const loader = document.getElementById("test-loader");

  if (testImage.length === 0) {
    alert("Please upload an image to test");
    return;
  }

  // Show loader and hide result
  loader.style.display = "block";
  testResult.classList.remove("active");

  // Simulate processing
  setTimeout(() => {
    // Hide loader and show result
    loader.style.display = "none";
    testResult.classList.add("active");

    // Generate random results for demo
    const trafficSigns = [
      "Speed Limit (30km/h)",
      "Speed Limit (50km/h)",
      "Stop",
      "Yield",
      "No Entry",
      "Priority Road",
      "Keep Right",
    ];

    const randomSign =
      trafficSigns[Math.floor(Math.random() * trafficSigns.length)];
    const randomConfidence = (92 + Math.random() * 7).toFixed(1);

    document.getElementById("predicted-label").textContent = randomSign;
    document.getElementById(
      "confidence-score"
    ).textContent = `Confidence: ${randomConfidence}%`;
  }, 2000);
});
