let model;

const inputImages = [];
const inputMasks = [];

function updateCount(isMask) {
  const el = document.getElementById(isMask ? "maskCount" : "imageCount");
  el.textContent = isMask
    ? `${inputMasks.length} Mask images loaded`
    : `${inputImages.length} RGB images loaded`;
}

function diceLoss(yTrue, yPred) {
  const smooth = 1e-6;
  const intersection = tf.sum(tf.mul(yTrue, yPred));
  const sum = tf.sum(yTrue).add(tf.sum(yPred));
  return tf.sub(1, tf.div(tf.mul(2, intersection).add(smooth), sum.add(smooth)));
}

function buildUNetModel() {
  const inputs = tf.input({shape: [256, 256, 3]});

  const c1 = tf.layers.conv2d({filters: 8, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(inputs);
  const p1 = tf.layers.maxPooling2d({poolSize: 2}).apply(c1);

  const c2 = tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(p1);
  const p2 = tf.layers.maxPooling2d({poolSize: 2}).apply(c2);

  const b = tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(p2);

  const u1 = tf.layers.conv2dTranspose({filters: 16, kernelSize: 2, strides: 2, padding: 'same'}).apply(b);
  const concat1 = tf.layers.concatenate().apply([c2, u1]);
  const d1 = tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(concat1);

  const u2 = tf.layers.conv2dTranspose({filters: 8, kernelSize: 2, strides: 2, padding: 'same'}).apply(d1);
  const concat2 = tf.layers.concatenate().apply([c1, u2]);
  const d2 = tf.layers.conv2d({filters: 8, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(concat2);

  const outputs = tf.layers.conv2d({filters: 1, kernelSize: 1, activation: 'sigmoid'}).apply(d2);

  return tf.model({inputs: inputs, outputs: outputs});
}

async function preprocessImage(file) {
  const img = await createImageBitmap(file);
  const tensor = tf.browser.fromPixels(img)
    .resizeNearestNeighbor([256, 256])
    .toFloat()
    .div(255.0);
  return tensor;
}

async function handleFolder(files, isMask) {
  const targets = isMask ? inputMasks : inputImages;
  targets.length = 0;

  const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name));

  for (const file of sorted) {
    try {
      const tensor = await preprocessImage(file);
      const processed = isMask ? tensor.mean(2).expandDims(2) : tensor;

      targets.push(processed);
      console.log(`[${isMask ? "Mask" : "Image"}] ${file.name} loaded`);
    } catch (err) {
      console.error(`Error loading ${file.name}:`, err);
    }
  }

  updateCount(isMask);
}

function setupDropZone(id, inputId, isMask) {
  const drop = document.getElementById(id);
  const input = document.getElementById(inputId);

  drop.addEventListener("click", (e) => {
    e.stopPropagation();
    input.click();
  });

  input.addEventListener("change", async e => {
    const files = e.target.files;
    await handleFolder(files, isMask);
    drop.classList.add("loaded");
  });

  drop.addEventListener("dragover", e => {
    e.preventDefault();
    drop.classList.add("dragging");
  });

  drop.addEventListener("dragleave", () => {
    drop.classList.remove("dragging");
  });

  drop.addEventListener("drop", async e => {
    e.preventDefault();
    drop.classList.remove("dragging");

    const items = [...e.dataTransfer.items];
    const files = [];

    for (const item of items) {
      const entry = item.webkitGetAsEntry?.();
      if (entry && entry.isDirectory) {
        await readDirectory(entry, files);
      } else if (item.kind === "file") {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }

    await handleFolder(files, isMask);
    drop.classList.add("loaded");
  });
}

async function readDirectory(entry, fileList) {
  const reader = entry.createReader();
  return new Promise((resolve) => {
    const read = () => {
      reader.readEntries(async entries => {
        if (!entries.length) return resolve();
        for (const e of entries) {
          if (e.isFile) {
            e.file(file => fileList.push(file));
          } else if (e.isDirectory) {
            await readDirectory(e, fileList);
          }
        }
        read();
      });
    };
    read();
  });
}
const history = {
  loss: [],
  acc: [],
  labels: []
};

document.getElementById('trainBtn').addEventListener('click', async () => {
  if (inputImages.length === 0 || inputMasks.length === 0) {
    alert("Upload both image and mask folders.");
    return;
  }

  model = buildUNetModel();
  model.compile({
    optimizer: tf.train.adam(),
    loss: diceLoss,
    metrics: ['accuracy']
  });

  const x = tf.stack(inputImages);
  const y = tf.stack(inputMasks);

  const epochs = 10;
  document.getElementById('trainBar').value = 0;
  document.getElementById('trainBar').max = epochs;
  document.getElementById('trainProgress').innerText = '';
  document.getElementById('trainStatus').innerText = '';

  await model.fit(x, y, {
    epochs,
    batchSize: 4,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        if (batch % 5 === 0) {
          document.getElementById('trainProgress').innerText =
            `Batch ${batch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${(logs.acc ?? logs.accuracy ?? 0).toFixed(4)}`;
          await new Promise(r => setTimeout(r, 0));
        }
      },
      onEpochEnd: async (epoch, logs) => {
        history.loss.push(logs.loss);
        history.acc.push(logs.acc ?? logs.accuracy ?? 0);
        history.labels.push(`Epoch ${epoch + 1}`);
        updateChart();

        document.getElementById('trainBar').value = epoch + 1;
        document.getElementById('trainProgress').innerText =
          `Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${(logs.acc ?? logs.accuracy ?? 0).toFixed(4)}`;
        await new Promise(r => setTimeout(r, 0));
      }
    }
  });

  await model.save('indexeddb://unet-model');
  await model.save('downloads://unet-model');
  document.getElementById('trainStatus').innerText = "✅ Training Complete and Model Saved!";
});

document.getElementById('predictBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('predictImage');
  const file = fileInput.files[0];
  if (!file || !model) {
    alert("Upload one RGB image for prediction.");
    return;
  }

  const tensor = await preprocessImage(file);
  const input = tensor.expandDims(0);
  const prediction = model.predict(input).squeeze();

  const threshold = parseFloat(document.getElementById('thresholdSlider')?.value || '0.5');
  const mask = prediction.greater(threshold).toFloat();
  const rgba = tf.stack([mask, mask, mask, tf.onesLike(mask)], 2);
  await tf.browser.toPixels(rgba, document.getElementById('resultCanvas'));
});

document.getElementById("downloadModelBtn")?.addEventListener("click", async () => {
  if (!model) {
    alert("Train a model before downloading.");
    return;
  }
  await model.save('downloads://unet-model');
  alert("✅ Model downloaded to your computer.");
});

document.getElementById("uploadModelInput")?.addEventListener("change", async (e) => {
  const files = e.target.files;
  const jsonFile = Array.from(files).find(f => f.name.endsWith(".json"));
  if (!jsonFile) return alert("Please upload the .json file of your model.");
  try {
    model = await tf.loadLayersModel(tf.io.browserFiles([...files]));
    model.compile({
      optimizer: tf.train.adam(),
      loss: diceLoss,
      metrics: ['accuracy']
    });
    alert("✅ Model uploaded and loaded successfully.");
  } catch (err) {
    console.error("Error loading model:", err);
    alert("❌ Failed to load model from uploaded files.");
  }
});

window.onload = async () => {
  const slider = document.getElementById("thresholdSlider");
  const label = document.getElementById("thresholdValue");
  if (slider && label) {
    slider.addEventListener("input", () => {
      label.textContent = slider.value;
    });
  }
  document.getElementById("clearModelBtn")?.addEventListener("click", async () => {
    await tf.io.removeModel('indexeddb://unet-model');
    alert("🧹 Model cleared from browser storage.");
  });
  try {
    const models = await tf.io.listModels();
    if ('indexeddb://unet-model' in models) {
      model = await tf.loadLayersModel('indexeddb://unet-model');
      model.compile({
        optimizer: tf.train.adam(),
        loss: diceLoss,
        metrics: ['accuracy']
      });
      console.log("✅ Loaded saved model from browser storage.");
    }
  } catch (e) {
    console.warn("⚠️ Failed to load model from IndexedDB:", e);
  }
  setupDropZone("imageDrop", "inputImage", false);
  setupDropZone("maskDrop", "inputMask", true);
  updateCount(false);
  updateCount(true);

  document.getElementById("predictDrop").addEventListener("click", () => {
    document.getElementById("predictImage").click();
  });

  document.getElementById("predictImage").addEventListener("change", () => {
    const fileInput = document.getElementById("predictImage");
    const el = document.getElementById("predictCount");
    const n = fileInput.files.length;
    el.textContent = n > 0 ? `${n} image${n > 1 ? 's' : ''} uploaded` : "No image uploaded";
  });
};

let chart;
function updateChart() {
  const ctx = document.getElementById('trainingChart').getContext('2d');
  if (!chart) {
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: history.labels,
        datasets: [
          {
            label: 'Loss',
            borderColor: 'red',
            data: history.loss,
            fill: false
          },
          {
            label: 'Accuracy',
            borderColor: 'green',
            data: history.acc,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
  } else {
    chart.data.labels = history.labels;
    chart.data.datasets[0].data = history.loss;
    chart.data.datasets[1].data = history.acc;
    chart.update();
  }
}
