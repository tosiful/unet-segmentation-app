let model;

const inputImages = [];
const inputMasks = [];
let rgbFiles = [];
let maskFiles = [];

function showToast(msg, type = "success", ms = 2200) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.className = type;
  toast.style.display = "block";
  setTimeout(() => { toast.style.display = "none"; }, ms);
}

function updateCount(isMask) {
  const el = document.getElementById(isMask ? "maskCount" : "imageCount");
  el.textContent = isMask
    ? `${maskFiles.length} Mask images loaded`
    : `${rgbFiles.length} RGB images loaded`;
}

function updatePreview(isMask, files) {
  const preview = document.getElementById(isMask ? "maskPreview" : "imagePreview");
  preview.innerHTML = "";
  [...files].slice(0, 6).forEach(file => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    preview.appendChild(img);
  });
  if (files.length > 6) {
    const more = document.createElement("span");
    more.textContent = `+${files.length - 6}`;
    more.style.marginLeft = "6px";
    more.style.fontWeight = "bold";
    preview.appendChild(more);
  }
}

async function preprocessImage(file) {
  const img = await createImageBitmap(file);
  const tensor = tf.browser.fromPixels(img)
    .resizeNearestNeighbor([256, 256])
    .toFloat()
    .div(255.0);
  return tensor;
}

function setupDropZone(id, inputId, isMask) {
  const drop = document.getElementById(id);
  const input = document.getElementById(inputId);

  drop.addEventListener("click", (e) => {
    e.stopPropagation();
    input.click();
  });

  input.addEventListener("change", async e => {
    const files = Array.from(e.target.files);
    if (isMask) {
      maskFiles = files;
    } else {
      rgbFiles = files;
    }
    updateCount(isMask);
    updatePreview(isMask, files);
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
    const files = Array.from(e.dataTransfer.files);
    if (isMask) {
      maskFiles = files;
    } else {
      rgbFiles = files;
    }
    updateCount(isMask);
    updatePreview(isMask, files);
    drop.classList.add("loaded");
  });
}

const history = {
  loss: [],
  acc: [],
  labels: []
};

function diceLoss(yTrue, yPred) {
  const smooth = 1e-6;
  const intersection = tf.sum(tf.mul(yTrue, yPred));
  const sum = tf.sum(yTrue).add(tf.sum(yPred));
  return tf.sub(1, tf.div(tf.mul(2, intersection).add(smooth), sum.add(smooth)));
}

// Shallow U-Net (default)
function buildShallowUNetModel() {
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

// Deep U-Net (BatchNorm, more layers)
function convBlock(input, filters) {
  let x = tf.layers.conv2d({
    filters: filters,
    kernelSize: 3,
    padding: 'same'
  }).apply(input);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.activation('relu').apply(x);
  return x;
}

function upConvBlock(input, skipFeatures, filters) {
  let x = tf.layers.upSampling2d({ size: [2, 2] }).apply(input);
  x = tf.layers.conv2d({
    filters: filters,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }).apply(x);
  x = tf.layers.concatenate().apply([x, skipFeatures]);
  return convBlock(x, filters);
}

function buildDeepUNetModel() {
  const inputs = tf.input({ shape: [256, 256, 3] });
  let s1 = convBlock(inputs, 8);
  let p1 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(s1);
  let s2 = convBlock(p1, 16);
  let p2 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(s2);
  let s3 = convBlock(p2, 32);
  let p3 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(s3);
  let s4 = convBlock(p3, 64);
  let p4 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(s4);
  let bridge = convBlock(p4, 64);
  let u1 = upConvBlock(bridge, s4, 32);
  let u2 = upConvBlock(u1, s3, 16);
  let u3 = upConvBlock(u2, s2, 8);
  let u4 = upConvBlock(u3, s1, 4);
  const outputs = tf.layers.conv2d({
    filters: 1,
    kernelSize: 1,
    activation: 'sigmoid'
  }).apply(u4);
  return tf.model({ inputs, outputs });
}

document.getElementById('trainBtn').addEventListener('click', async () => {
  if (rgbFiles.length === 0 || maskFiles.length === 0) {
    showToast("Upload both RGB and mask images.", "error");
    return;
  }
  if (rgbFiles.length !== maskFiles.length) {
    showToast("Number of RGB and mask images must match.", "error");
    return;
  }

  inputImages.length = 0;
  inputMasks.length = 0;

  for (let i = 0; i < rgbFiles.length; ++i) {
    const rgbTensor = await preprocessImage(rgbFiles[i]);
    const maskTensor = (await preprocessImage(maskFiles[i])).mean(2).expandDims(2);
    inputImages.push(rgbTensor);
    inputMasks.push(maskTensor);
  }

  // Model selection logic
  const modelType = document.getElementById('modelSelect').value;
  if (modelType === 'deep') {
    model = buildDeepUNetModel();
  } else {
    model = buildShallowUNetModel();
  }

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
  showToast("Training complete! Model saved.", "success");
});

document.getElementById('predictBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('predictImage');
  const file = fileInput.files[0];
  if (!file || !model) {
    showToast("Upload one RGB image for prediction.", "error");
    return;
  }

  const tensor = await preprocessImage(file);
  const input = tensor.expandDims(0);
  const prediction = model.predict(input).squeeze();

  const threshold = parseFloat(document.getElementById('thresholdSlider')?.value || '0.5');
  const mask = prediction.greater(threshold).toFloat();
  const rgba = tf.stack([mask, mask, mask, tf.onesLike(mask)], 2);
  await tf.browser.toPixels(rgba, document.getElementById('resultCanvas'));
  showToast("Prediction complete!", "success");
});

document.getElementById("downloadModelBtn")?.addEventListener("click", async () => {
  if (!model) {
    showToast("Train a model before downloading.", "error");
    return;
  }
  await model.save('downloads://unet-model');
  showToast("Model downloaded.", "success");
});

document.getElementById("uploadModelInput")?.addEventListener("change", async (e) => {
  const files = e.target.files;
  try {
    model = await tf.loadLayersModel(tf.io.browserFiles([...files]));
    model.compile({
      optimizer: tf.train.adam(),
      loss: diceLoss,
      metrics: ['accuracy']
    });
    showToast("✅ Model uploaded and loaded successfully.", "success");
  } catch (err) {
    console.error("Error loading model:", err);
    showToast("❌ Failed to load model from uploaded files.", "error");
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
    showToast("Model cleared from browser storage.", "success");
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
      showToast("Loaded saved model from browser storage.", "success");
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
    // Show preview
    const preview = document.getElementById("predictPreview");
    preview.innerHTML = "";
    if (n > 0) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(fileInput.files[0]);
      preview.appendChild(img);
    }
  });

  // Predict on Mask UI logic
  document.getElementById("maskPredictDrop").addEventListener("click", () => {
    document.getElementById("maskPredictImage").click();
  });

  document.getElementById("maskPredictImage").addEventListener("change", () => {
    const fileInput = document.getElementById("maskPredictImage");
    const el = document.getElementById("maskPredictCount");
    const n = fileInput.files.length;
    el.textContent = n > 0 ? `${n} mask${n > 1 ? 's' : ''} uploaded` : "No mask uploaded";
    // Show preview
    const preview = document.getElementById("maskPredictPreview");
    preview.innerHTML = "";
    if (n > 0) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(fileInput.files[0]);
      preview.appendChild(img);
    }
  });

  document.getElementById("showMaskBtn").addEventListener("click", async () => {
    const fileInput = document.getElementById("maskPredictImage");
    const file = fileInput.files[0];
    if (!file) {
      showToast("Upload a mask image to display.", "error");
      return;
    }
    // Display the mask as grayscale
    const img = await createImageBitmap(file);
    const canvas = document.getElementById("maskResultCanvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    showToast("Mask displayed.", "success");
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