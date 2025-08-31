/* app.js — TF.js U-Net trainer, image & video predictor (Dice metric)
   - Uses Dice loss + Dice metric (no accuracy)
   - Chart shows Loss, Dice, Val Loss, Val Dice (safe if Chart.js missing)
   - Image: auto-pair *_seg masks, train, predict → binary mask (white on black)
   - Video: per-frame segmentation (overlay or mask) + recording (MediaRecorder)
   - WebGL-safe: no tf.stack/concat in rendering; reuse canvases; aggressive cleanup
*/

"use strict";

/* ---------------- TFJS setup (paths + conservative WebGL + backend) ---------------- */
if (typeof tf !== "undefined") {
  // Point TFJS to local WASM binaries if stored them in ./libs/
  if (tf?.wasm?.setWasmPaths) tf.wasm.setWasmPaths("./libs/");

  // Free GPU textures aggressively (prevents WebGL context loss on some GPUs)
  if (tf?.env) tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 0);

  (async () => {
    const order = ["webgl", "wasm", "cpu"];
    for (const b of order) {
      try {
        await tf.setBackend(b);
        await tf.ready();
        console.log("TFJS backend:", tf.getBackend());
        break;
      } catch (e) {
        console.warn("Backend init failed:", b, e);
      }
    }
  })();
} else {
  console.error("TensorFlow.js not loaded. Ensure tf.min.js is included before app.js.");
}

/* -------------------------------- Globals / UI helpers -------------------------------- */
let model;
let rgbFiles = [];
let maskFiles = [];
let lastFolderFiles = null;

function showToast(msg, type = "success", ms = 2200) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = msg;
  toast.className = type;
  toast.style.display = "block";
  setTimeout(() => (toast.style.display = "none"), ms);
}

function showTrainSpinner() {
  const s = document.getElementById("trainSpinner");
  if (s) s.classList.add("show");
}
function hideTrainSpinner() {
  const s = document.getElementById("trainSpinner");
  if (s) s.classList.remove("show");
}

function setTrainingUI(isOn, text = "Training…") {
  const btn = document.getElementById("trainBtn");
  const modelSel = document.getElementById("modelSelect");
  const epochsIn = document.getElementById("epochsInput");
  const batchIn  = document.getElementById("batchSizeInput");
  const copyBtn  = document.getElementById("copyFilesBtn");
  const folderIn = document.getElementById("sourceFolderInput");

  if (!btn) return;

  if (isOn) {
    if (!btn.dataset._orig) btn.dataset._orig = btn.textContent;

    btn.textContent = text;
    btn.disabled = true;
    modelSel && (modelSel.disabled = true);
    epochsIn && (epochsIn.disabled = true);
    batchIn  && (batchIn.disabled  = true);
    copyBtn  && (copyBtn.disabled  = true);
    folderIn && (folderIn.disabled = true);
  } else {
    btn.textContent = btn.dataset._orig || "Train U-Net";
    btn.disabled = false;
    modelSel && (modelSel.disabled = false);
    epochsIn && (epochsIn.disabled = false);
    batchIn  && (batchIn.disabled  = false);
    copyBtn  && (copyBtn.disabled  = false);
    folderIn && (folderIn.disabled = false);
  }
}
function setTrainButtonText(t) {
  const btn = document.getElementById("trainBtn");
  if (btn && btn.classList.contains("loading")) btn.textContent = t;
}

function showTrainOverlay(text = "Preparing training…") {
  const el = document.getElementById("trainerOverlay");
  if (!el) return;
  el.style.display = "flex";
  const msg = el.querySelector(".msg");
  if (msg) msg.textContent = text;
}
function hideTrainOverlay() {
  const el = document.getElementById("trainerOverlay");
  if (el) el.style.display = "none";
}

function updateCount(isMask) {
  const el = document.getElementById(isMask ? "maskCount" : "imageCount");
  if (!el) return;
  el.textContent = isMask
    ? `${maskFiles.length} Mask images loaded`
    : `${rgbFiles.length} RGB images loaded`;
}

function updatePreview(isMask, files) {
  const preview = document.getElementById(isMask ? "maskPreview" : "imagePreview");
  if (!preview) return;
  preview.innerHTML = "";
  Array.from(files).slice(0, 6).forEach((file) => {
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

/* ----------------------------- Robust image loading ----------------------------- */
function isSupportedImageFile(file) {
  const okExt = /\.(png|jpg|jpeg|webp|bmp)$/i.test(file.name || "");
  const okType = /^(image\/png|image\/jpeg|image\/webp|image\/bmp)$/i.test(file.type || "");
  return okExt || okType;
}

async function loadBitmapCrossBrowser(file) {
  if (window.createImageBitmap) {
    try { return await createImageBitmap(file); } catch {}
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      try {
        const c = document.createElement("canvas");
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        const ctx = c.getContext("2d");
        ctx.drawImage(img, 0, 0);
        resolve(c);
      } catch (err) { reject(err); }
      finally { URL.revokeObjectURL(img.src); }
    };
    img.onerror = () => reject(new Error("Failed to load image."));
    img.src = URL.createObjectURL(file);
  });
}

async function preprocessImageFile(file) {
  if (!isSupportedImageFile(file)) { showToast("Unsupported image file.", "error"); throw new Error("Unsupported image."); }
  const src = await loadBitmapCrossBrowser(file);
  return tf.tidy(() =>
    tf.image.resizeBilinear(tf.browser.fromPixels(src, 3).toFloat().div(255), [256, 256], true)
  );
}

async function preprocessMaskFile(file) {
  if (!isSupportedImageFile(file)) {
    showToast("Unsupported mask file.", "error");
    throw new Error("Unsupported mask.");
  }
  const src = await loadBitmapCrossBrowser(file);
  return tf.tidy(() => {
    const m = tf.browser.fromPixels(src, 1).toFloat().div(255);         // (H,W,1) in [0,1]
    const bin = m.greater(tf.scalar(0.5)).toFloat();                    // binarize hard
    return tf.image.resizeNearestNeighbor(bin, [256, 256], true);       // preserve labels
  });
}

async function preprocessImageBitmap(imgBitmap) {
  return tf.tidy(() =>
    tf.image.resizeBilinear(tf.browser.fromPixels(imgBitmap, 3).toFloat().div(255), [256, 256], true)
  );
}

/* -------------------------------- Drag / Drop zones -------------------------------- */
function setupDropZone(id, inputId, isMask) {
  const drop = document.getElementById(id);
  const input = document.getElementById(inputId);
  if (!drop || !input) return;

  drop.addEventListener("click", (e) => { e.stopPropagation(); input.click(); });

  input.addEventListener("change", () => {
    const files = Array.from(input.files || []);
    if (isMask) maskFiles = files; else rgbFiles = files;
    updateCount(isMask); updatePreview(isMask, files);
    drop.classList.add("loaded");
  });

  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("dragging");
    const files = Array.from(e.dataTransfer.files || []);
    if (isMask) maskFiles = files; else rgbFiles = files;
    updateCount(isMask); updatePreview(isMask, files);
    drop.classList.add("loaded");
  });
}

/* ---------------------------- Folder auto-pairing ---------------------------- */
let pairLimit = 0; // 0 -> all

function getSortKeyNumericTail(s) {
  const m = s.match(/(\d+)(?!.*\d)/);
  return m ? parseInt(m[1], 10) : Number.POSITIVE_INFINITY;
}

function autoPairFromFolder(fileList) {
  const files = Array.from(fileList).filter((f) => /\.(png|jpg|jpeg|webp|bmp)$/i.test(f.name));
  const rgbMap = new Map();
  const maskMap = new Map();

  files.forEach((f) => {
    const nm = f.name;
    const mm = nm.match(/^(.*)_seg\.(png|jpe?g|webp|bmp)$/i);
    if (mm) { maskMap.set(mm[1].toLowerCase(), f); return; }
    const rm = nm.match(/^(.*)\.(png|jpe?g|webp|bmp)$/i);
    if (rm) {
      const key = rm[1].toLowerCase();
      if (!/_seg$/i.test(key)) rgbMap.set(key, f);
    }
  });

  let keys = [...rgbMap.keys()].filter((k) => maskMap.has(k));
  keys.sort((a, b) => {
    const na = getSortKeyNumericTail(a), nb = getSortKeyNumericTail(b);
    return na === nb ? a.localeCompare(b) : na - nb;
  });

  const limit = pairLimit > 0 ? Math.min(pairLimit, keys.length) : keys.length;
  const pairedRgb = [], pairedMask = [];
  for (let i = 0; i < limit; i++) { const k = keys[i]; pairedRgb.push(rgbMap.get(k)); pairedMask.push(maskMap.get(k)); }

  rgbFiles = pairedRgb; maskFiles = pairedMask;
  updateCount(false); updatePreview(false, rgbFiles);
  updateCount(true);  updatePreview(true,  maskFiles);

  const imgDrop = document.getElementById("imageDrop");
  const mskDrop = document.getElementById("maskDrop");
  if (imgDrop) imgDrop.classList.add("loaded");
  if (mskDrop) mskDrop.classList.add("loaded");

  showToast(`✅ Auto-paired ${limit} image–mask pairs.`, "success");
}

(function setupFolderPicker() {
  const folderInput = document.getElementById("sourceFolderInput");
  if (!folderInput) return;

  folderInput.addEventListener("change", (event) => {
    const files = event.target.files; lastFolderFiles = files;
    const label = document.getElementById("sourceFolderPath");
    if (files && files.length) {
      const any = files[0];
      const folderName = any.webkitRelativePath ? any.webkitRelativePath.split("/")[0] : "(selected)";
      if (label) label.textContent = `Selected Folder: ${folderName}`;
      autoPairFromFolder(files);
    } else {
      if (label) label.textContent = "No folder selected";
      showToast("No folder selected.", "error");
    }
  });

  const numInput = document.getElementById("numImagesInput");
  if (numInput) {
    numInput.addEventListener("input", (e) => {
      const v = parseInt(e.target.value, 10);
      pairLimit = Number.isFinite(v) && v > 0 ? v : 0;
      if (pairLimit === 0) showToast("Using all available pairs.", "success");
      else showToast(`Limit set to ${pairLimit} pairs.`, "success");
      if (lastFolderFiles) autoPairFromFolder(lastFolderFiles);
    });
  }

  const copyBtn = document.getElementById("copyFilesBtn");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      if (!lastFolderFiles) { showToast("Please select a source folder first.", "error"); return; }
      autoPairFromFolder(lastFolderFiles);
    });
  }
})();

/* --------------------------------- Models --------------------------------- */
function convBlock(input, filters) {
  let x = tf.layers.conv2d({ filters, kernelSize: 3, padding: "same" }).apply(input);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.activation("relu").apply(x);
  return x;
}
function upConvBlock(input, skipFeatures, filters) {
  let x = tf.layers.upSampling2d({ size: [2, 2] }).apply(input);
  x = tf.layers.conv2d({ filters, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
  x = tf.layers.concatenate().apply([x, skipFeatures]);
  return convBlock(x, filters);
}
function buildShallowUNetModel() {
  const inputs = tf.input({ shape: [256, 256, 3] });
  const c1 = tf.layers.conv2d({ filters: 8,  kernelSize: 3, activation: "relu", padding: "same" }).apply(inputs);
  const p1 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(c1);
  const c2 = tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: "relu", padding: "same" }).apply(p1);
  const p2 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(c2);
  const b  = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu", padding: "same" }).apply(p2);
  const u1 = tf.layers.conv2dTranspose({ filters: 16, kernelSize: 2, strides: 2, padding: "same" }).apply(b);
  const concat1 = tf.layers.concatenate().apply([c2, u1]);
  const d1 = tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: "relu", padding: "same" }).apply(concat1);
  const u2 = tf.layers.conv2dTranspose({ filters: 8,  kernelSize: 2, strides: 2, padding: "same" }).apply(d1);
  const concat2 = tf.layers.concatenate().apply([c1, u2]);
  const d2 = tf.layers.conv2d({ filters: 8,  kernelSize: 3, activation: "relu", padding: "same" }).apply(concat2);
  const outputs = tf.layers.conv2d({ filters: 1, kernelSize: 1, activation: "sigmoid" }).apply(d2);
  return tf.model({ inputs, outputs });
}
function buildDeepUNetModel() {
  const inputs = tf.input({ shape: [256, 256, 3] });
  let s1 = convBlock(inputs, 8);  let p1 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(s1);
  let s2 = convBlock(p1, 16);     let p2 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(s2);
  let s3 = convBlock(p2, 32);     let p3 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(s3);
  let s4 = convBlock(p3, 64);     let p4 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(s4);
  let bridge = convBlock(p4, 64);
  let u1 = upConvBlock(bridge, s4, 32);
  let u2 = upConvBlock(u1, s3, 16);
  let u3 = upConvBlock(u2, s2, 8);
  let u4 = upConvBlock(u3, s1, 4);
  const outputs = tf.layers.conv2d({ filters: 1, kernelSize: 1, activation: "sigmoid" }).apply(u4);
  return tf.model({ inputs, outputs });
}

/* -------------------------- Dice loss & metric -------------------------- */
function diceLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const smooth = tf.scalar(1e-6);
    const yT = tf.reshape(yTrue, [-1]);
    const yP = tf.reshape(yPred, [-1]);
    const inter = tf.sum(yT.mul(yP));
    const denom = yT.sum().add(yP.sum()).add(smooth);
    const dice  = inter.mul(2).add(smooth).div(denom);
    return tf.scalar(1).sub(dice);
  });
}
function diceCoef(yTrue, yPred) {
  return tf.tidy(() => {
    const smooth = tf.scalar(1e-6);
    const yT = tf.reshape(yTrue, [-1]);
    const yP = tf.reshape(yPred, [-1]);
    const inter = tf.sum(yT.mul(yP));
    const denom = yT.sum().add(yP.sum()).add(smooth);
    return inter.mul(2).add(smooth).div(denom);
  });
}

// ---- Binary morphology helpers (0/1 float masks) ----
function dilate2D(bin, k = 3) {
  return tf.tidy(() => {
    const x4 = bin.expandDims(0).expandDims(-1);     // [1,H,W,1]
    const y4 = tf.maxPool(x4, [k, k], [1, 1], "same");
    return y4.squeeze();                              // [H,W]
  });
}
function erode2D(bin, k = 3) {
  return tf.tidy(() => {
    const inv  = tf.scalar(1).sub(bin);
    const dilI = tf.maxPool(inv.expandDims(0).expandDims(-1), [k, k], [1, 1], "same").squeeze();
    return tf.scalar(1).sub(dilI);
  });
}
function open2D(bin, k = 3)  { return dilate2D(erode2D(bin, k), k); }
function close2D(bin, k = 3) { return erode2D(dilate2D(bin, k), k); }

/* --------------------------- Chart (safe guard) --------------------------- */
const history = { labels: [], loss: [], dice: [], val_loss: [], val_dice: [] };
let chart;

function updateChart() {
  const canvas = document.getElementById("trainingChart");
  if (!canvas || typeof window.Chart === "undefined") return;

  const ctx = canvas.getContext("2d");
  const data = {
    labels: history.labels,
    datasets: [
      { label: "Loss",     data: history.loss,     borderColor: "#e53935", borderWidth: 2, pointRadius: 3, fill: false },
      { label: "Dice",     data: history.dice,     borderColor: "#43a047", borderWidth: 2, pointRadius: 3, fill: false },
      { label: "Val Loss", data: history.val_loss, borderColor: "#fb8c00", borderWidth: 3, pointRadius: 4, fill: false, borderDash: [6,4] },
      { label: "Val Dice", data: history.val_dice, borderColor: "#1e88e5", borderWidth: 3, pointRadius: 4, fill: false, borderDash: [6,4] },
    ],
  };

  const options = {
    responsive: true,
    animation: false,
    scales: { y: { min: 0, max: 1 } },
    interaction: { mode: "nearest", intersect: false },
  };

  if (!chart) chart = new window.Chart(ctx, { type: "line", data, options });
  else { chart.data = data; chart.options = options; chart.update(); }
}

/* ------------------------------ Train ------------------------------ */
(function bindTrain() {
  const btn = document.getElementById("trainBtn");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    showTrainSpinner();
    setTrainingUI(true, "Preparing data…");

    try {
      if (rgbFiles.length === 0 || maskFiles.length === 0) {
        showToast("Upload/auto-pair both RGB and Mask images.", "error");
        return;
      }
      if (rgbFiles.length !== maskFiles.length) {
        showToast("Number of RGB and Mask images must match.", "error");
        return;
      }

      setTrainingUI(true, `Loading ${Math.min(rgbFiles.length, maskFiles.length)} pairs…`);

      const inputImages = [], inputMasks = [];
      for (let i = 0; i < rgbFiles.length; i++) {
        const rgbT = await preprocessImageFile(rgbFiles[i]);
        const mskT = await preprocessMaskFile(maskFiles[i]);
        inputImages.push(rgbT); inputMasks.push(mskT);
      }

      setTrainingUI(true, "Compiling model…");

      const modelType = (document.getElementById("modelSelect") && document.getElementById("modelSelect").value) || "shallow";
      model = modelType === "deep" ? buildDeepUNetModel() : buildShallowUNetModel();
      model.compile({ optimizer: tf.train.adam(), loss: diceLoss, metrics: [diceCoef] });

      const x = tf.stack(inputImages);  // [N,256,256,3]
      const y = tf.stack(inputMasks);   // [N,256,256,1]

      let epochs = parseInt((document.getElementById("epochsInput") && document.getElementById("epochsInput").value), 10);
      if (!Number.isFinite(epochs) || epochs < 1) epochs = 10;

      let batchSize = parseInt((document.getElementById("batchSizeInput") && document.getElementById("batchSizeInput").value), 10);
      if (!Number.isFinite(batchSize) || batchSize < 1)
        batchSize = Math.min(4, Math.max(1, Math.floor(rgbFiles.length / 4)));
      if (batchSize > rgbFiles.length) batchSize = rgbFiles.length;

      let validationSplit = 0.2;
      if (Math.floor(rgbFiles.length * validationSplit) < 1 && rgbFiles.length > 1)
        validationSplit = 1 / rgbFiles.length;

      const bar = document.getElementById("trainBar");
      const prog = document.getElementById("trainProgress");
      const status = document.getElementById("trainStatus");
      if (bar) { bar.value = 0; bar.max = epochs; }
      if (prog) prog.innerText = "";
      if (status) status.innerText = "";

      setTrainingUI(true, "Training… (epoch 1)");

      await model.fit(x, y, {
        epochs, batchSize, shuffle: true, validationSplit,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            const d  = logs.diceCoef ?? logs.dice ?? logs["dice_coef"] ?? 0;
            const vd = logs.val_diceCoef ?? logs.val_dice ?? logs["val_dice_coef"] ?? 0;

            history.loss.push(logs.loss);
            history.dice.push(d);
            history.val_loss.push(logs.val_loss ?? 0);
            history.val_dice.push(vd);
            history.labels.push(`Epoch ${epoch + 1}`);
            updateChart();

            if (bar) bar.value = epoch + 1;
            if (prog) {
              prog.innerText =
                `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, dice=${d.toFixed(4)} | ` +
                `val_loss=${(logs.val_loss ?? 0).toFixed(4)}, val_dice=${vd.toFixed(4)}`;
            }

            setTrainButtonText(`Training… (epoch ${Math.min(epoch + 2, epochs)})`);
            await tf.nextFrame();
          },
        },
      });

      setTrainingUI(true, "Saving model…");
      if (status) status.innerText = "✅ Training finished.";
      showToast("Training finished!", "success");

      if (window.confirm("Save the trained model in this browser for quick reuse?")) {
        if (typeof setTrainButtonText === "function") setTrainButtonText("Saving to browser…");
        await model.save("indexeddb://unet-model");
        showToast("Saved in browser storage.", "success");
      }

      if (window.confirm("Download the model files now?")) {
        if (typeof setTrainButtonText === "function") setTrainButtonText("Preparing download…");
        await model.save("downloads://unet-model");
        showToast("Download started.", "success");
      }

      x.dispose(); y.dispose();
      inputImages.forEach(t => t.dispose());
      inputMasks.forEach(t => t.dispose());

    } catch (err) {
      console.error(err);
      showToast("Training failed. See console.", "error");
    } finally {
      hideTrainSpinner();
      setTrainingUI(false);
    }
  });
})();

/* -------------------- Predict (left button) -------------------- */
async function handlePredictClick() {
  try {
    const fileInput = document.getElementById("predictImage");
    const file = fileInput && fileInput.files && fileInput.files[0];
    if (!file) { showToast("Select an RGB image for prediction.", "error"); return; }

    if (!model) {
      const models = await tf.io.listModels();
      if ("indexeddb://unet-model" in models) {
        model = await tf.loadLayersModel("indexeddb://unet-model");
        model.compile({ optimizer: tf.train.adam(), loss: diceLoss, metrics: [diceCoef] });
      } else { showToast("No model loaded. Train or upload a model first.", "error"); return; }
    }

    //read numeric threshold from #thresholdInput
    const tInput = document.getElementById("thresholdInput");
    let threshold = 0.5;
    if (tInput) threshold = Math.max(0, Math.min(1, Number(tInput.value) || 0.5));

    const resultCanvas = document.getElementById("resultCanvas");
    if (!resultCanvas) { showToast("Result canvas not found.", "error"); return; }

    const imgBitmap = await loadBitmapCrossBrowser(file);
    const input = await preprocessImageBitmap(imgBitmap);
    const pred  = model.predict(input.expandDims(0)).squeeze();
    input.dispose();

    let bin = tf.tidy(() => pred.greater(tf.scalar(threshold)).toFloat());
    bin = tf.tidy(() => close2D(open2D(bin, 3), 3));        // small clean-up

    await tf.browser.toPixels(bin, resultCanvas);

    pred.dispose(); bin.dispose();
    showToast("Prediction complete! (binary mask)", "success");
  } catch (err) {
    console.error(err);
    showToast("Prediction failed. See console.", "error");
  }
}
function bindPredictButton() {
  const btn = document.getElementById("predictBtn");
  if (!btn || btn.dataset.bound === "1") return;
  btn.dataset.bound = "1";
  btn.addEventListener("click", handlePredictClick);
}
document.addEventListener("DOMContentLoaded", bindPredictButton);
window.addEventListener("load", bindPredictButton);

/* ------------------- Predict UI (image upload preview) ------------------- */
(function bindPredictUI() {
  const predictDrop = document.getElementById("predictDrop");
  const predictInput = document.getElementById("predictImage");
  if (predictDrop && predictInput) {
    predictDrop.addEventListener("click", () => predictInput.click());
    predictInput.addEventListener("change", () => {
      const n = predictInput.files ? predictInput.files.length : 0;
      const el = document.getElementById("predictCount");
      if (el) el.textContent = n > 0 ? `${n} image${n > 1 ? "s" : ""} uploaded` : "No image uploaded";
      const preview = document.getElementById("predictPreview");
      if (preview) {
        preview.innerHTML = "";
        if (n > 0) {
          const img = document.createElement("img");
          img.src = URL.createObjectURL(predictInput.files[0]);
          preview.appendChild(img);
        }
      }
    });
  }
})();

/* --------------------- RIGHT PANEL: GT + Accuracy --------------------- */
async function drawMaskToCanvas(maskTensor /* (H,W) or (H,W,1) 0/1 */, canvas) {
  const t = maskTensor.rank === 3 ? maskTensor.squeeze() : maskTensor;
  await tf.browser.toPixels(t, canvas);
}

async function computePredictionAndScores() {
  const imgFile = document.getElementById("predictImage")?.files?.[0];
  const gtFile  = document.getElementById("gtMaskInput")?.files?.[0];
  const scoreEl = document.getElementById("scoreText");
  const gtStatus = document.getElementById("gtStatus");
  const leftCanvas = document.getElementById("resultCanvas");
  const rightCanvas = document.getElementById("gtCanvas");

  if (!imgFile) { showToast("Select an RGB image first.", "error"); return; }

  try {
    if (!model) {
      const models = await tf.io.listModels();
      if ("indexeddb://unet-model" in models) {
        model = await tf.loadLayersModel("indexeddb://unet-model");
        model.compile({ optimizer: tf.train.adam(), loss: diceLoss, metrics: [diceCoef] });
      } else { showToast("No model loaded. Train or upload a model first.", "error"); return; }
    }

    const tInput = document.getElementById("thresholdInput");
    let threshold = 0.5;
    if (tInput) threshold = Math.max(0, Math.min(1, Number(tInput.value) || 0.5));

    // predict
    const imgBitmap = await loadBitmapCrossBrowser(imgFile);
    const input = await preprocessImageBitmap(imgBitmap);
    const prob = model.predict(input.expandDims(0)).squeeze();
    input.dispose();

    let predBin = tf.tidy(() => prob.greater(tf.scalar(threshold)).toFloat());
    predBin = tf.tidy(() => close2D(open2D(predBin, 3), 3));
    await tf.browser.toPixels(predBin, leftCanvas);

    if (gtFile) {
      const gt = await preprocessMaskFile(gtFile);   // (256,256,1) 0/1
      await drawMaskToCanvas(gt, rightCanvas);

      const eps = tf.scalar(1e-7);
      const { dice, iou, prec, rec, acc } = tf.tidy(() => {
        const g = gt.squeeze();      // (256,256)
        const p = predBin;           // (256,256)
        const tp   = p.mul(g).sum();
        const fp   = p.sum().sub(tp);
        const fn   = g.sum().sub(tp);
        const total = tf.scalar(256 * 256, "float32");
        const tn   = total.sub(tp.add(fp).add(fn));

        const dice = tp.mul(2).add(eps).div(p.sum().add(g.sum()).add(eps));
        const iou  = tp.add(eps).div(tp.add(fp).add(fn).add(eps));
        const pr   = tp.add(eps).div(tp.add(fp).add(eps));
        const rc   = tp.add(eps).div(tp.add(fn).add(eps));
        const acc  = tp.add(tn).add(eps).div(total.add(eps));

        return {
          dice: dice.dataSync()[0],
          iou:  iou.dataSync()[0],
          prec: pr.dataSync()[0],
          rec:  rc.dataSync()[0],
          acc:  acc.dataSync()[0]
        };
      });

      if (scoreEl) {
        scoreEl.textContent =
          `Dice: ${dice.toFixed(3)} | IoU: ${iou.toFixed(3)} | ` +
          `Precision: ${prec.toFixed(3)} | Recall: ${rec.toFixed(3)} | ` +
          `Accuracy: ${acc.toFixed(3)}`;
      }
      if (gtStatus) gtStatus.textContent = gtFile.name;
      gt.dispose();
    } else {
      if (scoreEl) scoreEl.textContent = "No ground-truth provided. Upload a mask to compute scores.";
      if (rightCanvas) {
        const ctx = rightCanvas.getContext("2d");
        ctx.clearRect(0,0,rightCanvas.width,rightCanvas.height);
      }
    }

    prob.dispose();
    predBin.dispose();
    showToast("Prediction complete (with scoring).", "success");
  } catch (err) {
    console.error(err);
    showToast("Prediction/scoring failed. See console.", "error");
  }
}

function bindGTPanel() {
  const drop = document.getElementById("gtDrop");
  const input = document.getElementById("gtMaskInput");
  const status = document.getElementById("gtStatus");
  const previewCanvas = document.getElementById("gtCanvas");
  if (!drop || !input) return;

  const renderPreview = async (file) => {
    if (!file) { status && (status.textContent = "Click to select or drop a file"); return; }
    const gt = await preprocessMaskFile(file);
    await drawMaskToCanvas(gt, previewCanvas);
    status && (status.textContent = file.name);
    gt.dispose();
  };

  drop.addEventListener("click", () => input.click());
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    drop.classList.remove("dragging");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    input.files = e.dataTransfer.files;
    await renderPreview(f);
  });
  input.addEventListener("change", async () => {
    const f = input.files && input.files[0];
    await renderPreview(f);
  });

  const scoreBtn = document.getElementById("scoreBtn");
  if (scoreBtn) scoreBtn.addEventListener("click", computePredictionAndScores);
}

/* -------------------------- VIDEO SEGMENTATION -------------------------- */
function once(el, event) {
  return new Promise((resolve) => {
    const handler = () => { el.removeEventListener(event, handler); resolve(); };
    el.addEventListener(event, handler);
  });
}
function seek(video, t) {
  return new Promise((resolve) => {
    const onSeeked = () => { video.removeEventListener("seeked", onSeeked); resolve(); };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = Math.min(Math.max(0, t), Math.max(0, video.duration - 1e-3));
  });
}
function pickMimeType() {
  if (!("MediaRecorder" in window)) return "";
  const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4;codecs=h264", "video/mp4"];
  for (const t of types) if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t;
  return "";
}

let cancelVideo = false;

(function bindVideoUI() {
  const fileInput = document.getElementById("videoFile");
  const videoEl   = document.getElementById("inputVideo");
  if (fileInput && videoEl) {
    fileInput.addEventListener("change", () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f) return;
      videoEl.src = URL.createObjectURL(f);
      videoEl.style.display = "block";
      videoEl.load();
    });
  }
  const startBtn = document.getElementById("startVideoBtn");
  const stopBtn  = document.getElementById("stopVideoBtn");
  if (startBtn) startBtn.addEventListener("click", startVideoSegmentation);
  if (stopBtn)  stopBtn .addEventListener("click", () => { cancelVideo = true; });
})();

// REPLACED to use #thresholdInput
async function startVideoSegmentation() {
  var prevProb = null;
  const smooth = 0.7;

  try {
    const videoEl   = document.getElementById("inputVideo");
    const outCanvas = document.getElementById("videoCanvas");
    const statusEl  = document.getElementById("videoStatus");
    const prog      = document.getElementById("videoProgress");
    const link      = document.getElementById("videoDownload");
    const modeSel   = document.getElementById("videoMode");
    const fpsInput  = document.getElementById("videoFps");
    const strideIn  = document.getElementById("frameStride");
    const tInput    = document.getElementById("thresholdInput");

    if (link) link.style.display = "none";
    if (!videoEl || !videoEl.src) { showToast("Pick a video first.","error"); return; }

    if (!model) {
      const models = await tf.io.listModels();
      if ("indexeddb://unet-model" in models) {
        model = await tf.loadLayersModel("indexeddb://unet-model");
        model.compile({ optimizer: tf.train.adam(), loss: diceLoss, metrics: [diceCoef] });
      } else {
        showToast("No model loaded. Train or upload first.","error");
        return;
      }
    }

    if (!isFinite(videoEl.duration) || videoEl.videoWidth === 0) {
      await new Promise(r => videoEl.addEventListener("loadedmetadata", r, { once: true }));
    }
    const vw = videoEl.videoWidth || 640;
    const vh = videoEl.videoHeight || 360;

    outCanvas.width = vw; outCanvas.height = vh;
    const ctx = outCanvas.getContext("2d");

    const work = document.createElement("canvas"); work.width = 256; work.height = 256;
    const wctx = work.getContext("2d", { willReadFrequently: true });
    const overlayCanvas = document.createElement("canvas"); overlayCanvas.width = 256; overlayCanvas.height = 256;
    const octx = overlayCanvas.getContext("2d");
    const maskCanvas = document.createElement("canvas"); maskCanvas.width = 256; maskCanvas.height = 256;
    const mctx = maskCanvas.getContext("2d");

    const fps    = Math.max(1, Math.min(60, parseInt(fpsInput?.value || "15", 10)));
    const stride = Math.max(1, parseInt(strideIn?.value || "1", 10));
    const mode   = modeSel?.value || "overlay";
    const threshold = Math.max(0, Math.min(1, Number(tInput?.value ?? 0.5)));

    const mimeType = (function pick() {
      if (!("MediaRecorder" in window)) return "";
      const ts = ["video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm","video/mp4;codecs=h264","video/mp4"];
      for (const t of ts) if (MediaRecorder.isTypeSupported?.(t)) return t;
      return "";
    })();

    let recorder = null, chunks = [];
    if (mimeType) {
      const stream = outCanvas.captureStream(fps);
      recorder = new MediaRecorder(stream, { mimeType });
      recorder.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType });
        const url  = URL.createObjectURL(blob);
        if (link) {
          link.href = url;
          link.download = mimeType.includes("mp4") ? "segmented.mp4" : "segmented.webm";
          link.style.display = "inline-block";
        }
      };
      recorder.start();
    }

    const totalFrames = Math.floor(videoEl.duration * fps);
    cancelVideo = false;
    if (prog) { prog.max = 100; prog.value = 0; }
    if (statusEl) statusEl.textContent = `Processing… 0/${totalFrames}`;

    const seek = t => new Promise(res => {
      const h = () => { videoEl.removeEventListener("seeked", h); res(); };
      videoEl.addEventListener("seeked", h);
      videoEl.currentTime = Math.min(Math.max(0, t), Math.max(0, videoEl.duration - 0.001));
    });

    for (let f = 0; f < totalFrames && !cancelVideo; f += stride) {
      tf.engine().startScope();
      try {
        const t = f / fps;
        await seek(t);

        wctx.clearRect(0, 0, work.width, work.height);
        wctx.drawImage(videoEl, 0, 0, work.width, work.height);

        const input = tf.tidy(() =>
          tf.image.resizeBilinear(tf.browser.fromPixels(work, 3).toFloat().div(255), [256, 256], true)
        );
        const pred = model.predict(input.expandDims(0)).squeeze();
        input.dispose();

        let prob;
        if (prevProb) {
          prob = tf.tidy(() => prevProb.mul(smooth).add(pred.mul(1 - smooth)));
          prevProb.dispose();
        } else {
          prob = pred.clone();
        }
        prevProb = tf.keep(prob.clone());

        let bin = tf.tidy(() => prob.greater(tf.scalar(threshold)).toFloat());
        bin = tf.tidy(() => close2D(open2D(bin, 3), 3));

        ctx.clearRect(0, 0, vw, vh);
        if (mode === "overlay") {
          ctx.drawImage(videoEl, 0, 0, vw, vh);
          const octx2 = overlayCanvas.getContext("2d");
          octx2.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
          await tf.browser.toPixels(bin, overlayCanvas);
          octx2.globalCompositeOperation = "source-in";
          octx2.fillStyle = "rgba(255,0,0,0.35)";
          octx2.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
          octx2.globalCompositeOperation = "source-over";
          ctx.drawImage(overlayCanvas, 0, 0, vw, vh);
        } else {
          mctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
          await tf.browser.toPixels(bin, maskCanvas);
          ctx.drawImage(maskCanvas, 0, 0, vw, vh);
        }

        pred.dispose();
        bin.dispose();
        prob.dispose();

        await tf.nextFrame();
        if (statusEl) statusEl.textContent = `Processing… ${Math.min(f + 1, totalFrames)}/${totalFrames}`;
        if (prog)     prog.value = Math.round(((f + 1) / totalFrames) * 100);

      } finally {
        tf.engine().endScope();
      }
    }

    if (prevProb) { prevProb.dispose(); prevProb = null; }

    if (recorder && recorder.state !== "inactive") recorder.stop();
    if (cancelVideo) { statusEl && (statusEl.textContent = "Stopped."); showToast("Stopped video processing.","error"); }
    else { statusEl && (statusEl.textContent = "✅ Done."); showToast("Video segmentation complete!","success"); }

  } catch (err) {
    console.error(err);
    showToast("Video processing failed. See console.","error");
  } finally {
    cancelVideo = false;
  }
}

/* --------------------------------- Model I/O --------------------------------- */
(function bindModelIO() {
  const downloadBtn = document.getElementById("downloadModelBtn");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", async () => {
      if (!model) { showToast("Train or load a model before downloading.", "error"); return; }
      await model.save("downloads://unet-model");
      showToast("Model downloaded.", "success");
    });
  }

  const uploadInput = document.getElementById("uploadModelInput");
  if (uploadInput) {
    uploadInput.addEventListener("change", async (e) => {
      const files = e.target.files;
      try {
        model = await tf.loadLayersModel(tf.io.browserFiles([...files]));
        model.compile({ optimizer: tf.train.adam(), loss: diceLoss, metrics: [diceCoef] });
        showToast("✅ Model uploaded and loaded successfully.", "success");
      } catch (err) {
        console.error("Error loading model:", err);
        showToast("❌ Failed to load model from uploaded files.", "error");
      }
    });
  }

  const clearBtn = document.getElementById("clearModelBtn");
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      try { await tf.io.removeModel("indexeddb://unet-model"); showToast("Model cleared from browser storage.", "success"); }
      catch (e) { console.warn("Failed to clear model:", e); showToast("No saved model found.", "error"); }
    });
  }
})();

/* ---------------------------------- On load ---------------------------------- */
window.addEventListener("load", async () => {
  // the threshold input is numeric (#thresholdInput)
  const tInput = document.getElementById("thresholdInput");
  if (tInput) {
    // clamp to [0,1]
    tInput.addEventListener("change", () => {
      const v = Math.max(0, Math.min(1, Number(tInput.value) || 0.5));
      tInput.value = v.toString();
    });
  }

  try {
    const models = await tf.io.listModels();
    if ("indexeddb://unet-model" in models) {
      model = await tf.loadLayersModel("indexeddb://unet-model");
      model.compile({ optimizer: tf.train.adam(), loss: diceLoss, metrics: [diceCoef] });
      showToast("Loaded saved model from browser storage.", "success");
    }
  } catch (e) { console.warn("⚠️ Failed to load model from IndexedDB:", e); }

  setupDropZone("imageDrop", "inputImage", false);
  setupDropZone("maskDrop", "inputMask", true);
  updateCount(false); updateCount(true);

  // bind right-side GT + Accuracy UI
  bindGTPanel();
});
