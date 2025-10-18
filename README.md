
# 🧠 Browser-Based U-Net for Binary Image Segmentation

This project implements a **fully client-side U-Net model** for image segmentation using **TensorFlow.js**.
The system trains and runs directly inside a **web browser**, without any server or external GPU.
It supports both **online and offline** execution and provides live visualization of training metrics, prediction masks, and performance statistics.

---

## 📁 Project Structure

```
├── index.html        # Frontend UI
├── app.js            # Main training and prediction logic (TensorFlow.js)
├── style.css         # Styling for user interface
├── libs/             # Local TensorFlow.js and Chart.js offline libraries
├── README.md         # Project documentation

```

---

## 💾 Dataset

All experiments were done using the **Benchmark for Automatic Glottis Segmentation (BAGLS)** dataset [(Gómez et al., 2019)](https://www.bagls.org/).
It provides high-speed videoendoscopy frames and corresponding segmentation masks.
For this project, **1000 paired RGB and mask images** were used.
All images were resized to **256×256**, and masks were binarized at a **0.5 threshold** using nearest-neighbor interpolation.

---

## ⚙️ Implementation

The application uses:

* **TensorFlow.js (v4.20.0)** for model building and training
* **Chart.js (v4.4.4)** for real-time visualization
* **IndexedDB** for saving models in the browser
* **Service Worker (sw.js)** for offline functionality

Two U-Net variants are available:

* *Shallow U-Net*: Fast, suitable for smaller datasets
* *Deep U-Net*: Includes BatchNorm layers and deeper encoder-decoder structure

Training is based on the **Dice loss** with **Adam optimizer (lr = 0.001)**.
The number of epochs and batch size can be set directly from the interface.

---

## 📊 Results Summary

| Metric           | Value (WebGL, 1000 pairs) |
| :--------------- | :-----------------------: |
| Dice Coefficient |           0.890           |
| IoU              |           0.802           |
| Precision        |           1.000           |
| Recall           |           0.802           |
| Accuracy         |           0.994           |

Training time for 1000 pairs (WebGL backend): **~74 min**
CPU backend is slower but supports systems without GPU acceleration.

---

## 🔬 Reproducibility

You can reproduce the experiments locally by following these steps:

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/unet-segmentation-app.git
   cd unet-segmentation-app
   ```
2. Open `index.html` in your browser.
3. Upload paired RGB and mask images.
4. Train and visualize results directly in the browser.


No installation is required — everything runs **directly in the browser**.

---

## 📝 License

This project is licensed under the [MIT License](./LICENSE).
.

---


