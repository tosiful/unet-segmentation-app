# Browser-Based U-Net Image Segmentation

A client-side web application for training and using U-Net models for binary image segmentation directly in a web browser. It is built with plain HTML, CSS, JavaScript, and TensorFlow.js. No Python environment or application server is required to run the interface and model workflow.

The application provides two U-Net architectures, image-mask dataset preparation, browser-based training, image prediction, optional video-frame segmentation, metric visualization, and model import/export.

## Contents

- [Features](#features)
- [How the application works](#how-the-application-works)
- [Dataset](#dataset)
- [Model architectures](#model-architectures)
- [Training configuration](#training-configuration)
- [Image prediction and evaluation](#image-prediction-and-evaluation)
- [Video segmentation](#video-segmentation)
- [Browser backends and performance monitoring](#browser-backends-and-performance-monitoring)
- [Technologies](#technologies)
- [Repository structure](#repository-structure)
- [Run the project](#run-the-project)
- [Important implementation notes](#important-implementation-notes)
- [Reported experiment](#reported-experiment)
- [Possible future improvements](#possible-future-improvements)
- [License](#license)

## Features

- **Two U-Net variants:** choose between a Shallow U-Net and a Deep U-Net.
- **Image-mask folder pairing:** select a folder containing RGB images and masks named with the `_seg` suffix; matching files are paired automatically.
- **Dataset-size control:** use all available pairs or set a pair limit.
- **Browser-based training:** configure the number of epochs, batch size, and TensorFlow.js backend.
- **Training visualization:** view training and validation loss and Dice coefficient with Chart.js.
- **Image segmentation:** upload an image and generate a binary segmentation mask.
- **Configurable threshold:** adjust the probability threshold used to convert model output into a binary mask.
- **Prediction evaluation:** compare a prediction with a supplied ground-truth mask using Dice, IoU, precision, recall, and accuracy.
- **Video segmentation:** process a video frame by frame and produce either a red-mask overlay or a black-and-white mask video.
- **Model persistence:** save a trained model in browser IndexedDB, download it, upload an existing model, or clear the browser-saved model.
- **Timing and memory information:** inspect backend setup, preprocessing, model build/compile, tensor stacking, epoch duration, total training time, and TensorFlow.js memory information.
- **CSV export:** export the values displayed in the timing table.

## How the application works

The main workflow is:

1. Select a dataset folder or upload RGB images and their corresponding masks.
2. Pair each RGB image with its ground-truth mask.
3. Preprocess the image and mask to the model input size.
4. Select the Shallow or Deep U-Net architecture and configure training.
5. Train the model in the browser.
6. Review the training and validation curves.
7. Upload an image to generate a segmentation mask.
8. Optionally upload its ground-truth mask to calculate evaluation metrics.

## Dataset

The project README's original experiment description identifies the **Benchmark for Automatic Glottis Segmentation (BAGLS)** dataset ([BAGLS website](https://www.bagls.org/)), which contains video-endoscopy frames and corresponding segmentation masks.

The documented experiment used **1,000 paired RGB images and masks**. The application code expects RGB input images and corresponding single-channel masks.

### Preprocessing implemented in the code

- **Input images:** read as three-channel RGB, converted to floating-point values, divided by 255, and resized to **256 × 256** using bilinear interpolation.
- **Masks:** read as a single channel, converted to floating-point values, divided by 255, binarized using a threshold greater than **0.5**, and resized to **256 × 256** using nearest-neighbor interpolation.
- **Supported image formats:** PNG, JPG/JPEG, WebP, and BMP.
- **Automatic pairing:** a mask is recognized when its filename ends in `_seg` before the file extension. For example, `frame001.png` is paired with `frame001_seg.png`.

Nearest-neighbor resizing is used for masks to avoid introducing interpolated label values.

## Model architectures

Both models accept an input tensor of shape **256 × 256 × 3** and produce a one-channel output with a sigmoid activation. The output represents a per-pixel foreground probability.

### Shallow U-Net

The Shallow U-Net has two encoder downsampling stages:

- Encoder feature channels: 8 and 16.
- Bottleneck: 32 channels.
- Decoder: transposed convolutions, skip concatenations, and convolution layers.
- Skip connections combine encoder features with decoder features at matching spatial resolutions.
- Final layer: a 1 × 1 convolution with one output channel and sigmoid activation.

This is the smaller architecture intended to reduce model complexity and computation.

### Deep U-Net

The Deep U-Net has four encoder downsampling stages and a deeper bottleneck:

- Encoder feature channels: 8, 16, 32, and 64.
- Bottleneck: 64 channels.
- Convolution blocks use convolution, batch normalization, and ReLU activation.
- Decoder blocks upsample feature maps and concatenate them with the corresponding encoder features.
- Final layer: a 1 × 1 convolution with one output channel and sigmoid activation.

The deeper architecture can learn richer feature representations, but it also requires more computation. A deeper model is not automatically more accurate; performance must be evaluated on suitable validation or test data.

## Training configuration

The training interface allows the user to select:

| Setting | Implementation |
|---|---|
| Model | Shallow U-Net or Deep U-Net |
| Epochs | Configurable; interface default is 10 |
| Batch size | Configurable; interface default is 4 |
| Validation split | 20% when the dataset is large enough; adjusted for very small datasets |
| Optimizer | Adam |
| Loss | Dice loss |
| Training metric | Dice coefficient |
| Input size | 256 × 256 × 3 |
| Mask size | 256 × 256 × 1 |

The Dice loss implemented by the application is:

`DiceLoss = 1 - DiceCoefficient`

The Dice coefficient is calculated from the overlap between the ground-truth mask and predicted probabilities, with a small smoothing value to avoid division by zero.

During training, the application records loss, Dice, validation loss, and validation Dice for each epoch. These values are displayed as line charts.

## Image prediction and evaluation

After training, the user can upload an RGB image. The application:

1. Resizes and normalizes the image using the same preprocessing approach as training.
2. Runs the image through the loaded U-Net model.
3. Applies the selected threshold (default **0.5**) to convert probabilities into a binary mask.
4. Applies 3 × 3 morphological opening followed by closing to perform small mask clean-up.
5. Displays the resulting binary mask.

To calculate evaluation metrics, upload the corresponding ground-truth mask as well.

The application calculates:

| Metric | Meaning |
|---|---|
| **Dice coefficient** | Measures overlap between prediction and ground truth. |
| **Intersection over Union (IoU)** | Measures intersection divided by union. |
| **Precision** | Measures how many predicted foreground pixels are correct. |
| **Recall** | Measures how much of the ground-truth foreground is detected. |
| **Accuracy** | Measures the proportion of correctly classified pixels, including background pixels. |

For binary segmentation, the metrics are based on true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN):

- Dice = 2TP / (2TP + FP + FN)
- IoU = TP / (TP + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Evaluation note:** accuracy can be high when most pixels belong to the background. Dice and IoU should therefore also be considered when assessing foreground segmentation quality.

## Video segmentation

The video feature applies the trained model to frames from an uploaded video.

Available options include:

- **Overlay mode:** displays the predicted mask as a semi-transparent red overlay on the original frame.
- **Mask-only mode:** displays the binary mask without the original frame.
- **Target FPS:** controls the frame rate used for processing and recording (interface default: 15).
- **Frame stride:** controls how many frame intervals are skipped between processed frames (interface default: 1).
- **Stop control:** requests that frame processing stop.
- **Download:** records the output canvas with the browser's MediaRecorder API and provides a downloadable video when a supported recording format is available.

The code also applies temporal probability smoothing between consecutive processed frames:

`smoothed = 0.7 × previous + 0.3 × current`

This is intended to reduce abrupt changes in predictions between frames. Video processing is frame-based and should not be described as guaranteed real-time processing; speed depends on the browser, hardware, backend, model, video resolution, and selected frame settings.

## Browser backends and performance monitoring

The application uses TensorFlow.js and offers three execution backends:

- **WebGL:** uses browser graphics capabilities for tensor operations when supported.
- **WASM:** uses the TensorFlow.js WebAssembly backend.
- **CPU:** uses the JavaScript CPU backend.

At initialization, the code attempts WebGL, then WASM, then CPU. The interface also allows the user to select a backend.

The timing panel records backend setup, image/mask preprocessing, model build and compilation, tensor stacking, epoch times, total training time, and a TensorFlow.js memory snapshot. The timing table can be exported as CSV.

Backend speed is hardware- and browser-dependent. No backend should be assumed to be fastest on every device.

## Technologies

- **HTML5** — application structure and controls.
- **CSS3** — layout and styling.
- **JavaScript (ES6+)** — data handling, model workflow, prediction, and video processing.
- **TensorFlow.js 4.20.0** — U-Net model construction, training, and inference in the browser.
- **TensorFlow.js WASM backend** — optional WebAssembly execution.
- **Chart.js 4.4.4** — training and validation charts.
- **Canvas API** — image/mask rendering and video-frame composition.
- **MediaRecorder API** — recording segmented video output.
- **IndexedDB** — storing a trained model in browser storage.

## Repository structure

```text
unet-segmentation-app/
├── index.html       # Application interface and script references
├── app.js           # TensorFlow.js setup, preprocessing, models, training and prediction
├── style.css        # Application styling
├── libs/            # Local TensorFlow.js, WASM backend and Chart.js assets
├── LICENSE          # MIT license
└── README.md        # Project documentation
```

## Run the project

### Option 1: Open the application

1. Clone the repository:

   ```bash
   git clone https://github.com/tosiful/unet-segmentation-app.git
   cd unet-segmentation-app
   ```

2. Open `index.html` in a modern browser.

3. Select a dataset folder containing matching RGB and `_seg` mask files, or upload the RGB images and masks separately.

4. Select a model architecture, set the epochs and batch size, choose a backend, and click **Train U-Net**.

5. After training, upload an image in the prediction section. Upload its ground-truth mask as well if you want to calculate evaluation metrics.

### Option 2: Use a local static web server

If browser file or resource restrictions cause problems when opening the HTML file directly, serve the project directory with a local static server. For example, if Python is installed:

```bash
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

No Python ML environment is required; Python is only used here to serve the static files.

## Important implementation notes

- Training and inference run in the browser; the repository does not contain a model-training API or application backend.
- Browser memory and hardware can limit the dataset size and model complexity that can be used comfortably.
- The HTML currently includes CDN script references as well as local library assets. Therefore, fully offline operation should be verified in the target environment rather than assumed.
- The repository currently does not contain a `sw.js` service-worker file. This README therefore does not claim service-worker-based offline support.
- Reported scores and training times depend on the dataset, train/validation split, model configuration, browser, and hardware. They should be presented with the experimental setup.

## Reported experiment

The previous version of this README reported the following values for a **1,000-pair WebGL experiment**:

| Metric | Previously reported value |
|---|---:|
| Dice coefficient | 0.890 |
| IoU | 0.802 |
| Precision | 1.000 |
| Recall | 0.802 |
| Accuracy | 0.994 |
| Training time | Approximately 74 minutes |

These are retained as previously reported project results, not as independently verified benchmark results from this documentation update. The repository code alone does not establish the exact hardware, browser version, data split, or repeatability of that run. Include those details when reporting the result in a thesis, presentation, or interview, if available.

## Possible future improvements

- Add data augmentation and evaluate its effect on generalization.
- Add reproducible train/validation/test splits and save experiment configurations.
- Compare Shallow and Deep U-Net using the same split and backend conditions.
- Add batch-wise or streaming data loading to reduce browser memory pressure on larger datasets.
- Add more detailed benchmark exports, including per-epoch metrics and hardware/browser information.
- Test performance across browsers and devices.
- Verify and document a complete offline-loading workflow.

## License

This project is licensed under the [MIT License](./LICENSE).
