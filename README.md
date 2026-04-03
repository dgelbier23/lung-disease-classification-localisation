<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0a,50:1a1a2e,100:16213e&height=200&section=header&text=Lung%20Disease%20Classification%20and%20Localisation&fontSize=28&fontColor=ffffff&fontAlignY=38&desc=Transfer%20Learning%20%7C%20Xception%20%7C%20Grad-CAM%20%7C%20Chest%20X-Ray%20Imaging&descAlignY=60&descSize=15&descColor=a8b2d8" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-D00000?style=flat-square&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

</div>

---

## Overview

This project develops a **multi-class lung disease classifier** capable of distinguishing between **Healthy**, **Pneumonia**, and **COVID-19** chest X-ray images — with spatial localisation of pathological regions via **Grad-CAM**.

Rather than treating this as a straightforward fine-tuning exercise, the system was designed around the constraints of real clinical deployment: class imbalance, overlapping radiographic features, limited data, and the asymmetric cost of false negatives. Every architectural and training decision was made with those constraints in mind.

**Best model (B12):** Macro F1 = **0.9869** · COVID Recall = **0.9926** · Pneumonia Recall = **0.9851**

---

## Motivation

Chest radiography is among the most common diagnostic tools in clinical medicine, yet radiologist throughput remains a bottleneck — particularly for time-critical conditions like COVID-19 and community-acquired pneumonia. Automated screening systems that prioritise sensitivity on disease classes could meaningfully reduce diagnostic delay and support clinical triage.

The core challenge is not classification accuracy per se — it is building a system that **fails safely**: one that is more likely to escalate a borderline case than to dismiss it. This framing shaped every design decision in this project.

---

## Methodology

### Data

| Subset | Images | Split |
|---|---|---|
| Training | 10,606 | 70% |
| Validation | 2,274 | 15% |
| Test | 2,273 | 15% |
| **Total** | **15,153** | — |

- Three classes: `HEALTHY`, `PNEUMONIA`, `COVID`
- Class distribution is imbalanced — Healthy images significantly outnumber disease cases
- Stratified splitting preserves class proportions across all subsets
- Labels: `0 = Healthy`, `1 = Pneumonia`, `2 = COVID`

### Preprocessing Pipeline

All images follow a fixed preprocessing path regardless of split:

```
Raw Chest X-ray (Grayscale)
        ↓
Grayscale → RGB (3-channel conversion for Xception compatibility)
        ↓
Resize + Pad to 299×299 (aspect-ratio preserving, no distortion)
        ↓
Cast to Float32
        ↓
Xception normalisation ([-1, 1] scaling)
        ↓  ← Training only: brightness + contrast augmentation
Model Input
```

Design rationale:
- **Minimal augmentation** was a deliberate choice. Aggressive augmentation on medical images risks destroying the subtle intra-class signal (e.g. ground-glass opacity patterns) that discriminates COVID from Pneumonia.
- **Padding over cropping** preserves anatomical structures at image boundaries.
- Preprocessing was fixed across all experiments to ensure controlled comparison.

### Class Imbalance Strategy

**Class weighting** was selected over oversampling or undersampling:

| Strategy | Rejected Because |
|---|---|
| Undersampling | Loses majority-class data; underfitting risk on a 15K dataset |
| SMOTE / synthetic oversampling | Generates artificial medical image features; high risk of spurious learning |
| Simple image duplication | Overfits minority classes through repetition |
| **Class weighting** ✓ | Preserves all real data; scales loss contribution without synthetic artefacts |

Weights were computed from training class frequencies and injected directly into the loss function.

### Model Architecture

The final architecture combines a **pretrained Xception backbone** with a deliberately lightweight classification head:

```
Input: 299 × 299 × 3
        ↓
Xception Backbone (ImageNet pretrained)
  └─ Depthwise separable convolutions
  └─ 22.9M parameters, 126 layers
  └─ Frozen in Phase A; partially unfrozen in Phase B
        ↓
Global Average Pooling
  └─ Aggregates spatial features without parameter explosion
  └─ Preserves distributed texture patterns (superior to MaxPool for diffuse pathology)
        ↓
Dropout (p=0.5)
  └─ Regularisation against co-adaptation
        ↓
Dense (Softmax, 3 classes)
        ↓
Output: [P(Healthy), P(Pneumonia), P(COVID)]
```

**Why Xception over alternatives:**

| Model | Size | Parameters | Decision |
|---|---|---|---|
| Xception ✓ | 88 MB | 22.9M | Native 299×299; depthwise separable convolutions ideal for texture |
| VGG19 | 549 MB | 143.7M | Far too large; parameter inefficient |
| ResNet50 | 98 MB | 25.6M | Viable alternative; Xception preferred for input resolution match |
| InceptionV3 | 92 MB | 23.9M | Similar; Xception has better spatial efficiency |

The classification head was kept minimal by design. Adding dense layers would increase overfitting risk without improving the backbone's feature representation — a common mistake in transfer learning for small medical datasets.

### Training Strategy

Training was split into two phases across a staged experimental design:

**Phase A — Head Training (4 epochs)**
- Backbone fully frozen
- Adam optimiser, LR = 1×10⁻³
- Rapid adaptation of randomly initialised head layers

**Phase B — Fine-tuning (15 epochs, selected experiments)**
- Top N% of backbone layers unfrozen
- Backbone LR = 1×10⁻⁴ (order of magnitude lower than head LR)
- Smaller LR preserves pretrained representations while allowing domain adaptation

**Shared training config:**
- Early stopping: patience = 5 epochs
- `ReduceLROnPlateau`: evaluated every 3 epochs
- Batch size: 16 (GPU memory constraint with 299×299 input)
- Max epochs: 15 (fine-tuning phase)

### Experimental Design

Experiments were run across 4 controlled stages, varying one factor at a time:

| Stage | Variable | Best Finding |
|---|---|---|
| 0 | Baseline | Adam + LR 1e-3 |
| 1 | Learning Rate Sensitivity | Higher LR wins for head training; LR decay hurts |
| 2 | Weight Decay | No benefit for head-only training; WD = 0 |
| 3 | Optimiser Comparison | Adam outperforms AdamW and SGD (momentum 0.9) |
| 4 | Unfreezing Strategy | Top 40% (B12) achieves best macro F1 |

This structure ensures that observed performance differences are **causally attributable** to the variable under test — not confounded by inherited optimisation state.

---

## Results

### Quantitative Performance — Best Model (B12, Top 40% Unfrozen)

| Class | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| Healthy | 0.9961 | 0.9915 | 0.9938 | 1,529 |
| Pneumonia | 0.9707 | 0.9851 | 0.9779 | 202 |
| COVID | 0.9853 | 0.9926 | 0.9890 | 542 |
| **Macro avg** | — | — | **0.9869** | 2,273 |

### Unfreezing Strategy Comparison

| Exp ID | Top % Unfrozen | Trainable Params | Macro F1 |
|---|---|---|---|
| B10 | 0% (frozen) | 6,147 | 0.8890 |
| B11 | 20% | 8,408,507 | 0.9816 |
| **B12** | **40%** | **12,712,443** | **0.9869** |
| B13 | 100% | 20,867,627 | 0.9865 |

B12 was selected as the best model: higher macro F1 than all alternatives, with stronger minority-class recall — the primary clinical objective. B13 (full fine-tuning) showed marginal degradation, likely due to catastrophic forgetting of lower backbone layers.

### Key Findings

- **Transfer learning converges fast:** Even with a frozen backbone and 4 training epochs, the model reaches 70–90% accuracy. Fine-tuning is necessary for clinical-grade performance, but the efficiency of convergence is notable.
- **Recall drives decisions:** COVID recall of 0.9926 means fewer than 1 in 100 COVID cases are missed on the test set.
- **Class weighting resolves imbalance:** No significant performance gap between majority and minority classes in the final model.
- **Lightweight head generalises better:** A single Dense layer outperformed deeper configurations in preliminary runs. Backbone capacity is sufficient; head complexity is not the bottleneck.
- **Calibration is imperfect:** Calibration curves show slight overconfidence in disease classes — a residual artefact of class weighting. Probability outputs should be post-processed before clinical threshold setting.

### Grad-CAM Localisation

Grad-CAM was applied to the final convolutional layer to generate spatial activation maps. In the best model, activation consistently localises to **pathologically relevant lung regions** — consolidation zones, ground-glass opacities — rather than background artefacts or image borders.

This suggests the model is learning clinically meaningful features, not dataset biases. It does not, however, constitute clinical validation.

> 📂 See [`/figures/gradcam/`](figures/gradcam/) for examples across all three classes.

---

## Visual Outputs

| Output | Location | Description |
|---|---|---|
| Preprocessing pipeline diagram | `figures/preprocessing/pipeline.png` | Annotated flowchart |
| Architecture diagram | `figures/architecture/model_architecture.png` | Two-phase transfer learning |
| Confusion matrix (test set) | `figures/results/confusion_matrix.png` | Normalised, all 3 classes |
| Class-wise F1 scores | `figures/results/classwise_f1.png` | Bar chart across experiments |
| Macro F1 by experiment | `figures/results/macro_f1_by_experiment.png` | B0→B13 comparison |
| Calibration curves | `figures/results/calibration_*.png` | Pneumonia + COVID |
| Grad-CAM overlays | `figures/gradcam/*.png` | One example per class |

---

## Repository Structure

```
lung-disease-classifier/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore
├── lung_disease_classifier.ipynb  # Full pipeline, interactive
│
├── src/
│   ├── preprocessing.py               # Loading, augmentation, dataset splitting
│   ├── model.py                       # Xception + custom head definition
│   ├── train.py                       # Two-phase training, callbacks
│   ├── evaluate.py                    # All metrics: F1, PR, ROC, calibration
│   └── gradcam.py                     # Grad-CAM implementation
│
├── experiments/
│   ├── exp_B0_baseline/
│   │   ├── history.json
│   │   └── results.json
│   ├── exp_B12_unfreezing_test/       # BEST MODEL
│   │   ├── README.md
│   │   ├── history.csv
│   │   └── results.json
│   └── [B1-B13 logs, no .keras]      # Full experimental record
│
├── figures/
│   ├── architecture/
│   ├── preprocessing/
│   ├── results/
│   └── gradcam/
│
└── Lung Data/
│   ├── COVID/
│   ├── HEALTHY/
│   ├── PNEUMONIA/
|   └── README.md                      # Dataset source + setup instructions
```

---

## How to Run

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/lung-disease-classifier.git
cd lung-disease-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare data

Organise the dataset as:

```
data/Lung Data/
    ├── HEALTHY/
    ├── PNEUMONIA/
    └── COVID/
```

### 4. Train

```bash
# Full two-phase training with 40% backbone unfreezing (best config)
python src/train.py --experiment B12 --unfreeze_pct 40
```

### 5. Evaluate

```bash
python src/evaluate.py --model_path experiments/exp_B12_unfreezing_test/model.keras
```

### 6. Generate Grad-CAM

```bash
python src/gradcam.py \
  --model_path experiments/exp_B12_unfreezing_test/model.keras \
  --image_path data/sample.jpg \
  --class_idx 2
```

Or run everything interactively in `notebooks/lung_disease_classifier.ipynb`.

---

## Limitations

These constrained every design decision — they are acknowledged, not hidden:

- **Dataset scale (15K images):** Sufficient for controlled experimentation. Not sufficient for deployment. Real-world screening systems require multi-site data, regulatory validation, and prospective evaluation.
- **Single-source data:** Performance on different scanner types, patient demographics, or disease severity distributions is unknown. External validation is required.
- **Calibration gap:** Overconfidence in disease probabilities observed in calibration curves. Temperature scaling or isotonic regression should be applied before any threshold-based clinical decision.
- **Grad-CAM resolution:** Grad-CAM produces coarse spatial heatmaps, not pixel-precise segmentation masks. It is an interpretability tool, not a structured detection system.
- **Sequential search:** Stage-by-stage grid search is reproducible but not globally optimal. Bayesian optimisation would explore the joint space more efficiently.
- **Augmentation conservatism:** Minimal augmentation preserves clinical features but may limit robustness to real-world imaging variation.

---

## Future Work

Tractable extensions, not wishlist items:

- **Bayesian hyperparameter optimisation** (Optuna) to replace sequential grid search
- **Temperature scaling** to correct calibration overconfidence
- **Monte Carlo Dropout** for uncertainty quantification — knowing when the model is uncertain is clinically as valuable as a confident prediction
- **Segmentation head** — move from Grad-CAM heatmaps to structured lesion proposals using a U-Net decoder on the Xception encoder
- **External validation** on CheXpert or NIH ChestX-ray14 for cross-dataset robustness
- **Pathology-aware augmentation** — systematically evaluate whether elastic deformations (preserving opacity texture) improve minority-class recall

---

## Technical Stack

| Component | Tool |
|---|---|
| Framework | TensorFlow / Keras |
| Backbone | Xception (ImageNet pretrained) |
| Interpretability | Grad-CAM (Selvaraju et al., 2017) |
| Optimiser | Adam / AdamW / SGD (evaluated) |
| Visualisation | Matplotlib, Seaborn |
| Environment | Google Colab (T4 GPU) |

---

## References

1. Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. *CVPR 2017*.
2. Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*.
3. Keras Grad-CAM Tutorial: https://keras.io/examples/vision/grad_cam/

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:16213e,50:1a1a2e,100:0a0a0a&height=100&section=footer" width="100%"/>

</div>
