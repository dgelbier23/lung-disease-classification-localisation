# Dataset

## Source

The dataset used in this project is the **Lung Disease** chest X-ray dataset, consisting of 15,153 images across three diagnostic categories.

## Organisation

Place the dataset in the following structure relative to the repository root:

```
data/
└── Lung Data/
    ├── HEALTHY/       # 0 = Healthy lungs
    ├── PNEUMONIA/     # 1 = Pneumonia-infected lungs
    └── COVID/         # 2 = COVID-19-infected lungs
```

All modifications (splitting, normalisation, augmentation) are handled programmatically in the preprocessing pipeline. Do not pre-split the dataset manually.

## Class Distribution

| Class | Images (approx.) | Split |
|---|---|---|
| HEALTHY | ~10,500 | Majority |
| PNEUMONIA | ~1,400 | Minority |
| COVID | ~3,600 | Minority |

Class imbalance is handled via **class weighting** during training — see `src/preprocessing.py` and the main README for rationale.

## Notes

- All images are grayscale chest X-rays, converted to RGB during preprocessing for Xception compatibility.
- Images are provided at 299×299 resolution, matching the Xception native input size — no resizing is applied.
- Do not apply any manual preprocessing before feeding data into the pipeline.
