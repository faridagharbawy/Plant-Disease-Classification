# 🌿 PlantVillage Leaf Disease Classification
### CAI3105 — Deep Learning | 12th Week Project
**Prof. Nashwa El-Bendary | Smart Village Campus | May 2026**

---

## 👥 Team Members

| # | Name | Student ID |
|---|------|-----------|
| 01 | Farida Gamal | 231027763 |
| 02 | Mahinour Tamer | 231017705 |

---

## 📋 Project Overview

A comparative study between two deep learning paradigms for automated plant leaf disease classification using the **PlantVillage dataset (54,306 images, 38 classes)**:

| Approach | Method | Accuracy | Train Time |
|---|---|---|---|
| **Approach 1** | ResNet50 (frozen) → SVM | **98%** | ~8.4 min |
| **Approach 2** | ResNet50 End-to-End DL | **98%** | ~42.3 min |
| Bonus 1 | MobileNet + SVM | 96% | ~6.9 min |
| Bonus 3 | ResNet50 on Grayscale | 89% | ~40.1 min |

---

## 📁 Project Structure

```
project/
│
├── plant.ipynb   ← Main Jupyter notebook (run this)
├── dashboard.html                        ← Interactive results dashboard
├── README.md                             ← This file
│
└── results/                             ← Auto-generated after running notebook
    ├── dashboard_data.json
    ├── fig1_class_distribution.png
    ├── fig2_sample_images.png
    ├── fig3_cm_approach1_svm.png
    ├── fig4_cm_approach2_e2e.png
    ├── fig5_learning_curves_resnet50.png
    ├── fig6_learning_curves_mobilenet.png
    ├── fig7_comparative_analysis.png
    ├── hyperparameter_table.csv
    └── results_summary.csv
```

---

## 🗂️ Dataset

**PlantVillage Dataset** — Color variant

- **Total images:** 54,306
- **Classes:** 38 disease categories across 14 plant species
- **Split:** 70% Train (38,014) / 15% Val (8,146) / 15% Test (8,146)
- **Download:** [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

Expected folder structure after download:
```
plantvillage dataset/
    ├── color/          ← used as main dataset
    ├── grayscale/      ← used for bonus comparison
    └── segmented/      
```

---

## ⚙️ Requirements

- Python **3.9 – 3.11** (Python 3.12 not supported by TensorFlow)
- At least **8 GB RAM** (16 GB recommended)
- GPU optional but recommended for faster training

### Install dependencies

```bash
pip install tensorflow==2.13.0
pip install scikit-learn matplotlib seaborn pandas pillow tqdm jupyter
```

---

## 🚀 How to Run

### Step 1 — Set dataset paths

Open `plant.ipynb` and update these 3 lines at the top:

```python
DATA_DIR_COLOR = r'C:\path\to\plantvillage dataset\color'
DATA_DIR_GRAY  = r'C:\path\to\plantvillage dataset\grayscale'
RESULTS_DIR    = r'C:\path\to\results'
```

> **Mac/Linux:** use forward slashes — `'/home/user/plantvillage/color'`

### Step 2 — Run the notebook

```bash
jupyter notebook plant.ipynb
```

Then: **Kernel → Restart & Run All**

⏱️ **Expected runtime:** 1–2 hours on CPU | ~20 minutes with GPU

### Step 3 — View the dashboard

1. Copy `dashboard_data.json` from your `RESULTS_DIR` folder
2. Paste it in the **same folder** as `dashboard.html`
3. Open `dashboard.html` in Chrome

> No server or internet required — the dashboard works offline.

---

## 📊 Dashboard Features

The interactive dashboard (`dashboard.html`) contains **7 tabs:**

| Tab | Content |
|---|---|
| 📊 Overview | Summary stats, accuracy & time comparison charts |
| 📁 Dataset | Metadata, N=54,306, split breakdown, augmentation, all 38 classes |
| 🧠 Models | ResNet50 architecture (He et al., 2016), hyperparameter table |
| 🔬 Results | Both approaches with metrics, animated bars, confusion matrices |
| 📈 Analysis | Grouped bar chart, learning curves, radar chart, bubble chart |
| 🎁 Bonus | MobileNet comparison, Color vs Grayscale analysis |
| 🔍 Live Demo | Upload leaf image → predicted disease class + confidence |

---

## 🧠 Model Architecture

### ResNet50 — Primary Model
> He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. DOI: 10.1109/CVPR.2016.90

- **Depth:** 50 layers with residual (skip) connections
- **Parameters:** 25.6 Million
- **Feature vector:** 2048-D (GlobalAveragePooling)
- **Pre-training:** ImageNet

### MobileNetV1 — Bonus Model
> Howard, A.G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861

- **Parameters:** 4.2 Million (~6x fewer than ResNet50)
- **FLOPs:** ~569M (~9x fewer than ResNet50)
- **Best for:** Edge / mobile deployment

---

## 🔑 Key Findings

1. **Both approaches achieve 98% accuracy** — frozen ImageNet features are highly transferable to leaf disease classification
2. **Approach 1 is ~5x faster** — SVM training on extracted features is dramatically quicker than end-to-end DL training
3. **Color is critical** — grayscale accuracy drops ~9% since many diseases show chromatic symptoms (yellowing, browning)
4. **MobileNet is best for edge deployment** — 96% accuracy with 9x fewer computations

---

## 📂 Output Files

After running the notebook, the `results/` folder will contain:

| File | Description |
|---|---|
| `dashboard_data.json` | All results serialized for the dashboard |
| `fig1_class_distribution.png` | Class frequency histogram |
| `fig2_sample_images.png` | Sample images grid |
| `fig3_cm_approach1_svm.png` | 38×38 confusion matrix — ResNet50+SVM |
| `fig4_cm_approach2_e2e.png` | 38×38 confusion matrix — ResNet50 E2E |
| `fig5_learning_curves_resnet50.png` | Accuracy & loss curves — ResNet50 |
| `fig6_learning_curves_mobilenet.png` | Accuracy & loss curves — MobileNet |
| `fig7_comparative_analysis.png` | All approaches comparison chart |
| `hyperparameter_table.csv` | Complete hyperparameter reference |
| `results_summary.csv` | All metrics for all approaches |

---

## 🐛 Common Issues

| Problem | Solution |
|---|---|
| `SyntaxError` on imports | Make sure `from tensorflow.keras.callbacks import ...` is on **one line** |
| `No such file or directory` | Check that `DATA_DIR_COLOR` path is correct and dataset is downloaded |
| `pip install` fails | Run terminal as Administrator, or try `pip install tensorflow` (no version pin) |
| Dashboard shows wrong results | Make sure `dashboard_data.json` is in the same folder as `dashboard.html` |
| Out of memory error | Reduce `BATCH_SIZE` from 32 to 16 in the notebook config cell |

---


<div align="center">
  <sub>CAI3105 Deep Learning · Smart Village Campus · May 2026</sub>
</div>
