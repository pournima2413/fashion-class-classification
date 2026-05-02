<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        👗  FASHION CLASS CLASSIFICATION                       ║
║            using Deep Learning & CNN                          ║
║                                                               ║
║        Can a machine learn fashion? Yes — 91.9% of the time. ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

![Accuracy](https://img.shields.io/badge/Test%20Accuracy-91.9%25-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Fashion--MNIST-blueviolet?style=flat-square)
![Images](https://img.shields.io/badge/Images-70%2C000-blue?style=flat-square)
![Classes](https://img.shields.io/badge/Classes-10-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-14B8A6?style=flat-square)

</div>

---

## ◈ The Idea

> *Retailers receive millions of product images daily. Manually tagging each one is slow, expensive, and error-prone.*
> *What if a machine could look at a clothing image and instantly know — T-shirt? Sneaker? Bag? Dress?*

This project builds exactly that — a **Convolutional Neural Network** trained on **70,000 fashion images** that classifies clothing into 10 categories with **91.9% accuracy.**

The real-world applications:
- 🛍️ E-commerce auto-cataloguing
- 📸 Instagram outfit recognition
- 🎯 Personalised product recommendations
- 📦 Automated inventory tagging

---

## ◈ Dataset — Fashion-MNIST

Developed by **Zalando Research** as a harder, more meaningful alternative to digit MNIST.

<div align="center">

| Property | Value |
|:---:|:---:|
| Total Images | 70,000 |
| Training Set | 60,000 |
| Test Set | 10,000 |
| Image Size | 28 × 28 pixels |
| Color | Grayscale |
| Classes | 10 |

</div>

**The 10 Classes:**

```
  [0] 👕 T-shirt/Top       [1] 👖 Trouser
  [2] 🧥 Pullover          [3] 👗 Dress
  [4] 🧥 Coat              [5] 👡 Sandal
  [6] 👔 Shirt             [7] 👟 Sneaker
  [8] 👜 Bag               [9] 👢 Ankle Boot
```

> ⚠️ Dataset not included due to GitHub's 25MB limit.
> Download from 🔗 [Kaggle — Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

---

## ◈ Why CNN and Not a Simple Neural Network?

Traditional neural networks flatten images into a 1D vector —
losing all spatial information. A shirt's sleeve and collar
mean nothing when pixels are treated as independent numbers.

**CNNs preserve spatial structure:**

```
Raw Image (28×28)
      │
      ▼
┌─────────────────────────────────────────────┐
│  Convolutional Layer — detects edges, curves │
│  "Is there a collar here? A sole there?"     │
└─────────────────────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────┐
│  Pooling Layer — compresses, keeps key info  │
│  "Keep what matters, discard the noise"      │
└─────────────────────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────┐
│  Deeper Conv Layers — learns complex shapes  │
│  "That's a sneaker shape. That's a bag."     │
└─────────────────────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────┐
│  Dense + Softmax — final classification      │
│  "91.9% sure this is a Sneaker."             │
└─────────────────────────────────────────────┘
```

---

## ◈ Data Preprocessing

| Step | What Was Done | Why |
|---|---|---|
| **Load** | Read CSV files using Pandas | Fashion-MNIST provided as flattened pixel arrays |
| **Normalize** | Scaled pixel values 0–255 → 0–1 | Faster convergence during training |
| **Reshape** | (784,) → (28, 28, 1) | Restore 2D structure for CNN input |
| **Split** | 80% train / 20% validation | Monitor overfitting during training |

---

## ◈ Model Architecture

```
INPUT: (28, 28, 1) grayscale image
        │
        ▼
┌─────────────────────────┐
│  Conv2D — 32 filters    │  → Detects low-level patterns (edges, lines)
│  Kernel: 3×3 | ReLU     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  MaxPooling2D — 2×2     │  → Reduces size, keeps strongest features
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Conv2D — 64 filters    │  → Detects high-level patterns (shapes, textures)
│  Kernel: 3×3 | ReLU     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  MaxPooling2D — 2×2     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Dropout (0.3)          │  → Randomly disables 30% of neurons → prevents overfitting
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Flatten                │  → 2D feature map → 1D vector
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Dense — 32 neurons     │  → Combines all learned features
│  ReLU                   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Dense — 10 neurons     │  → One output per class
│  Softmax                │  → Converts to probabilities
└─────────────────────────┘

Loss: Categorical Crossentropy  |  Optimizer: Adam  |  Metric: Accuracy
```

---

## ◈ Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch Size | 512 |
| Validation Split | 20% |
| Framework | TensorFlow / Keras |
| Regularization | Dropout (rate = 0.3) |

---

## ◈ Results

<div align="center">

| Configuration | Train Accuracy | Test Accuracy |
|:---:|:---:|:---:|
| 32 filters, no dropout | 95.0% | 91.1% |
| 64 filters, no dropout | 96.0% | 91.6% |
| 64 filters + dropout | 94.0% | **91.9% ✅** |

</div>

> **Best model:** 64 filters + Dropout — slightly lower training accuracy but strongest generalisation on unseen data. Dropout was the key.

---

## ◈ What the Model Gets Right (and Wrong)

```
EASY TO CLASSIFY ✅              HARDER TO CLASSIFY ⚠️
──────────────────               ──────────────────────
  Trouser   → Very distinct       Shirt   → Confused with T-shirt
  Sandal    → Unique shape        Shirt   → Confused with Pullover
  Bag       → Clear silhouette    Coat    → Similar to Pullover
  Sneaker   → Strong features
  Ankle Boot
```

The confusion between Shirt, T-shirt, and Pullover makes intuitive sense —
humans sometimes struggle with the same distinction in grayscale photos.

---

## ◈ Visualisations

| Visual | What It Shows |
|---|---|
| `sample_grid.png` | Random sample images from each of the 10 classes |
| `class_distribution.png` | Bar chart — perfectly balanced dataset (7,000 per class) |
| `accuracy_curve.png` | Training vs validation accuracy across 50 epochs |
| `loss_curve.png` | Training vs validation loss — model is learning, not memorising |
| `confusion_matrix.png` | Heatmap — bright diagonal = correct, off-diagonal = mistakes |

> 📁 All images stored in `/images` folder

---

## ◈ Dropout — The Regularisation Trick

Without Dropout, the model memorises training images instead of learning patterns.
With Dropout (rate = 0.3), 30% of neurons are randomly switched off each batch.

```
Without Dropout:   Train 96% → Test 91.6%   (gap = 4.4%)
With Dropout:      Train 94% → Test 91.9%   (gap = 2.1%)
```

Smaller gap = better generalisation = model works on real-world unseen images.

---

## ◈ Future Enhancements

- [ ] Replace Fashion-MNIST with **DeepFashion** (800K real-world color images)
- [ ] Apply **Transfer Learning** — ResNet50, VGG16, MobileNetV2
- [ ] Deploy as a **Streamlit or Gradio web app**
- [ ] Add **Grad-CAM** visualisation — show which pixels influenced each prediction
- [ ] Extend to **RGB channels** for color-aware classification

---

## ◈ Skills Demonstrated

```txt
✅  CNN architecture design from scratch (Conv → Pool → Dense → Softmax)
✅  Image preprocessing — normalisation, reshaping, train/val split
✅  Dropout regularisation to improve generalisation
✅  Model evaluation — confusion matrix, F1-score, precision, recall
✅  Visualisation of training curves to diagnose overfitting
✅  Iterative model improvement — 3 configurations tested and compared
✅  Real-world business framing of a computer vision problem
```

---

## ◈ How to Run

```bash
# Clone the repository
git clone https://github.com/pournima2413/fashion-class-classification
cd fashion-class-classification

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter

# Download dataset from Kaggle and place in /data folder
# https://www.kaggle.com/datasets/zalando-research/fashionmnist

# Launch the notebook
jupyter notebook Fashion_Class_Classification.ipynb
```

---

## ◈ Project Structure

```
fashion-class-classification/
│
├── Fashion_Class_Classification.ipynb   ← Full analysis and model notebook
│
├── data/
│   ├── fashion-mnist_train.csv          ← Training set (download from Kaggle)
│   └── fashion-mnist_test.csv           ← Test set (download from Kaggle)
│
├── images/
│   ├── sample_grid.png
│   ├── class_distribution.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── confusion_matrix.png
│
└── README.md
```

---

<div align="center">

**Pournima Kamble** — MS Computer Science @ Cleveland State University (2026)
Seeking Data Analyst & Data Engineer roles · Available June 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/pournimakamble)
[![GitHub](https://img.shields.io/badge/GitHub-pournima2413-333?style=flat-square&logo=github&logoColor=white)](https://github.com/pournima2413)
[![Email](https://img.shields.io/badge/Email-pournima2413@gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:pournima2413@gmail.com)

</div>
