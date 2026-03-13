<div align="center">

# 🤖 Machine Learning From Scratch — Zero to Hero

### The Ultimate ML Roadmap: Foundations → Algorithms → Real-World Projects

[![GitHub stars](https://img.shields.io/github/stars/Devanik21/machine-learning-from-scratch-Zero-to-Hero-ML-Roadmap?style=for-the-badge&logo=github&color=yellow)](https://github.com/Devanik21/machine-learning-from-scratch-Zero-to-Hero-ML-Roadmap/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Devanik21/machine-learning-from-scratch-Zero-to-Hero-ML-Roadmap?style=for-the-badge&logo=github&color=blue)](https://github.com/Devanik21/machine-learning-from-scratch-Zero-to-Hero-ML-Roadmap/network)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?style=for-the-badge&logo=jupyter)](https://jupyter.org)

<br/>

> **Master Machine Learning step by step — from your first regression line to neural networks, with hands-on real-world datasets at every stage.**

<br/>

[ Get Started](#-getting-started) • [📚 Curriculum](#-curriculum) • [🏗️ Projects](#️-real-world-projects) • [🛠️ Tech Stack](#️-tech-stack) • [👤 Author](#-author)

</div>

---

## 📖 What Is This?

This repository is a **structured, chapter-by-chapter machine learning curriculum** built entirely in Jupyter Notebooks. Every concept is taught through working code on real datasets — no hand-wavy theory without implementation, no implementation without understanding the math behind it.

Whether you're a complete beginner writing your first `model.fit()` or someone solidifying fundamentals before moving into deep learning, this roadmap covers the full arc:

- **Regression** — from a single line through the data all the way to regularized ensembles
- **Classification** — from logistic regression through SVMs, decision trees, and Naive Bayes
- **Neural Networks** — ANN foundations from scratch
- **End-to-End Projects** — disease prediction, spam detection, sea-level forecasting, rainfall prediction, and more

Every chapter includes at least one guided notebook and one independent project notebook so you can practice what you just learned.

---

## 📚 Curriculum

### 🔵 Part 1 — Regression (Chapters 1–7)

The backbone of supervised learning. You'll build strong geometric and mathematical intuition for how models fit data before layering in complexity.

---

#### `Chapter 1` — Simple Linear Regression

The starting point. One input, one output, one line. Here you build the foundational mental model that everything else extends from.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| House Price Prediction (p1) | `homeprices.csv` | Fitting a regression line, plotting predictions |
| Salary Prediction (p2) | `Salary_Data.csv` | Train/test split, evaluating model accuracy |
| Per Capita Income (p3) | `canada_per_capita_income.xlsx` | Real economic data, trend forecasting |

---

#### `Chapter 2` — Multiple Linear Regression

Extend the model to multiple features. You'll handle real estate data with area, bedrooms, and location simultaneously — and see why the math generalizes cleanly.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| House Price Prediction (p1) | `homeprices.csv` | Multiple feature regression, coefficient interpretation |
| Advanced House Price (p2) | `homes.xls` | Feature engineering, model comparison |

---

#### `Chapter 3` — One Hot Encoding

Categorical variables break linear models unless you encode them properly. This chapter is the bridge between raw data and model-ready features.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Home Price with Destination | `homeprices.csv` | Dummy variable encoding, dropping reference categories |
| Car Prices | `carprices.csv` | Multi-category encoding, price prediction |

---

#### `Chapter 4` — Polynomial Regression

When the relationship between variables is curved, a straight line won't do. You'll transform features and discover that polynomial regression is still linear regression — just in a higher-dimensional space.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Salary by Position (p1) | `Salary_Position.csv` | Degree selection, overfitting intuition |
| Salary by Experience (p2) | `salary.csv` | Comparing linear vs. polynomial fit quality |

---

#### `Chapter 5` — Ridge Regression (L2 Regularization)

Overfitting is the enemy. Ridge regression penalizes large coefficients, shrinking the model toward simplicity without zeroing out any feature entirely.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Height–Weight (p1) | Internal | Bias-variance tradeoff, λ tuning |
| Boston Housing (p2) | `boston_houses.csv` | Cross-validation for regularization strength |

---

#### `Chapter 6` — LASSO Regression (L1 Regularization)

LASSO goes further — it drives irrelevant coefficients all the way to zero. You get a model and built-in feature selection in a single step.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Boston Housing (p1) | `boston_houses.csv` | Sparsity, automatic feature selection |
| Advertising Spend (p2) | `Advertising.csv` | Comparing LASSO vs. Ridge in practice |

---

#### `Chapter 7` — ElasticNet Regression

The best of both worlds: ElasticNet combines L1 and L2 penalties with a mixing parameter, giving you both coefficient shrinkage and automatic feature selection.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Advertising Analysis | `Advertising.csv` | ElasticNet parameter tuning (α, l1_ratio) |
| Diabetes Progression | `diabetes.csv` | Medical data regression, clinical feature analysis |

---

### 🟢 Part 2 — Classification (Chapters 8–13)

The shift from predicting numbers to predicting categories. These chapters cover every major classical classification algorithm with progressively more complex real-world datasets.

---

#### `Chapter 8` — Logistic Regression

Despite the name, this is a classification algorithm — and one of the most important in the field. You'll learn how a sigmoid function converts a linear score into a probability.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Insurance Prediction (p1) | `insurance_data.csv` | Binary classification, decision boundary |
| Employee Attrition (p2) | `HR_comma_sep.csv` | Multi-feature HR analytics, feature importance |
| Handwritten Digits (p3) | MNIST (sklearn) | Multiclass classification, confusion matrix |

---

#### `Chapter 9` — Support Vector Machines (SVM)

SVMs find the maximum-margin hyperplane between classes. You'll explore both linear and kernel-based SVMs and develop geometric intuition for why margins matter.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Iris Flower Classification (p1) | sklearn Iris | Linear SVM, 2D decision boundaries |
| Handwritten Digits (p2) | MNIST (sklearn) | RBF kernel, high-dimensional classification |

---

#### `Chapter 10` — Naive Bayes

Probabilistic classification using Bayes' theorem with the "naive" conditional independence assumption. Surprisingly powerful for text — you'll build a working spam detector.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Can Play Cricket? | `cricket.csv` | Gaussian NB, prior/posterior intuition |
| SMS Spam Classifier | `SMSSpamCollection` | CountVectorizer, full text classification pipeline |
| Message Grouping | Internal | Multinomial NB, document classification |

---

#### `Chapter 11` — K-Nearest Neighbors (KNN)

No training phase, no parameters to estimate — KNN classifies by democratic vote of its closest neighbors. Simple, powerful, and a great model for building intuition.

| Dataset | What You'll Learn |
|---------|-------------------|
| `breast-cancer-wisconsin.data` | Choosing K, distance metrics, medical classification |
| `diabetes.csv` | Feature scaling impact, accuracy vs. K curve |

---

#### `Chapter 12` — Decision Trees

Interpretable machine learning at its finest. Decision trees split data recursively on the most informative features — and you can actually read and explain the resulting model.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Can Play Cricket? | `cricket1.csv` | Gini impurity, entropy, tree visualization |
| Salary Classification | `salaries.csv` | Tree depth control, pruning, avoiding overfitting |

---

#### `Chapter 13` — Random Forest

Wisdom of crowds applied to decision trees. An ensemble of diverse trees votes on the final prediction, dramatically reducing variance and improving generalization over any single tree.

| Notebook | Dataset | What You'll Learn |
|----------|---------|-------------------|
| Can Play Cricket? | Internal | Bagging, feature subsampling, OOB error |
| Salary Prediction | `Salary_Experience.csv` | Feature importance, single tree vs. forest comparison |

---

### 🔴 Part 3 — Neural Networks

#### `ANN` — Artificial Neural Networks

The gateway to deep learning. A full implementation and walkthrough of feedforward neural networks — layers, activations, backpropagation, and training dynamics from the ground up.

| Notebook | What You'll Build |
|----------|------------------|
| `ANN.ipynb` | Perceptron → hidden layers → activation functions → gradient descent → full training loop |

---

## 🏗️ Real-World Projects

The `/Testings` folder and root-level notebooks contain standalone end-to-end projects where you choose the model, preprocess real data, and evaluate results — no hand-holding.

| Project | Dataset | Techniques |
|---------|---------|------------|
| 🏥 Heart Disease Prediction | `heart.csv` / `heartDisease.csv` | Logistic Regression, SVM, classification metrics |
| 🩺 Diabetes Detection | `diabetes.csv` | KNN, Logistic Regression, feature scaling |
| 🦠 Cancer Cell Classification | `breast-cancer-wisconsin.data` | SVM, KNN, binary classification |
| 🌸 Iris Flower Classification | sklearn Iris | Multi-class SVM, Decision Tree |
| 📩 SMS Spam Classifier | SMS dataset | Naive Bayes, TF-IDF, NLP pipeline |
| ⚖️ BMI Calculator & Predictor | `bmi.csv` | Regression, threshold-based classification |
| 📉 Sea Level Predictor | `epa-sea-level.csv` | Linear Regression, trend extrapolation |
| 🌧️ Rainfall Prediction | `rainfallPred.csv` | Regression, weather feature analysis |
| 🏏 Can Play Cricket? | `cricket.csv` | Decision Tree, Naive Bayes, Random Forest |
| 💰 Salary Classification | `salaries.csv` | Decision Tree, SVM comparison |
| 📊 Data Analysis Projects | Internal | EDA, visualization, pandas, matplotlib |

---

## 🛠️ Tech Stack

```
Python 3.8+
├── Data & Math       → NumPy, Pandas
├── Visualization     → Matplotlib, Seaborn
├── Machine Learning  → Scikit-Learn
├── Deep Learning     → ANN (from scratch)
└── Notebooks         → Jupyter Lab / Jupyter Notebook
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/Devanik21/machine-learning-from-scratch-Zero-to-Hero-ML-Roadmap.git
cd machine-learning-from-scratch-Zero-to-Hero-ML-Roadmap
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter openpyxl
```

**3. Launch Jupyter**
```bash
jupyter notebook
```

**4. Follow the roadmap** — start with `1. simple linear Regression` and work chapter by chapter. Each folder contains everything you need: the notebook, the dataset, and multiple practice variations.

---

## 🗺️ Visual Roadmap

```
📦 Start Here
│
├── 📈 REGRESSION  ────────────────────────────── Chapters 1–7
│   ├── 1 → Simple Linear Regression       (House prices, Salary, Per capita income)
│   ├── 2 → Multiple Linear Regression     (Multi-feature real estate)
│   ├── 3 → One Hot Encoding               (Handling categorical data)
│   ├── 4 → Polynomial Regression          (Curved relationships)
│   ├── 5 → Ridge Regression               (L2 regularization)
│   ├── 6 → LASSO Regression               (L1 + feature selection)
│   └── 7 → ElasticNet                     (L1 + L2 combined)
│
├── 🧠 CLASSIFICATION  ─────────────────────────  Chapters 8–13
│   ├── 8  → Logistic Regression           (Binary & multiclass, digits)
│   ├── 9  → SVM                           (Maximum margin, kernel trick)
│   ├── 10 → Naive Bayes                   (Probabilistic + spam detection)
│   ├── 11 → KNN                           (Instance-based, cancer/diabetes)
│   ├── 12 → Decision Trees                (Interpretable rule learning)
│   └── 13 → Random Forest                 (Ensemble, feature importance)
│
├── 🔥 NEURAL NETWORKS  ────────────────────────
│   └── ANN                                (Layers, backprop, training loop)
│
└── 🏗️ REAL-WORLD PROJECTS  ──────────────────── /Testings
    └── Heart, Cancer, Diabetes, Spam, Sea Level, Rainfall, Iris, BMI...
```

---

## 📁 Repository Structure

```
machine-learning-from-scratch/
│
├── 1. simple linear Regression/
├── 2. Multiple Linear Regression/
├── 3. One Hot Encoding/
├── 4. Polynomial Linear Regression/
├── 5. Ridge Regression/
├── 6. LASSO Regression/
├── 7. Elasticnet Regression/
├── 8. LOGISTIC Regression/
├── 9. SVM/
├── 10. NB/                    ← Naive Bayes
├── 11. KNN/
├── 12. Decision Trees/
├── 13. Random Forest/
├── ANN/                       ← Neural Network
├── Testings/                  ← End-to-end projects
├── rain1.ipynb                ← Rainfall prediction
└── rainfallPred.csv
```

---

## 💡 Who Is This For?

- **Beginners** who want a structured path through ML without getting lost in scattered tutorials
- **Students** looking for a practical companion to ML coursework with real datasets for every concept
- **Developers** brushing up on classical ML before moving into deep learning
- **Anyone** who learns best by doing — every chapter is working code, not slides

---

## 🌟 Support This Project

If this roadmap helps you learn, the best way to say thanks is to **⭐ star the repository** — it helps others find it and motivates adding more content.

[![Star this repo](https://img.shields.io/github/stars/Devanik21/machine-learning-from-scratch-Zero-to-Hero-ML-Roadmap?style=social)](https://github.com/Devanik21/machine-learning-from-scratch-Zero-to-Hero-ML-Roadmap)

---

## 👤 Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

<div align="center">

*Built one notebook at a time. Every model trained. Every concept earned.*

**⭐ Star to save your progress · 🍴 Fork to make it yours**

</div>
