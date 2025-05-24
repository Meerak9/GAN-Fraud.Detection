# GAN Fraud Detection Project

This project explores the use of **Generative Adversarial Networks (GANs)** to address the issue of **imbalanced datasets**, particularly in the context of **credit card fraud detection**.

The goal was to generate synthetic fraud samples using two GAN models:
- **Vanilla GAN**
- **Wasserstein GAN (WGAN)**

These generated samples were then used to balance the dataset and improve the performance of a classification model (Random Forest).

---

## 📁 Contents

| File | Description |
|------|-------------|
| `gan_fraud_detection.ipynb` | Jupyter Notebook containing the full implementation |
| `creditcard.csv` | Original dataset used for training (Credit Card Fraud Detection) |
| `project_report.pdf` | Final report in PDF format (4–6 pages) |
| `README.md` | This file — project description and instructions |
| `requirements.txt` | List of required Python packages |

---

## 🔍 Problem Statement

In many real-life applications, datasets are highly imbalanced — meaning one class is overrepresented while the other is rare. This leads to poor performance on the minority class, which is often the most important one to detect (e.g., fraud cases).

This project addresses this issue by generating synthetic fraud data using GANs to balance the dataset and improve classification accuracy.

---

## 🧪 Dataset Description

- **Dataset**: Credit Card Fraud Detection (from Kaggle)
- **Total Transactions**: 284,807
- **Fraud Cases (Class = 1)**: 492 (0.17%)
- **Non-Fraud Cases (Class = 0)**: 284,315 (99.83%)

Clearly, the dataset is **highly imbalanced**, making it difficult for machine learning models to detect fraud effectively.

---

## 🧠 Implemented Models

### 1. Vanilla GAN
- Generator + Discriminator architecture
- Trained for 100 epochs
- Generated 500 synthetic fraud samples

### 2. Wasserstein GAN (WGAN)
- Uses a Critic instead of Discriminator
- More stable training with weight clipping
- Trained for 200 epochs
- Generated 500 synthetic fraud samples

---

## 📊 Classification Results

| Metric      | Original | Vanilla GAN | WGAN     |
|-------------|----------|-------------|----------|
| Precision   | 0.946    | 0.9995      | 0.9781   |
| Recall      | 0.833    | 0.9976      | 0.9436   |
| F1-Score    | 0.886    | 0.9985      | 0.9605   |
| ROC-AUC     | 0.963    | 0.9993      | 0.9781   |

Results show that both GAN models improved classification performance significantly, with **WGAN** offering better stability and realistic sample generation.

---


## 🛠 Requirements

To run the code, install the following dependencies:

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn

---

## 📌 Notes

All code was implemented in a Google Colab environment using PyTorch for GAN training and scikit-learn for classification. The generated synthetic data was added to the original dataset to create balanced versions, which significantly improved the classifier's ability to detect fraud cases.

This project demonstrates how GANs can be used not only for image generation but also for improving machine learning models by generating realistic synthetic samples for underrepresented classes.

