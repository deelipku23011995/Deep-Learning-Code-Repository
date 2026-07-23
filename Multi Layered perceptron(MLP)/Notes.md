# 🧠 MNIST Deep Learning Optimization Experiments

An extensive deep learning benchmarking project using **Keras / TensorFlow** on the classic **MNIST Handwritten Digit Dataset**. This repository explores the impact of different network architectures, weight initializations, regularization methods, and optimization techniques on model training performance and accuracy.

---

## 📌 Project Overview

The main objective of this project is to analyze how varying hyperparameter combinations—specifically **depth (number of layers)**, **weight initialization**, **regularization**, and **optimization algorithms**—influence convergence speed and classification performance.

---

## 🛠️ Experimentation & Hyperparameters

We evaluate multiple combinations of hyperparameter configurations across varying network depths:

### 1. Network Architectures
* **Variable Layer Depths:** Shallow to deep fully connected architectures (e.g., 2-layer, 3-layer, 5-layer+ feedforward neural networks).

### 2. Weight Initialization Techniques
* **He Normal (`he_normal`):** Optimized for layers with ReLU activation functions.
* **Xavier / Glorot (`glorot_uniform` / `glorot_normal`):** Optimized for layers with Sigmoid or Tanh activations.

### 3. Regularization & Normalization
* **Dropout:** Prevents overfitting by randomly deactivating neurons during training.
* **Batch Normalization:** Stabilizes training by normalizing layer inputs, allowing higher learning rates and faster convergence.

### 4. Optimization Algorithms
* **AdaGrad:** Adapts learning rates based on historical gradients (ideal for sparse data).
* **AdaDelta:** An extension of AdaGrad that restricts window sizes to prevent decaying learning rates.
* **Adam (Adaptive Moment Estimation):** Combines Momentum and RMSProp for robust gradient descent.

---

## 📁 Dataset

The **MNIST** dataset consists of $70,000$ grayscale images of handwritten digits ($0$ through $9$) at a resolution of $28 \times 28$ pixels:
* **Training set:** 60,000 images
* **Test set:** 10,000 images

---

## ⚙️ Requirements & Installation

To run the notebook locally, clone the repository and install the dependencies:

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Install required packages
pip install tensorflow numpy matplotlib scikit-learn jupyter
