# **Anomaly Detection on Capsule Dataset**

**Overview :**
This project focuses on anomaly detection using the Capsule Dataset from MVTec. The trained model identifies and detects anomalies such as:

* Crack

* Scratch

* Squeeze

* Poke

* Faulty Imprint

The model is trained using Anomalib, a deep learning framework designed for industrial anomaly detection.

**Dataset :** The MVTec AD Capsule Dataset contains both normal and defective images of capsules. The dataset is publicly available and widely used in anomaly detection research https://www.mvtec.com/company/research/datasets/mvtec-ad

**Model & Training Details :**

Framework: Anomalib

Architecture: Padim

**Training :**

* Preprocessed images to ensure consistency

* Trained on normal samples

* Evaluated on defective images

**Evaluation Metrics :**

* AUC (Area Under Curve)

* Precision & Recall
