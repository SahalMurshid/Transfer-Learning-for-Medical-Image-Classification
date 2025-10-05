# High-Performance X-Ray Classifier (ResNet50 Transfer Learning)



## 🎯 Project Goal



This project aims to build a robust, high-performance system for the binary classification of **Chest X-ray images** to screen for signs of **Viral Pneumonia**. The primary objective is to maximize **Recall (Sensitivity)** on the positive class to ensure clinical safety by minimizing **False Negatives (FN)**—critical in medical pre-screening applications.



***



## ✨ Key Findings



1. **Class Collapse Resolution:** Initial training failed (0% True Positives). This was solved by implementing a **Two-Phase Fine-Tuning Strategy** (Freezing ResNet50 layers, then unfreezing with an extremely low learning rate).

2. **High Clinical Sensitivity:** The final model achieved **93.1% Recall** on the test set, demonstrating reliability in correctly identifying patients with the condition.

3. **Threshold Optimization:** An optimal decision threshold of **0.700** was calculated and applied, moving away from the default $0.500$ to maximize the model's **Balanced Accuracy (89.3%)** for production deployment.



***



## 📊 Results and Visuals (3 Figures Included)



The final model was evaluated on a dedicated test set, yielding the following performance metrics:



| Metric | Value | Interpretation |

| :--- | :--- | :--- |

| **Overall Accuracy** | 90.2% | High generalization ability on unseen data. |

| **Recall (Positive)** | 93.1% | High rate of finding actual positive cases (low FNs). |

| **Decision Threshold** | 0.700 | Optimized cutoff point for classification. |



### 1. Training History and Performance



These visuals confirm the successful convergence of the model during the two-phase fine-tuning process.



| Figure | Description | Visual |

| :--- | :--- | :--- |

| **Figure 1** | **Training History (Loss & AUC)** | ![Training History Loss and AUC](images/Figure_1_Training_History.png) |

| **Figure 2** | **Validation ROC Curve** | ![Validation ROC Curve](images/Figure_2_ROC_Curve.png) |



***



### 2. Final Confusion Matrix



The confusion matrix shows the breakdown of the 624 images tested, confirming the low number of False Negatives (FN = 27).



| Figure | Description | Visual |

| :--- | :--- | :--- |

| **Figure 3** | **Test Set Confusion Matrix** | ![Test Set Confusion Matrix](images/Figure_3_Confusion_Matrix.png) |



**Overall Accuracy: 90.22%**



***



## ⚙️ Project Structure and Setup



### Technologies Used



* **Python 3.8+**

* **Deep Learning:** `TensorFlow / Keras`

* **Model:** `ResNet50` (Transfer Learning)

* **Data Processing:** `numpy`, `pillow`

* **Metrics:** `sklearn`



### Installation and Execution



1. **Clone the repository:**

   ```bash

   git clone [YOUR-REPO-URL]

   cd Transfer-Learning-for-Medical-Image-Classification



---

**Author:** Sahal Murshid

**Date:** October 2025
