Here’s a clean, well-structured `README.md` version of your MediTrack project, formatted using proper Markdown conventions for GitHub:

---

# 💊 MediTrack: AI-Powered Drug Inventory & Supply Chain Management

**MediTrack** is an intelligent, multi-faceted system designed to revolutionize pharmaceutical logistics by leveraging Artificial Intelligence. This project automates and optimizes critical aspects of the drug supply chain — from demand forecasting to inventory risk management — ensuring that essential medicines are available **when** and **where** they are needed most.

---

## 🚀 Core Features

MediTrack consists of specialized machine learning models, each addressing a vital aspect of pharmaceutical supply chain management:

---

### 📈 1. Demand Forecasting & Supply Chain Optimization

* **Goal:** Predict future pharmaceutical demand to prevent stockouts and reduce waste.
* **Algorithm:** Two-Part Model using an **XGBoost Regressor**.

  * Part 1: Predicts the probability of a shipment.
  * Part 2: Estimates the shipment size.
* **Key Features Used:**

  * Lag features
  * Rolling averages
  * Cyclic time-based variables (e.g., sine/cosine of month/week)

---

### 📉 2. Inventory Risk Scoring & Stockout Prediction

* **Goal:** Classify each drug into **High**, **Medium**, or **Low** risk categories.
* **Algorithm:** **Random Forest Classifier** with **94% accuracy**, and **100% accuracy on high-risk items**.
* **Key Features Used:**

  * Demand volatility
  * Stock coverage (in weeks)
  * Forecast-to-stock ratio

---

### 🕒 3. Expiry Prediction & Shelf-Life Monitoring

* **Goal:** Predict remaining shelf life of drug batches to minimize wastage.
* **Algorithm:** **XGBoost Regressor**
* **Key Features Used:**

  * Initial shelf life
  * Current age
  * Storage conditions (e.g., *Standard*, *Cold-Chain*)

---

### 📦 4. Medicine Pack Verification (Computer Vision)

* **Goal:** Automatically verify identity and integrity of drug packages via images.
* **Algorithm:** Pre-trained **CNN** (e.g., ResNet or EfficientNet)

  * **Multi-Class Classification:** Identify among 150 drug types
  * **Binary Classification:** Detect if package is *Damaged* or *Undamaged*

---

## 🧰 Tech Stack

| Component                   | Technology Used                      |
| --------------------------- | ------------------------------------ |
| **Backend & Modeling**      | Python                               |
| **ML Libraries**            | `scikit-learn`, `XGBoost`, `Prophet` |
| **Data Handling**           | `pandas`, `NumPy`                    |
| **Visualization**           | `Matplotlib`, `Seaborn`              |
| **API Framework (Planned)** | `Flask`                              |

---

## 🛠️ Getting Started

Follow these instructions to set up MediTrack locally.

### ✅ Prerequisites

* Python 3.8 or higher
* Git

---

### 📥 Installation

Clone the repository:

```bash
git clone https://github.com/SohamAmberkar/risk_detection.git
cd risk_detection
```

Create and activate a virtual environment:

**Windows:**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## 📊 How to Run the Models

Run each model in the following sequence:

---

### 1️⃣ Train the Demand Forecasting Model

Trains and saves the optimized `xgboost_classifier.joblib` and `xgboost_regressor.joblib`.

```bash
python advanced_model_trainer.py
```

---

### 2️⃣ Generate Inventory Data for Risk Scoring

Generates `drug_inventory_levels.csv` from raw shipment data.

```bash
python generate_inventory_data.py
```

---

### 3️⃣ Train the Inventory Risk Model

Trains the risk classification model and saves `risk_classifier_model.joblib`.

```bash
python train_risk_model.py
```

---

### 4️⃣ Generate the Risk Model Performance Report

Generates evaluation plots and a report for the trained model.

```bash
python generate_risk_report.py
```

---



