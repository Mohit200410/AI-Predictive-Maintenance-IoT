# ⚙️ AI-Powered Predictive Maintenance System for IoT Devices

---

## 📌 Overview

This project is an AI-based Predictive Maintenance System designed to monitor machine health and predict potential failures using sensor data.

It uses the NASA CMAPSS dataset to simulate real-world industrial IoT environments such as manufacturing plants, power plants, and aviation systems.

The system analyzes 24 sensor readings and predicts whether a machine is at risk of failure, helping industries reduce downtime and maintenance costs.

---

## 🚀 Key Features

* 🤖 Machine failure prediction using Machine Learning
* 🧠 Random Forest Classification model (~96% accuracy)
* 📊 Confusion Matrix for performance evaluation
* 🔥 Feature Importance for model interpretability
* 🏭 Real-time Streamlit dashboard
* 📡 Sensor-based simulation (24 IoT signals)
* ⚠️ Failure risk probability detection

---

## 🔄 System Workflow

```
Sensor Data → Preprocessing → Feature Engineering → Model Training → Prediction → Visualization → Dashboard
```

---

## 🧠 Tech Stack

| Category         | Tools Used          |
| ---------------- | ------------------- |
| Programming      | Python              |
| Data Handling    | Pandas, NumPy       |
| Machine Learning | Scikit-learn        |
| Visualization    | Matplotlib, Seaborn |
| Dashboard        | Streamlit           |
| Model Storage    | Joblib              |
| Version Control  | Git & GitHub        |

---

## 📊 Model Output

### ✔ Performance

* Accuracy: ~96%
* Precision, Recall, F1-score

---

## 📸 Screenshots

### 🔥 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

### 📊 Feature Importance

![Feature Importance](images/feature_importance.png)

---

### 🏭 Dashboard Preview

![Dashboard](images/dashboard_main.png)

---

## 🖥️ How to Run

### 1️⃣ Train Model

```bash
python main.py
```

### 2️⃣ Run Dashboard

```bash
streamlit run dashboard/app.py
```

### 3️⃣ Open in Browser

```
http://localhost:8501
```

---

## 📁 Project Structure

```
AI-Predictive-Maintenance-IoT/
│
├── data/
│   └── raw/
│       └── train_FD001.txt
│
├── models/
│   └── model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── images/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── dashboard_main.png
│
├── dashboard/
│   └── app.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🎯 Conclusion

This project demonstrates how AI can be applied in Predictive Maintenance systems to detect failures early and improve operational efficiency.

It simulates real-world industrial IoT monitoring systems and provides hands-on experience in machine learning, data analysis, and dashboard development.

---

## 👩‍💻 Author

**Yusra Sheikh**

---

⭐ If you like this project, consider giving it a star!
