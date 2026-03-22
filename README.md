# 🧠 NeuroAge: Brain Age Prediction (EEG ds003775 Edition)

This platform predicts functional brain age from Resting-state EEG spectral power features (25 features across 5 regions).

## 📋 Prerequisites
- **Python 3.9+**
- **pip** (Python package manager)

## 🚀 Getting Started

Follow these steps in your terminal to run the application:

### 1. (Optional) Create a Virtual Environment
It is recommended to run this in a clean environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Required Packages
Install the core ML and web dependencies:
```powershell
pip install -r requirements.txt
```

### 3. Start the Flask Server
Run the application using the standard Python command:
```powershell
python app.py
```

### 4. Access the Website
Once the server starts and displays `* Running on http://localhost:5000`, open your web browser and navigate to:
**[http://localhost:5000](http://localhost:5000)**

---

## ⚡ How to use the Demo
1. **Quick Load**: Scroll to the input section and click on any **Sample Subject** (e.g., `sub-004`). This will instantly populate all 25 EEG spectral power features.
2. **Select Model**: Choose a prediction algorithm (e.g., **Random Forest** or **Ridge Regression**).
3. **Predict**: Click **"Predict Brain Age"**. 
4. **Results**: Review the **Brain Age Gap**, the **10-Model Comparison** table, and your personalized **Neuro-Health Recommendations** based on your brainwave profile!
