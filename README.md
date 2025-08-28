# 📊 Churn Prediction & Survival Analysis Dashboard  

## 📌 Project Overview  
This project focuses on predicting customer churn using **classification models** and estimating customer retention over time with **survival analysis models**.  
Both models are deployed in a **Flask-based interactive dashboard** where users can upload customer data and view predictions, insights, and recommendations.  

The dashboard consists of four main tabs:  
1. **Overview** – Business insights, KPIs, and action areas.  
2. **Classification** – Churn predictions, churn drivers, and feature importance.  
3. **Survival** – Survival curves, cumulative hazard plots, and survival metrics.  
4. **Recommendations** – Customer-level churn & survival predictions with tailored recommendations.  

---

## 🚀 Features  
- Upload customer data in **CSV format**  
- Get **real-time predictions** from pre-trained models  
- Visualize churn drivers and survival metrics  
- Generate **customer-specific recommendations**  
- Scalable architecture, ready for cloud deployment  

---

## 🛠️ Tech Stack  
- **Programming Language**: Python  
- **Frameworks & Libraries**:  
  - Machine Learning: `scikit-learn`, `lifelines`  
  - Data Processing: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`, `plotly`  
  - Deployment: `Flask`  
- **Model Storage**: `joblib`  
- **Future Hosting Options**: Docker, Heroku, AWS, Azure  

---

## 📂 Repository Structure  
- app.py # Flask web application
- models/ # Saved classification & survival models
- static/ # CSS, JS, and images for the dashboard
- templates/ # HTML templates for Flask
-  data/ # Sample data (if included)
-   notebooks/ # Jupyter notebooks (EDA, training)
- quirements.txt # Python dependencies
-  README.md # Project documentation


---

## ⚙️ Installation & Usage  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
### 2️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Flask app
```bash
python app.py
```
### 5️⃣ Open in browser
http://127.0.0.1:5000

