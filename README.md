# 🧬 MedAI India — AI-Powered Disease Prediction System

<div align="center">

![MedAI India](https://img.shields.io/badge/MedAI-India-blue?style=for-the-badge&logo=activity&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1.3-000000?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

**An intelligent healthcare platform combining symptom-based ML prediction and AI-powered skin disease image classification.**

🔗 **Live Demo:** [https://medais.onrender.com](https://medais.onrender.com)

</div>

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🩺 **Symptom Analysis** | Select symptoms from 130+ options and get instant AI disease predictions |
| 🖼️ **Image Disease Detection** | Upload a skin image and classify it using a fine-tuned EfficientNetB0 deep learning model |
| 🔐 **Secure Authentication** | User registration & login with bcrypt-hashed passwords and strong validation rules |
| 📊 **Prediction History** | View all past predictions (symptom-based and image-based) tied to your account |
| 📱 **Responsive Design** | Fully mobile-friendly UI with modern dark-mode aesthetics |

---

## 🚀 Tech Stack

### Backend
- **Flask** — Web framework
- **Flask-Login** — Session-based authentication
- **Werkzeug** — Password hashing
- **Gunicorn** — Production WSGI server

### Machine Learning
- **Scikit-learn** — Random Forest classifier for symptom-based disease prediction
- **TensorFlow / Keras** — EfficientNetB0 CNN for skin disease image classification
- **Pandas / NumPy** — Data processing

### Database
- **MongoDB Atlas** — Cloud NoSQL database for users and prediction history

### Deployment
- **Render** — Cloud hosting platform
- **Docker** — Containerized environment

---

## 🧠 AI Models

### 1. Symptom-Based Disease Predictor (`disease_model.pkl`)
- **Algorithm:** Random Forest Classifier
- **Input:** 130+ binary symptom features
- **Output:** Predicted disease name + confidence probability
- **Dataset:** 41 disease classes

### 2. Skin Disease Image Classifier (`final_disease_model.h5`)
- **Architecture:** EfficientNetB0 (transfer learning from ImageNet)
- **Training:** Two-phase — feature extraction then fine-tuning
- **Input:** 224×224 RGB images
- **Classes:** Acne, Candidiasis, Infestations/Bites, Psoriasis, Skin Cancer, Warts
- **Preprocessing:** EfficientNet-specific normalization

---

## 🔒 Security & Validation

- **Client-side:** Real-time username and password strength checker with visual feedback
- **Server-side:** Regex-based validation in Flask before any DB operation
- **Password rules:** Min 8 chars, uppercase, lowercase, number, and special character required
- **Username rules:** 4–15 chars, must start with a letter, alphanumeric + underscores only
- **Passwords:** Stored as bcrypt hashes via Werkzeug — never in plaintext

---

## 📁 Project Structure

```
disease_prediction_project/
│
├── app.py                        # Main Flask application
├── train_models.py               # EfficientNet training script
├── model.py                      # ML model utilities
├── Procfile                      # Gunicorn process definition
├── Dockerfile                    # Docker container setup
├── requirements.txt              # Python dependencies
│
├── disease_model.pkl             # Trained symptom classifier
├── final_disease_model.h5        # Trained skin disease CNN
├── dataset.csv                   # Symptom dataset (41 diseases, 130+ features)
│
├── templates/
│   ├── base.html                 # Shared navigation layout
│   ├── index.html                # Landing page
│   ├── login.html                # Login form
│   ├── register.html             # Registration form
│   ├── analyze.html              # Symptom checker page
│   ├── image_disease.html        # Image prediction page
│   └── history.html              # Prediction history page
│
└── static/
    ├── style.css                 # Main stylesheet
    ├── auth.css                  # Auth pages stylesheet
    ├── script.js                 # Frontend JavaScript
    └── images/                   # Carousel and UI assets
```

---



## 🩺 How It Works

### Symptom Prediction
1. User selects symptoms from a searchable list
2. A binary feature vector (130+ symptoms) is constructed
3. The Random Forest model predicts the most likely disease
4. Confidence probability is displayed and saved to history

### Image Prediction
1. User uploads a skin image (JPG/PNG)
2. Image is resized to 224×224 and EfficientNet-preprocessed
3. The fine-tuned CNN classifies the skin condition
4. Top prediction with confidence score is shown and saved

---


<div align="center">
  Made with ❤️ using Flask + TensorFlow + MongoDB
</div>
