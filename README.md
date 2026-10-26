# Explainable Pneumonia Detection

**An Explainable Deep Learning Framework for Automated Pneumonia Detection and Risk Assessment from Chest X-Ray Images**

A full stack research prototype that classifies chest X-rays as normal or pneumonia, shows *why* the model made its decision with a visual explanation heatmap, and reports a risk assessment. Built with a Django REST backend, a React frontend and a deep learning model.

> **Disclaimer:** This is a research and educational project. It is **not** a medical device and must not be used for clinical diagnosis.

---

## Features

- Upload a chest X-ray and get a **Normal / Pneumonia** prediction with a confidence score
- **Explainable AI:** a heatmap highlighting the lung regions that influenced the prediction (`[Grad-CAM / LIME / SHAP]`)
- **Risk assessment** based on the model's output probability
- REST API connecting the model to a responsive React interface
- Clear separation of model code (`ml`), API (`predictor`) and UI (`frontend`)

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | React.js, JavaScript (ES6+), HTML5, CSS3 |
| Backend | Python, Django, Django REST Framework |
| Deep learning | `[TensorFlow / PyTorch]`, `[model name]` |
| Explainability | `[Grad-CAM / LIME / SHAP]` |
| Database | SQLite (development) |

---

## Architecture

```
React UI  -->  Django REST API  -->  Model inference  -->  Explanation heatmap + risk score
```

---

## Project Structure

```
explainable-pneumonia-detection/
├── backend/
│   ├── config/          # Django settings and URLs
│   ├── predictor/       # API views and prediction logic
│   ├── ml/              # Model training, loading and explanation code
│   ├── manage.py
│   └── requirements.txt
├── frontend/
│   ├── src/             # React components
│   ├── index.html
│   └── package.json
└── README.md
```

---

## Dataset

- **Source:** `[dataset name and link]`
- **Classes:** Normal, Pneumonia
- The dataset is not included because of its size. Download it and place it in `backend/dataset/`.

---

## Model and Results

| Item | Details |
|---|---|
| Architecture | `[model name]` |
| Explanation method | `[Grad-CAM / LIME / SHAP]` |

| Metric | Value |
|---|---|
| Accuracy | `[xx.x%]` |
| Precision | `[xx.x%]` |
| Recall | `[xx.x%]` |
| F1-score | `[xx.x%]` |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/KalpanaJayapaul/explainable-pneumonia-detection.git
cd explainable-pneumonia-detection
```

### 2. Run the backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Limitations

- Trained on a single public dataset, so performance on other scanners or hospitals may differ
- Heatmaps show where the model looked, not a clinical explanation
- Not validated by medical professionals

---

## Author

**Kalpana Jayapaul**
Full Stack Developer | React.js · Django · Python | LLM & AI Integration
GitHub: [@KalpanaJayapaul](https://github.com/KalpanaJayapaul)
