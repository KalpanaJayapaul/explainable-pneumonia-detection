# Explainable Pneumonia Detection

### An Explainable Deep Learning Framework for Automated Pneumonia Detection and Risk Assessment from Chest X-Ray Images

A full-stack AI research prototype that analyzes chest X-ray images to classify them as **Normal** or **Pneumonia**. The system integrates deep learning with explainable AI techniques to provide visual explanations of model predictions and a probability-based risk assessment.

Built using **TensorFlow/Keras, ResNet50, Python, Django REST Framework, and React.js**, this project demonstrates how deep learning can be integrated into a full-stack application for medical image analysis.

> **Disclaimer:** This project is intended for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## Features

- **Chest X-Ray Classification:** Upload an X-ray image and receive a Normal or Pneumonia prediction.
- **Prediction Confidence:** Display the model's prediction probability or confidence score.
- **Explainable AI:** Visualize the image regions influencing the model's prediction using an explainability technique such as Grad-CAM, once implemented and verified.
- **Risk Assessment:** Provide a model-based risk category derived from the prediction probability.
- **REST API:** Connect the deep learning model to a React.js frontend through Django REST Framework.
- **Full-Stack Architecture:** Separate the frontend, backend API, and machine learning components for maintainability.
- **Research-Oriented Design:** Support experimentation with model performance, interpretability, and medical image classification.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | React.js, JavaScript (ES6+), HTML5, CSS3 |
| Backend | Python, Django, Django REST Framework |
| Deep Learning | TensorFlow, Keras, ResNet50 |
| Explainable AI | Grad-CAM / selected method, once verified |
| Image Processing | Python image-processing libraries |
| Database | SQLite for development |
| API Testing | Postman |
| Version Control | Git, GitHub |

---

## System Architecture

```text
                  ┌──────────────────────┐
                  │    React Frontend    │
                  │  Upload X-Ray Image  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Django REST API      │
                  │  Image Validation     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  ResNet50 Model       │
                  │  Image Classification │
                  └──────────┬───────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
       ┌──────────────────┐   ┌──────────────────┐
       │ Prediction and   │   │ Visual Explanation│
       │ Probability      │   │ Heatmap, if       │
       │                  │   │ implemented       │
       └─────────┬────────┘   └─────────┬────────┘
                 └───────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Results in React UI  │
                  └──────────────────────┘
```

---

## Project Structure

```text
explainable-pneumonia-detection/
├── backend/
│   ├── config/                 # Django project configuration
│   ├── predictor/              # Prediction API and request handling
│   ├── ml/                     # Model loading and inference code
│   ├── manage.py
│   └── requirements.txt
├── frontend/
│   ├── src/                    # React components and application UI
│   ├── public/                  # Public frontend assets, if used
│   ├── index.html
│   └── package.json
├── .gitignore
└── README.md
```

*Note: The structure above is a suggested logical layout. Update it to match the actual files and folders in your repository.*

---

## Dataset

**Dataset:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **Classification classes:** NORMAL and PNEUMONIA.
- **Image type:** Chest X-ray images.
- **Purpose:** Training and evaluating the deep learning classification model.
- **Dataset storage:** The dataset is not included in this repository because of its size.

Download the dataset from Kaggle and configure the dataset path according to the training scripts.

Do not upload the full dataset or large model-weight files to GitHub unless you have a suitable storage and distribution strategy.

---

## Model and Evaluation

### Model Details

| Item | Details |
|---|---|
| Model architecture | ResNet50 |
| Framework | TensorFlow / Keras |
| Input image size | 224 × 224 pixels |
| Number of classes | 2 |
| Best validation AUC | 0.98490 |
| Explainability method | Add the method actually implemented in the project |

### Evaluation Metrics

| Metric | Result |
|---|---|
| Validation AUC | 0.98490 |
| Validation accuracy | 89.44% |
| Validation precision | 98.54% |
| Validation recall | 87.08% |
| F1-score | Add the measured value |

The validation metrics above reflect the recorded model results. They should not be interpreted as evidence of clinical effectiveness or generalization to other hospitals or patient populations.

---

## Getting Started

### Prerequisites

Install the following tools before running the application:

- Python 3.9 or a compatible version supported by your dependencies
- Node.js and npm
- Git
- A code editor such as Visual Studio Code

### 1. Clone the Repository

```bash
git clone https://github.com/KalpanaJayapaul/explainable-pneumonia-detection.git
cd explainable-pneumonia-detection
```

### 2. Set Up the Django Backend

Open a terminal in the project directory:

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the backend dependencies:

```bash
pip install -r requirements.txt
```

Apply database migrations:

```bash
python manage.py migrate
```

Start the Django development server:

```bash
python manage.py runserver
```

The backend should be available at:

`http://127.0.0.1:8000/`

### 3. Set Up the React Frontend

Open a **second terminal** in the project root:

```bash
cd frontend
npm install
npm run dev
```

Open the local development URL displayed in the terminal.

> Ensure that the frontend API configuration points to the correct Django backend URL and that CORS settings permit requests from the frontend development server.

### 4. Configure the Model and Dataset

Before using the prediction functionality:

1. Download the dataset from Kaggle if you intend to reproduce training.
2. Configure the correct dataset path in the training code.
3. Ensure the trained model file is available at the path expected by the backend.
4. Verify that preprocessing during inference matches the preprocessing used during training.

Follow the actual project configuration and dependency versions when setting up the environment.

---

## Limitations

- The model is trained and evaluated using a public dataset; performance may vary across hospitals, scanners, populations, and image acquisition conditions.
- Prediction probabilities are not equivalent to clinically calibrated risk estimates.
- An explanation heatmap, when implemented, highlights image regions associated with a model prediction; it does not establish medical causation.
- Validation metrics alone do not establish clinical safety or reliability.
- The project has not been established as a clinically validated medical diagnostic system.

---

## Future Enhancements

- Integrate and evaluate Grad-CAM visual explanations.
- Add prediction history and downloadable reports.
- Improve input validation and error handling.
- Evaluate model calibration and performance on independent datasets.
- Add automated tests for the Django API and React frontend.
- Explore explainability methods and more robust model architectures.

---

## Author

**Kalpana Jayapaul**

Full-Stack Developer | React.js · Django · Python · Deep Learning · AI Integration

GitHub: [@KalpanaJayapaul](https://github.com/KalpanaJayapaul)

---

*This repository is an academic research and educational project. It is not intended for clinical use.*
