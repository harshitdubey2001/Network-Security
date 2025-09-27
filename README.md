# Network Security Detection ML Project

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Model Details](#model-details)
* [Results](#results)
* [Experiment Tracking](#experiment-tracking)
* [Deployment](#deployment)
* [Contributing](#contributing)
* [Contact](#contact)

---

## Project Overview

This project aims to detect network security threats using machine learning algorithms. It leverages a dataset provided by the instructor to train and evaluate models capable of predicting potential security breaches or anomalies.

The project includes:

* **Streamlit app** for real-time predictions.
* **MLflow** for experiment tracking.
* **DagsHub** for version control, remote experiment tracking, and dataset management.
* **FastAPI backend** for serving the trained model as an API.

---

## Features

* Preprocessing and feature engineering for network data.
* Training and evaluation of multiple ML models.
* Experiment tracking with MLflow and DagsHub integration.
* REST API deployment using FastAPI.
* Streamlit app for interactive prediction.
* Exported model in `.pkl` format for easy deployment.
* Performance metrics including Accuracy, Precision, Recall, and F1-Score.

---

## Dataset

* It came **pre-standardized and feature-engineered**, allowing the project to focus on model training, optimization, and deployment.
* Target column: `RESULT`
* No additional preprocessing of raw data was required.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/harshitdubey2001/Network-Security.git
   cd Network-Security
   ```
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

* Upload a test CSV file in the app interface to see predictions.
* The app will display the predicted result for each entry along with model confidence.

### Running the FastAPI Server

```bash
uvicorn main:app --reload
```

* The API will be available at `http://127.0.0.1:8000`
* You can send POST requests with input JSON to get predictions.

### Example Test CSV

```csv
feature1,feature2,feature3,...,featureN
val1,val2,val3,...,valN
```

---

## Model Details

* Best model: **Random Forest**
* Best hyperparameters:

  ```python
  {'n_estimators': 128}
  ```
* Training Accuracy: 0.991
* Test Accuracy: 0.975
* Metrics: Precision, Recall, F1-Score

---

## Results

| Model         | Train Score | Test Score |
| ------------- | ----------- | ---------- |
| Random Forest | 0.9906      | 0.9751     |
| Logistic Reg. | 0.92        | 0.91       |
| Decision Tree | 0.98        | 0.96       |

---

## Experiment Tracking

* **MLflow** is used for tracking experiments, logging metrics, and storing model artifacts locally and remotely.
* **DagsHub** is integrated for remote experiment tracking, dataset versioning, and collaboration.

---

## Deployment

* The trained model is served using **FastAPI** for API-based predictions.
* Streamlit is used as a **frontend interface** for user interaction.

---

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. Make your changes.
4. Commit your changes: `git commit -m "Add feature"`
5. Push to the branch: `git push origin feature-name`
6. Create a Pull Request.

---

## Contact

* **Harshit Dubey**
* Email: [harshitdubey7896@gmail.com](mailto:harshitdubey7896@gmail.com)
* GitHub: [https://github.com/harshitdubey2001](https://github.com/harshitdubey2001)
