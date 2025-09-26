# Network Security Detection ML Project

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.2-green.svg)


---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview
This project aims to detect network security threats using machine learning algorithms. It leverages a dataset provided by the instructor to train and evaluate models capable of predicting potential security breaches or anomalies.  

The project includes a **Streamlit-based interactive app** to visualize and test model predictions in real-time.

---

## Features
- Preprocessing and feature engineering for network data.
- Training and evaluation of multiple ML models.
- Streamlit app for interactive prediction.
- Exported model in `.pkl` format for easy deployment.
- Performance metrics including Accuracy, Precision, Recall, and F1-Score.

---

## Dataset
- The dataset is provided by the instructor and contains features related to network activity.
- Target column: `RESULT`
- File structure:
  ```
  phisingData.csv
  train.csv
  test.csv
  ```
> ⚠️ Ensure the dataset files are placed in the `Artifacts` folder before running the app.

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
- Upload a test CSV file in the app interface to see predictions.
- The app will display the predicted result for each entry along with model confidence.

### Example Test CSV
```csv
feature1,feature2,feature3,...,featureN
val1,val2,val3,...,valN
```

---

## Model Details
- Best model: **Random Forest**
- Best hyperparameters:
  ```python
  {'n_estimators': 128}
  ```
- Training Accuracy: 0.991  
- Test Accuracy: 0.975  
- Metrics: Precision, Recall, F1-Score

---

## Results
| Model         | Train Score | Test Score |
|---------------|------------|------------|
| Random Forest | 0.9906     | 0.9751     |
| Logistic Reg. | 0.92       | 0.91       |
| Decision Tree | 0.98       | 0.96       |

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
- **Harshit Dubey**  
- Email: harshitdubey7896@gmail.com  
- GitHub: [https://github.com/harshitdubey2001](https://github.com/harshitdubey2001)

