# Course Completion Prediction with XGBoost

This repository contains my midterm project for the Zoomcamp ML Bootcamp. The goal of this project is to predict the likelihood of course completion using engagement data from an online education platform. The dataset is sourced from Kaggle and analyzed using Python.

## Project Overview

The project involves:

1. **Dataset**: Course engagement data with features such as time spent on courses, quizzes taken, scores, and device type.
2. **Model**: XGBoost classifier for binary classification.
3. **Environment**: Pipenv for managing Python dependencies and virtual environments.
4. **Deployment**: Dockerized Flask application exposing an API for prediction.

## Features and Workflow

### Data Preprocessing

- Handled numerical and categorical data:
  - **Numerical Features**: Scaled using `StandardScaler`.
  - **Categorical Features**: Encoded using `OneHotEncoder`.
  
### Model Training

- Model: XGBoost with hyperparameter tuning for optimal performance.
- Metric: Evaluated using AUC (Area Under the Curve).

### Deployment

- Deployed as a Docker container with Flask to provide an API for predictions.
- Used AWS Elastic Beanstalk for cloud deployment.

### Tools and Technologies

- **Python**: Core language for model development.
- **XGBoost**: Machine learning framework for training the model.
- **Pipenv**: For managing dependencies.
- **Docker**: For containerization.
- **AWS S3**: To store the trained model.
- **AWS Elastic Beanstalk**: For deploying the Flask API.

## API Usage

Once deployed, the API accepts a `POST` request with JSON input representing a student's data and returns the predicted probability of course completion.

Example Request:

```json
{
    "coursecategory": "programming",
    "devicetype": "laptop",
    "numberofquizzestaken": 3,
    "quizscores": 100,
    "timespentoncourse": 100,
    "numberofvideoswatched": 100,
    "completionrate": 100
}
```

## How to Run the Project Locally

Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
```

```bash
pip install pipenv
pipenv install
pipenv shell
```

```bash
python train.py
```

```bash
python predict.py
```

```bash
docker build -t course_completion .
docker run -it -p 4545:4545 course_completion:latest
```
