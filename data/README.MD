# 📧 Email Spam Detection Using Machine Learning with Python

## Overview

Email Spam Detection is a Machine Learning project that automatically classifies emails as **Spam** or **Ham (Legitimate Email)**. The system uses **Natural Language Processing (NLP)** techniques and a **Naive Bayes Classifier** to analyze email content and detect unwanted or fraudulent messages.

The project also includes a simple **Flask web application** that allows users to test email messages in real time.

---

## Features

* Email classification into Spam or Ham
* Text preprocessing using NLP techniques
* TF-IDF feature extraction
* Machine Learning model using Naive Bayes
* Real-time prediction through Flask Web Application
* High accuracy spam detection
* Easy-to-use interface

---

## Technologies Used

### Programming Language

* Python 3.8+

### Libraries

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Regex (re)
* Flask
* Matplotlib

### Machine Learning

* TF-IDF Vectorization
* Multinomial Naive Bayes

---

## Project Architecture

1. Data Collection
2. Data Preprocessing
3. Feature Extraction (TF-IDF)
4. Model Training
5. Model Evaluation
6. Flask Deployment
7. Real-Time Spam Prediction

---

## Dataset

The model can be trained using publicly available datasets such as:

* SMS Spam Collection Dataset
* Enron Email Dataset
* SpamAssassin Public Corpus

Dataset contains labeled messages:

* Spam
* Ham (Not Spam)

---

## Methodology

### 1. Data Preprocessing

* Convert text to lowercase
* Remove punctuation
* Remove special characters
* Remove stop words
* Tokenization

### 2. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert email text into numerical vectors.

### 3. Model Training

A Naive Bayes classifier is trained using the extracted TF-IDF features.

### 4. Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/email-spam-detection.git
cd email-spam-detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## Project Structure

```text
Email-Spam-Detection/
│
├── data/
│   ├── spam.csv
│   └── spam_aug_8000.csv
│
├── models/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── __pycache__/
│
├── app.py
├── train_model.py
├── predict.py
├── preprocess.py
├── data_loader.py
├── dummy_models.py
├── create_dummy_model.py
├── augment_data.py
├── test_trained_model.py
├── predict_debug_test.py
├── utils.py
├── utils_test_runner.py
├── requirements.txt
└── README.md
```

### File Description

| File                  |Description      |
| app.py                | Flask web application for spam prediction             |
| train_model.py        | Trains the machine learning model                     |
| predict.py            | Predicts whether an email is Spam or Ham                       |
| preprocess.py         | Text cleaning and preprocessing functions             |
| data_loader.py        | Loads and manages datasets                            |
| augment_data.py       | Creates additional training data for better accuracy  |
| dummy_models.py       | Contains dummy classifier implementations for testing |
| create_dummy_model.py | Generates dummy models for validation                 |
| test_trained_model.py | Tests the trained model performance                   |
| predict_debug_test.py | Debugging and testing prediction outputs              |
| utils.py              | Utility/helper functions used across the project      |
| utils_test_runner.py  | Runs utility function tests                           |
| spam_model.pkl        | Trained Machine Learning model                        |
| vectorizer.pkl        | Saved TF-IDF vectorizer                               |
| spam.csv              | Original spam dataset                                 |
| spam_aug_8000.csv     | Augmented dataset with additional samples             |

```
```


---

## Sample Output

Input:

```text
Congratulations! You have won a free iPhone. Click here now.
```

Prediction:

```text
Spam
```

Input:

```text
Hello, let's meet tomorrow at 10 AM.
```

Prediction:

```text
Ham
```

---

## Future Enhancements

* Gmail API Integration
* Outlook Integration
* Multi-language Spam Detection
* Deep Learning Models (LSTM, BERT)
* Mobile-Friendly Interface
* Continuous Model Retraining

---

## Results

The system successfully classifies emails with high accuracy and helps users reduce spam and phishing threats.
## Model Used

- TF-IDF Vectorization
- Multinomial Naive Bayes Classifier

## Accuracy

The model was trained on spam and ham email datasets and achieved approximately 90%+ accuracy on the test dataset.

## How to Run

1. Clone Repository
2. Install Dependencies

pip install -r requirements.txt

3. Train Model

python train_model.py

4. Start Flask Application

python app.py

5. Open Browser

http://127.0.0.1:5000
---

## Contributors

* Varun T 
---

## License

This project is developed for educational and learning purposes.
