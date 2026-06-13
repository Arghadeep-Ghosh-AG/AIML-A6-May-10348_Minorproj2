# AIML-A6-May-10348_Minorproj2

The Fake News Detection System is a Natural Language Processing (NLP) and Machine Learning based application designed to automatically classify news articles as real or fake. With the rapid spread of misinformation through online platforms, automated verification systems have become increasingly important.

This project utilizes a labeled dataset containing news articles and their corresponding authenticity labels. The text data is preprocessed using NLP techniques such as text cleaning, tokenization, and stop-word removal. TF-IDF (Term Frequency–Inverse Document Frequency) vectorization is applied to convert textual information into numerical feature vectors suitable for machine learning algorithms.

A classification model is trained using the processed data to identify patterns that distinguish genuine news from misleading or fabricated content. The trained model can then predict whether a new article is real or fake based on its textual content. A user-friendly Streamlit interface is provided to allow users to input news text and obtain instant predictions.

The project demonstrates the practical application of machine learning and NLP in combating misinformation and improving information reliability. Future enhancements may include deep learning models, real-time news verification, multilingual support, and integration with fact-checking APIs for improved accuracy and performance.



Features
Detects whether a news article is REAL or FAKE
NLP-based text preprocessing
TF-IDF feature extraction
Logistic Regression classifier
Streamlit-based web interface
High prediction accuracy
Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Streamlit
Natural Language Processing (NLP)
TF-IDF Vectorization
Logistic Regression
Dataset

This project uses the Fake and Real News Dataset from Kaggle.


Dataset Link:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset contains:

Fake.csv
True.csv

Place the dataset files inside the following directory:

dataset/
├── Fake.csv
└── True.csv
Project Structure
Fake-News-Detector/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── fake_news_model.pkl



## Sample Output

### Example 1

Input:

The government announced new economic reforms to boost employment and industrial growth across the country.

Output:

Prediction: REAL

### Example 2

Input:

Scientists confirm that drinking a special herbal drink can make humans live for 200 years.

Output:

Prediction: FAKE



## Model Performance

Accuracy: 98.56%
Algorithm: Logistic Regression
Feature Extraction: TF-IDF Vectorization


## Author

Arghadeep Ghosh

B.Tech Computer Science Engineering
