# Minor Project 2: Email Spam Classifier

## Objective
Classify emails as spam or not spam (ham) using NLP and Naive Bayes.

## Tech Stack
- Python
- Scikit-learn (Naive Bayes, CountVectorizer/TF-IDF)
- NLTK for text preprocessing
- Pandas

## Implementation Steps
1. Use a standard spam dataset (e.g., SMS Spam Collection).
2. Preprocess text: lowercase, remove punctuation, remove stopwords, stemming/lemmatization.
3. Convert text to numerical features using TF-IDF.
4. Train a Multinomial Naive Bayes classifier.
5. Evaluate using Accuracy, Precision, Recall, and F1-score.
6. Create a script to classify new email text.
