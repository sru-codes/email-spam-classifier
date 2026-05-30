# Email Spam Classifier (Minor Project)

**Authors:** Srustisri Panda, Puja Rani Mishra, Kajal Roul  
**Project Type:** Group Project  
**Topic:** Classification Projects  
**Project ID:** Minor Project #13 (from AI/ML Project List)

---

## 📌 Project Overview
The **Email Spam Classifier** is an Intelligent system that automatically categorizes incoming emails as either **Spam** or **Ham** (Legitimate). This project leverages **Natural Language Processing (NLP)** and the **Multinomial Naive Bayes** algorithm to identify patterns and keywords commonly found in fraudulent communications.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, Scikit-learn, NLTK, Joblib
- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)

## 📊 Features & Functionality
- **Text Vectorization:** Converts raw text into numerical data using the TF-IDF technique.
- **Stopword Removal:** Filters out common words that don't contribute to classification.
- **High Efficiency:** Naive Bayes provides fast and accurate results even with limited datasets.
- **Model Evaluation:** Provides detailed reports including Accuracy, Precision, Recall, and F1-Score.

## 🚀 Getting Started

### Prerequisites
Install the necessary Python packages:
```bash
pip install pandas scikit-learn joblib
```

### Running the Project
1. **Train the Classifier:**
   ```bash
   python train.py
   ```
   This will process the sample dataset, train the model, and save the classifier and vectorizer for deployment.

---
*Developed as part of the AI/ML with Python Course.*
