# Email Spam Classifier (Minor Project)

**Authors:** Srustisri Panda (sru-codes), Puja Rani Mishra (mishrapujarani11-web), Kajal Roul (kajalroul2007-ops)

**Contact:**
- Srustisri Panda: [LinkedIn](https://www.linkedin.com/in/srustisri-panda-661090398)
- Puja Rani Mishra: [mishrapujarani11@gmail.com](mailto:mishrapujarani11@gmail.com), [LinkedIn](https://www.linkedin.com/in/pujarani-mishra-0453303a6)
- Kajal Roul: [kajalroul2007@gmail.com](mailto:kajalroul2007@gmail.com), [LinkedIn](https://www.linkedin.com/in/kajal-roul-523b0b316)  
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
