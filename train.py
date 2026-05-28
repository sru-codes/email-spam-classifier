import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Create a sample dataset
data = {
    'text': [
        'Congratulations! You have won a free lottery of $1000. Click here to claim.',
        'Hey, are we still meeting for lunch today at 1 PM?',
        'URGENT: Your account has been compromised. Please reset your password now.',
        'The meeting notes from yesterday are attached. Let me know if you have questions.',
        'Get a 50% discount on all luxury watches. Limited time offer!',
        'Can you please send me the report by the end of the day?',
        'Win a free iPhone 15 by participating in our survey.',
        'Hi Mom, I will be home late tonight. Do not wait for dinner.',
        'Exclusive deal just for you! Unsubscribe if you do not want to receive emails.',
        'Please find the invoice for your recent purchase attached below.'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1 for Spam, 0 for Ham
}

df = pd.DataFrame(data)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 3. Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 6. Save model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model saved successfully!")
