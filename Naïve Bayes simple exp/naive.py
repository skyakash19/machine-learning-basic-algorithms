# Import necessary libraries
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

# Step 1: Load Dataset
file_path = "spam.csv"  # Ensure you uploaded the file
data = pd.read_csv(file_path, encoding='latin-1')

# Step 2: Clean Data (Keep only required columns)
data = data[['Category', 'Message']]
data.columns = ['Label', 'Message']
data['Spam'] = data['Label'].apply(lambda x: 1 if x == 'spam' else 0)  # Convert labels to numbers

# Step 3: Display Ham & Spam Counts
ham_count = (data['Spam'] == 0).sum()
spam_count = (data['Spam'] == 1).sum()
print(f"\n📊 Total Messages: {len(data)}")
print(f"✅ Ham Messages: {ham_count}")
print(f"🚨 Spam Messages: {spam_count}")

# Step 4: Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Spam'], test_size=0.25, random_state=42)

# Step 5: Create and Train Naïve Bayes Classifier
clf = Pipeline([
    ('vectorizer', CountVectorizer()),  # Convert text to numbers
    ('nb', MultinomialNB())  # Train Naïve Bayes model
])

clf.fit(X_train, y_train)  # Train the model

# Step 6: Test and Compute Accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Test with Custom Messages
sample_messages = [
    "Congratulations! You have won a lottery. Claim your prize now!",
    "Hey, are we still meeting for coffee tomorrow?"
]
predictions = clf.predict(sample_messages)

print("\n🔍 Spam Predictions:")
for msg, pred in zip(sample_messages, predictions):
    print(f"📩 '{msg}' ➝ {'🚨 Spam' if pred == 1 else '✅ Not Spam'}")