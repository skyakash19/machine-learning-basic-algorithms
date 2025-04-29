# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset (Replace 'spam.csv' with your actual dataset file)
df = pd.read_csv("spam.csv", encoding="latin-1")

# Check column names and keep only relevant ones
df = df[['Category', 'Message']]  # 'Category' (spam/ham), 'Message' (text content)
df.columns = ['Label', 'Message']

# Convert labels to numerical values ('spam' → 1, 'ham' → 0)
df['Spam'] = df['Label'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Spam'], test_size=0.25, random_state=42)

# Convert text data into numerical format using CountVectorizer and TF-IDF
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train a Naïve Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Transform test data and make predictions
X_test_counts = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = classifier.predict(X_test_tfidf)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Display the count of ham and spam messages
print("\nMessage Counts:")
print(df['Label'].value_counts())
