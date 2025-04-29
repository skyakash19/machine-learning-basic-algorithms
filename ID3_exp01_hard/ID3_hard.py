import pandas as pd # type: ignore
import numpy as np # type: ignore
from collections import Counter
from google.colab import files # type: ignore

# Node class for Decision Tree
class Node:
    def __init__(self, attribute=None, answer=None):  # Fixed init method
        self.attribute = attribute
        self.answer = answer
        self.children = []

    def add_child(self, value, node):
        self.children.append((value, node))

# Function to calculate entropy
def entropy(data):
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * np.log2(count / total) for count in label_counts.values())

# Function to calculate information gain
def information_gain(data, attribute_index):
    total_entropy = entropy(data)
    values = set(row[attribute_index] for row in data)

    weighted_entropy = 0
    for value in values:
        subset = [row for row in data if row[attribute_index] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy

# Function to find best attribute
def best_attribute(data, features):
    gains = [information_gain(data, i) for i in range(len(features))]
    return gains.index(max(gains))

# Function to build the decision tree
def build_tree(data, features):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return Node(answer=labels[0])

    if not features:
        return Node(answer=Counter(labels).most_common(1)[0][0])

    best_index = best_attribute(data, features)
    best_feature = features[best_index]
    root = Node(attribute=best_feature)

    feature_values = set(row[best_index] for row in data)
    new_features = features[:best_index] + features[best_index + 1:]

    for value in feature_values:
        subset = [row[:best_index] + row[best_index + 1:] for row in data if row[best_index] == value]
        root.add_child(value, build_tree(subset, new_features))

    return root

# Function to classify a test instance
def classify(node, x_test, features):
    if node.answer:
        print("Predicted Label:", node.answer)
        return

    x_test_values = x_test[1:]  # Ignore first column (Day)

    try:
        pos = features.index(node.attribute)
    except ValueError:
        print(f"Error: Attribute {node.attribute} not found in features {features}")
        return

    for value, child in node.children:
        if x_test_values[pos] == value:
            classify(child, x_test_values, features)
            return
    print("No matching branch found.")

# Function to load dataset
def load_excel(filename):
    df = pd.read_excel(filename)
    df.drop(df.columns[0], axis=1, inplace=True)  # Drop first column (Day)
    features = list(df.columns[:-1])  # Extract features excluding label
    return df.values.tolist(), features

# Function to upload file in Colab
def upload_file():
    uploaded = files.upload()
    for filename in uploaded.keys():
        return "/content/" + filename

# Main Execution
if __name__ == "__main__":  # Fixed main execution check
    print("📤 Upload the Training Dataset (Excel file)")
    training_file = upload_file()
    dataset, features = load_excel(training_file)

    print("\n🌳 Building the decision tree using the ID3 algorithm...")
    tree = build_tree(dataset, features)
    
    print("\n🌳 The decision tree for the dataset is:")
    if tree.attribute:
        print(tree.attribute)
    else:
        print("Decision Tree is a single-node tree with label:", tree.answer)

    print("\n📤 Upload the Test Dataset (Excel file)")
    test_file = upload_file()
    test_data, _ = load_excel(test_file)

    print("\n🔍 Classifying test instances:")
    for xtest in test_data:
        print("The test instance:", xtest)
        print("The label for test instance:", end=" ")
        classify(tree, xtest, features)
