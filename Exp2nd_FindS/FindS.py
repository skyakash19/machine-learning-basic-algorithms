import pandas as pd # type: ignore

# Function to implement FIND-S algorithm
def find_s_algorithm(data):
    # Extract features and labels
    attributes = data.iloc[:, :-1]  # All columns except last (features)
    labels = data.iloc[:, -1]  # Last column (target class)

    # Initialize most specific hypothesis with the first positive example
    for i in range(len(labels)):
        if labels[i] == "Yes":  # Assuming 'Yes' represents a positive example
            hypothesis = list(attributes.iloc[i])
            break

    # Compare with other positive examples and update the hypothesis
    for i in range(len(labels)):
        if labels[i] == "Yes":
            for j in range(len(hypothesis)):
                if attributes.iloc[i, j] != hypothesis[j]:
                    hypothesis[j] = "?"  # Generalize attribute

    return hypothesis

# Load training data from CSV file
filename = "Weather.csv"  # Replace with your actual file name
data = pd.read_csv(filename)

# Apply FIND-S algorithm
final_hypothesis = find_s_algorithm(data)

# Display the final hypothesis
print("Most Specific Hypothesis:", final_hypothesis)
