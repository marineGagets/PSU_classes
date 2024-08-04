'''
Machine Language Programming assignment 2
see the assignment description in the README.md file
'''

import numpy as np
import pandas as pd

# Load the raw data
rawdata = pd.read_csv('datasets\spambase.data', header=None)
print("rawdata shape", rawdata.shape)

#
# step 1: split the data into training and testing sets
#
# Define the labels
labels = rawdata.iloc[:, -1]

# Split the data into training and testing sets
# Select 2300 random rows with 40% labels 1 and 60% labels 0
test_data = rawdata.sample(n=2300, weights=labels.map({0: 0.6, 1: 0.4}), random_state=42)
print("test data shape", test_data.shape)

# Define the labels for test_data
test_targets = test_data.iloc[:, -1]

# Remove the selected test_data rows from rawdata to get training_data
training_data = rawdata.drop(test_data.index)
print("training data shape", training_data.shape)

# Define the labels for training_data
training_targets = training_data.iloc[:, -1]

#
# step 2: create probalistic models for each class
#

# Determine each of the hypothesis ratios for each hypotesis (H), based on the test data
# P(H|D) = P(D|H) * P(H) / P(D)
hypotheses, frequencies = np.unique(training_targets, return_counts=True)
hypothesis_ratios = frequencies / len(test_targets)

# Calculate the prior probabilities P(H)
# P(H) = P(D|H) * P(H) / P(D)

print(f"{'Hypothesis':<15} {'Frequency':<15} {'Ratio':<15}")
for h, f, r in zip(hypotheses, frequencies, hypothesis_ratios):
    print(f"{h:<15} {f:<15} {r:<15.2f}")
    print()
    print("prior probability of each hypothesis")
    print("hyposis: spam,  P(h1):", hypothesis_ratios[1])
    print("hyposis: not spam,  P(h0):", hypothesis_ratios[0])
    print(" P(+|h1):", 1 - hypothesis_ratios[1], "P(-|h1):", hypothesis_ratios[1])
    print(" P(+|h0):", 1 - hypothesis_ratios[0], "P(-|h0):", hypothesis_ratios[0])

# Filter rows for each unique value in training_targets
grouped_data = training_data.groupby(training_targets)

for clas, group in grouped_data:
    mean = group.mean()
    std_dev = group.std()

    print(f"\nGroup for target = {clas}")
    for col in range(len(mean)):
        print(f"Column {col:<10} Mean: {mean[col]:<15.4f} Standard Deviation: {std_dev[col]:<15.4f}")





# Gaussian Naive Bayes Algorithm `
