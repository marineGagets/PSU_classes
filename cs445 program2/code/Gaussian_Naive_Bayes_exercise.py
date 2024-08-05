'''
Machine Language Programming assignment 2
see the assignment description in the README.md file
'''

import numpy as np
import pandas as pd

# define a function to process the feature labels.
def read_and_truncate_file(file_path):
    feature_labels = []
    with open(file_path, 'r') as file:
        for line in file:
            feature_label = line.split(':')[0]
            feature_labels.append(feature_label)
    return feature_labels

# Load the raw data
rawdata = pd.read_csv('datasets\spambase.data', header=None)
print("rawdata shape", rawdata.shape)
# load the feature labels for the rawdata
feature_labels = read_and_truncate_file('datasets\\feature_labels.txt')

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
test_data = test_data.iloc[:, :-1]

# Remove the selected test_data rows from rawdata to get training_data
training_data = rawdata.drop(test_data.index)
# Define the labels for training_data
training_targets = training_data.iloc[:, -1]
# trim off the targets from the training data
training_data = training_data.iloc[:, :-1]
print("training data shape", training_data.shape)



#
# step 2: create probalistic models for each class
#

# Determine each of the hypothesis ratios for each hypotesis (H), based on the test data
# P(H|D) = P(D|H) * P(H) / P(D)
hypotheses, frequencies = np.unique(training_targets, return_counts=True)
hypothesis_ratios = frequencies / len(test_targets)

# Calculate the prior probabilities P(H)
# P(H) = P(D|H) * P(H) / P(D)

for h, f, r in zip(hypotheses, frequencies, hypothesis_ratios):
    print(f"{'Hypothesis':<15} {'Frequency':<15} {'Ratio':<15}")
    print(f"{h:<15} {f:<15} {r:<15.2f}")
    print()
    print("prior probability of each hypothesis")
    print("hyposis: spam,  P(h1):", hypothesis_ratios[1])
    print("hyposis: not spam,  P(h0):", hypothesis_ratios[0])
    print(" P(+|h1):", 1 - hypothesis_ratios[1], "P(-|h1):", hypothesis_ratios[1])
    print(" P(+|h0):", 1 - hypothesis_ratios[0], "P(-|h0):", hypothesis_ratios[0])

# Filter rows for each unique value in training_targets
grouped_data = training_data.groupby(training_targets)

# Calculate the mean and standard deviation for each group

for clas, group in grouped_data:
    mean = group.mean()
    std_dev = group.std()
    print(f"\nGroup for target = {clas}")
    for col in range(len(mean)):
        print(f"Column {col:<10} Mean: {mean[col]:<15.4f} Standard Deviation: {std_dev[col]:<15.4f}")

# probalistic Model
# build the variance table from the means for each column and the column variance,
# where the variance is the square of the standard deviation

variance_table_data = []
for clas, group in grouped_data:
    mean = group.mean()
    std_dev = group.std()
    variance = std_dev ** 2
    for i, feature_label in enumerate(feature_labels):
        print("class", clas, "feature", feature_label, "mean", mean[i], "std_dev", std_dev[i], variance[i])
        variance_table_data.append({'Class': clas, 'Feature': feature_label, 'Mean': mean[i], 'Standard Deviation': std_dev[i],\
                                     'Variance': variance[i]})
variance_table = pd.DataFrame(variance_table_data)
variance_table_class0 = variance_table[variance_table['Class'] == 0]
variance_table_class1 = variance_table[variance_table['Class'] == 1]

print(np.shape(variance_table)    )

print("\nVariance Table")
print()
print(f"{'                                     class 0':37}{'                         class 1':<25}")
print(f"{'Feature':<30} {'Spam_mean':<10} {'Spam_variance':<15} {'Not Spam_mean':<13} {'Not Spam_variance':<16}")
for i in range(57):
    print(f"{variance_table_class0.iloc[i, 1]:<30}", f"{variance_table_class0.iloc[i, 2]:<10.4f}", \
          f"{variance_table_class0.iloc[i, 4]:<15.4f}", f"{variance_table_class1.iloc[i, 2]:<13.4f}", \
          f"{variance_table_class1.iloc[i, 4]:<16.4f}")


# Gaussian Naive Bayes Algorithm `
