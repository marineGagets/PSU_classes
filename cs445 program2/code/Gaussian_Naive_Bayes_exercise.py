'''
Machine Language Programming assignment 2
see the assignment description in the README.md file
'''

import numpy as np
import pandas as pd

print("Gaussian Naive Bayes Algorithm")
print ("Machine Language Programming assignment 2")
print("Author: Larry Bilodeau")
print()
print()

# define a function to process the feature labels.
def read_and_truncate_file(file_path):
    feature_labels = []
    with open(file_path, 'r') as file:
        for line in file:
            feature_label = line.split(':')[0]
            feature_labels.append(feature_label)
    return feature_labels

# Load the raw data
rawdata = pd.read_csv('..\datasets\spambase.data', header=None)
print("Data set: spambase.data")
print("rawdata shape", rawdata.shape)
# load the feature labels for the rawdata
feature_labels = read_and_truncate_file('..\\datasets\\feature_labels.txt')

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
    print()
    print(f"{'Hypothesis':<15} {'Frequency':<15} {'Ratio':<15}")
    print(f"{h:<15} {f:<15} {r:<15.2f}")
    print()
    print("Prior Probability of each hypothesis")
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
    #print(f"\nGroup for target = {clas}")
    #for col in range(len(mean)):
    #    print(f"Column {col:<10} Mean: {mean[col]:<15.4f} Standard Deviation: {std_dev[col]:<15.4f}")

# probalistic Model
# build the variance table from the means for each column and the column variance,
# where the variance is the square of the standard deviation
print()
variance_table_data = []
for clas, group in grouped_data:
    mean = group.mean()
    std_dev = group.std()
    std_dev[std_dev == 0.0000] = 0.0001
    variance = std_dev ** 2
    variance[variance == 0.0000] = 0.0001        
    for i, feature_label in enumerate(feature_labels):
        print(f"class: {clas:<10} feature: {feature_label:<30} mean: {mean[i]:<10.4f} std_dev: {std_dev[i]:<10.4f} \
               variance: {variance[i]:<10.4f}")
        variance_table_data.append({'Class': clas, 'Feature': feature_label, 'Mean': mean[i], 'Standard Deviation': std_dev[i],\
                                     'Variance': variance[i]})
variance_table = pd.DataFrame(variance_table_data)
variance_table_class0 = variance_table[variance_table['Class'] == 0]
variance_table_class1 = variance_table[variance_table['Class'] == 1]

#print(np.shape(variance_table)    )

print("\nVariance Table")
print(f"{'                                     class 0':37}{'                         class 1':<25}")
print(f"{'Feature':<30} {'Spam_mean':<10} {'Spam_variance':<15} {'Not Spam_mean':<13} {'Not Spam_variance':<16}")
row_size = training_data.shape[1] -1
for i in range(row_size):
    print(f"{variance_table_class0.iloc[i, 1]:<30}", f"{variance_table_class0.iloc[i, 2]-1:<10.4f}", \
          f"{variance_table_class0.iloc[i, 4]:<15.4f}", f"{variance_table_class1.iloc[i, 2]-1:<13.4f}", \
          f"{variance_table_class1.iloc[i, 4]:<16.4f}")
print()

# step 3
# Gaussian Naive Bayes Algorithm `

#calculatethe gaussian probability density function

def gaussian_pdf(x, mean, variance):
    if variance == 0:
        variance = 0.0001
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))

def class_probabilites(data, variance_table_class0, variance_table_class1, hypothesis_ratios):
    # initialize the probabilities
    probabilities = []
    for i in range(len(data)):
        # initialize the probabilities for each class
        probabilities_class0 = 0
        probabilities_class1 = 0
        for j in range(len(data.iloc[i])):
            # calculate the probabilities for each class    
            probabilities_class0 += np.log(gaussian_pdf(data.iloc[i, j], variance_table_class0.iloc[j, 2], variance_table_class0.iloc[j, 4]))
            probabilities_class1 += np.log(gaussian_pdf(data.iloc[i, j], variance_table_class1.iloc[j, 2], variance_table_class1.iloc[j, 4]))
        # calculate the probabilities for each class
        probabilities_class0 += np.log(hypothesis_ratios[0])
        probabilities_class1 += np.log(hypothesis_ratios[1])
        # append the probabilities for each class
        probabilities.append([probabilities_class0, probabilities_class1])
    return probabilities

# calculate the class probabilities for the training data
probabilities = class_probabilites(training_data, variance_table_class0, variance_table_class1, hypothesis_ratios)
print()
print("Training_data class probabilities")
print("class 0", "      class 1")
for i in range(10):
    print(f"{probabilities[i][0]:10.4f} {probabilities[i][1]:10.4f}")
print()

  
# calculate the class probabilities for the test data
probabilities = class_probabilites(test_data, variance_table_class0, variance_table_class1, hypothesis_ratios)
print("Test_data class probabilities")
[print("class 0", "            class 1")]
for i in range(10):
    print(f"{probabilities[i][0]:10.4f} {' ' * 6} {probabilities[i][1]:10.4f}")
print()

# calculate the accuracy, precision, and recall of the model on the test_dataand the test_targets
def accuracy_precision_recall(test_targets, probabilities):
    # initialize the accuracy, precision, and recall
    accuracy = 0
    precision = 0
    recall = 0
    for i in range(len(test_targets)):
        # determine the predicted class
        predicted_class = 0 if probabilities[i][0] > probabilities[i][1] else 1
        # determine the actual class
        actual_class = test_targets.iloc[i]
        # increment the accuracy, precision, and recall
        accuracy += 1 if predicted_class == actual_class else 0
        precision += 1 if predicted_class == 1 and actual_class == 1 else 0
        recall += 1 if actual_class == 1 else 0
    # calculate the accuracy, precision, and recall
    accuracy /= len(test_targets)
    precision /= recall
    recall /= len(test_targets)
    return accuracy, precision, recall  

print()
print("Accuracy    Precision   Recall")
accuracy, precision, recall = accuracy_precision_recall(test_targets, probabilities)
print(f"{accuracy:.4f}     {precision:.4f}     {recall:.4f}")
print()  

# generate a confusion matrix for the test data given the test_targets
def confusion_matrix(test_targets, probabilities):
    # initialize the confusion matrix
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(test_targets)):
        # determine the predicted class
        predicted_class = 0 if probabilities[i][0] > probabilities[i][1] else 1
        # determine the actual class
        actual_class = test_targets.iloc[i]
        # increment the confusion matrix
        confusion_matrix[actual_class][predicted_class] += 1
    return confusion_matrix

print()
print("Confusion Matrix")
print(confusion_matrix(test_targets, probabilities))
