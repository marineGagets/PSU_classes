from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import csv



landing_path = 'PSU_classes/cs445_group_project/code/Keras Music Genres Classification/generated_data_sets'


#Reading the dataset and dropping unnecessary columns
data = []
with open(f'{landing_path}/generated_feature_set.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the first row
    for row in reader:
        data.append(row)



# data = np.array(data)
data = pd.DataFrame(data)
#data = pd.read_csv(f'{landing_path}/generated_feature_set.csv', dtype={'Unnamed: 0': object})
#print(data.iloc[:, -1])
data.head() 

print("labels")
last_column = data.columns[27]
print(data[last_column].astype(str))


data = data.drop(data.columns[0], axis=1)
data.head()
print(data[0:5], data.head())

#Encoding genres into integers
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
print(y)

# normalizing the dataset
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
print(X)

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow import keras
from keras import layers, models
