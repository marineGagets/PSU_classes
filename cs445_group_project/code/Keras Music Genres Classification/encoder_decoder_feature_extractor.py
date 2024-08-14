'''
A program to utilize an encoder-decoder ML model to extract features from a dataset.
first version (0.0.1) uses wav files as input and genre as output

Step 1 read in the Kaggle GTZAND audio Data files and process the files
       into mel spectrograms in decibles.
Step 2 create a dataset of the mel spectrograms and the genre of the file
Step 3 create an encoder-decoder model to extract features from the dataset
Step 4 train the model on the dataset

'''
import librosa
import numpy as np
import os
import csv
import sys
import sklearn
import pandas as pd

# note: to import keras, install only tensorflow and not keras
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Input


landing_path = 'generated_data_sets/'
data_source_path = "d:/workspace/data_sets/Kaggle GTZAND audio Data/genres_original"
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()


# Step 1 - load audio files (wav) and convert to mel spectrograms

def compute_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None) # digitize the wav with sampling rate found in the file
#  to be considered: set the windowing parameters for the load function to limit the size of the data
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128) # n_mels is the number of mel bands to generate
    S_dB = librosa.power_to_db(S, ref=np.max) # Convert to dB using the maximum power as reference
    return S_dB

# Step 2 - create a dataset of the mel spectrograms and the genre of the file

# for reading in the GTZAND audio files for training teh encoder-decoder model
# we need to walk down through the folders to read each file by its genres folder,
# convert the file to a specrogram and create a dataset of the spectrogram and the genre

def create_raw_dataset(data_source_path, genres):
    # walk through the folders of the GTZAND audio files by genres
    # create a dataset of the mel spectrograms and the genre of the file

    S_dB_dataset = [] # create an empty dataset
    for g in genres:
        for filename in os.listdir(f'{data_source_path}/{g}'):
            filename_path = f'{data_source_path}/{g}/{filename}'
            S_dB = compute_mel_spectrogram(filename_path) # convert the file to a mel spectrogram in db
            #filename = filename_path.split('/')[-1] # get the last node of the filename_path
            S_dB_dataset.append((filename, S_dB, g)) # add the spectrogram and genre to the dataset
        # since this may take a while, save the dataset to a file in csv format
        file = open(f'{landing_path}/generated_feature_set.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(S_dB_dataset)
    # trim the dataset to make it homogeneous
    #min_length = min(len(item[1]) for item in S_dB_dataset)
    #S_dB_dataset = [(item[0], item[1][:min_length], item[2]) for item in S_dB_dataset]
    # convert the dataset to a numpy array
    S_dB_dataset = np.array(S_dB_dataset, dtype=object)
    #S_dB_dataset = np.array([list(item) for item in S_dB_dataset])

    return S_dB_dataset


# Step 3 - create an encoder-decoder model to extract features from the dataset

# define the steps as functions

def preprocess_dataset(dataset):
    # preprocess the dataset by scaling and normalizing the data
    genre_list = dataset[:, -1]
    # drop the filename and the genres from the dataset
    data = np.array(dataset[:, 1:-1])
    # detrermine the mean for each column except the first (filename) and the last (genres)
    # and subtract the mean from each column
    mean = np.mean(data[:, 1:-1], axis=0)
    data[:, 1:-1] -= mean
    # determine the max for each column except the first (filename) and the last (genres)
    # and divide each value in the respective column
    max = np.max(data[:, 1:-1], axis=0)
    data[:, 1:-1] /= max
    X = data

   # dataset_df = pd.DataFrame(dataset[:, 1:-1])
   # scaled_features = pd.DataFrame(scaler.fit_transform(dataset_df), columns=dataset_df.columns)
   # X = scaled_features
    #genre_list = dataset[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    print("y:", y)
    print("encoder:", encoder, "genre_list:", genre_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))
    return X_train, X_test, y_train, y_test


def define_encoder_decoder_model(num_encoder_features, num_decoder_features):
    # Define the encoder
    encoder_inputs = layers.Input(shape=(None, num_encoder_features))
    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Define the decoder
    decoder_inputs = layers.Input(shape=(None, num_decoder_features))
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_features, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn encoder_inputs and decoder_inputs into decoder_outputs
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model, of note the adam optimizer uses gradient descent each layer is full connected to the next
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    return model

class FileLogger(keras.callbacks.Callback):
    def __init__(self, filename):
        super(FileLogger, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
            f.write(f"Epoch {epoch + 1}: {logs}\n")

def train_encoder_decoder_model(model, X_train, y_train, X_test, y_test):
    """
    Trains an encoder-decoder model using the provided data.

    Args:
        model: The encoder-decoder model to train.
        X_train: The input training data.
        y_train: The target training data.
        X_test: The input test data.
        y_test: The target test data.

    Returns:
        The trained encoder-decoder model.
    """
    print("shapes: X_train:", np.shape(X_train)," y_train: ", np.shape(y_train)," X_test: ", np.shape(X_test)," y_test: ", np.shape(y_test))
    # Train the model
    # append to a file the training log for each epoch
    file_logger = FileLogger('training.log')
    y_train_T = tf.convert_to_tensor(np.array([y_train]).T)
    y_test_T = tf.convert_to_tensor(np.array([y_test]).T)
    #x_train = tf.convert_to_tensor(X_train)
    model.fit(X_train, y_train_T,
              batch_size=128,
              epochs=100,
              validation_data=(X_test, y_test_T),
              callbacks=[file_logger])
    return model


# main program

# read files and process into a from the wav files into mel spectrograms
if len(sys.argv) >= 2:
    data_source_path = sys.argv[1]
    print("Building raw dataset from the command line parameter for folders of audio files")
if len(sys.argv) >= 2:
    if len(sys.argv[3]) < 1:
        print("No command line parameter for genres, will use default genres list")
        genres = genres
    elif len(sys.argv[3]) >= 3:   
        genres = sys.argv[2].split()
        print("Building raw dataset from the command line parameter for folders of audio files")
    else:
        print("Creating the dataset from folders of GTZAND audio files")
        genres = genres
S_dB_dataset = create_raw_dataset(data_source_path, genres)
if S_dB_dataset is None:
    print("No dataset created, end program.")
    exit()



# preprocess the dataset and split into training and testing sets
X_train, X_test, y_train, y_test = preprocess_dataset(S_dB_dataset)
print()
print("S_db_dataset shape: ", np.shape(S_dB_dataset), " X_train:", np.shape(X_train[788][0]), "X_test:", np.shape(X_test[1][0]), "y_train:", np.shape(y_train), "y_test:", np.shape(y_test))
# define the encoder-decoder model
# set up some parameters for the model
print(S_dB_dataset[0],"shape:", np.shape(S_dB_dataset[0]), "shape:", np.shape(S_dB_dataset[0][1]))
print(S_dB_dataset[1],"shape:", np.shape(S_dB_dataset[1]), "shape:", np.shape(S_dB_dataset[1][1]))
number_columns = len(S_dB_dataset[0][1][0])
number_inputs = len(S_dB_dataset[0][1])
print(S_dB_dataset[0][1])
num_encoder_features = number_columns
num_decoder_features = 26
encoder_states = []
decoder_outputs = []
en_decoder_model = define_encoder_decoder_model(
                            num_encoder_features,
                            num_decoder_features)
# train the model and save the model's epoch data to a file
en_decoder_model = train_encoder_decoder_model(en_decoder_model, X_train, y_train, X_test, y_test)
# save trained encode-decoder model.
en_decoder_model.save('encoder_decoder_model.h5')
# Save the encoder model for future use.
encoder_model = Model(en_decoder_model.input[0], en_decoder_model.layers[2].output)
encoder_model.save('encoder_model.h5')
# calculate accuracy using the test data
test_loss, test_acc = en_decoder_model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)

# predictions
predictions = en_decoder_model.predict(X_test)
print(np.argmax(predictions[0]))





















# Creating the model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)

# calculate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc: ', test_acc)

# Save the model
model.save('music_genres_classifier.h5')

# predictions
predictions = model.predict(X_test)
print(np.argmax(predictions[0]))
