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
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


landing_path = 'PSU_classes/cs445_group_project/code/Keras Music Genres Classification/generated_data_sets'
data_source_path = "d:/workspace/data_sets/Kaggle GTZAND audio Data/genres_original"
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

Dataset = create_raw_dataset(data_source_path, genres)



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

    filepath = os.listdir(f'{data_source_path}')
    S_dB_dataset = [] # create an empty dataset
    for g in genres:
        for filename in os.listdir(f'{data_source_path}/{g}'):
            filename = f'{data_source_path}/{g}/{filename}'
            S_dB = compute_mel_spectrogram(filename) # convert the file to a mel spectrogram in db
            S_dB_dataset.append((filename, S_dB, g)) # add the spectrogram and genre to the dataset
        # since this may take a while, save the dataset to a file in csv format
        file = open(f'{landing_path}/generated_feature_set.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(S_dB_dataset.split())
    return S_dB_dataset

#  create and store a file copy of the dataset in csv format

if len(sys.args) == 1:
    print("creating the dataset ")
    S_dB_dataset = create_raw_dataset(data_source_path, genres)
    print("No command line parameters set, processing the GTZAND audio files")
elif 'S_dB_dataset' not in locals() or S_dB_dataset is None or len(S_dB_dataset) == 0:
    print("No dataset found")
    print("program terminated")
    sys.exit(1)

# Step 3 - create an encoder-decoder model to extract features from the dataset

# set up some parameters for the model
number_columns = len(S_dB_dataset[3])
number_inputs = len(S_dB_dataset)
num_encoder_features = number_columns
num_decoder_features = 26
encoder_states = []
decoder_outputs = []

# define the steps as functions

def preprocess_dataset(dataset):
    # preprocess the dataset by scaling and normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(dataset.iloc[:, :-1], dtype=float))
    genre_list = dataset.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    # preprocess the dataset
    X, y = preprocess_dataset(S_dB_dataset)
    genre_list = S_dB_dataset.iloc[:, -1]

    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    print("y:", y,)
    print("encoder:", encoder, "genre_list:", genre_list)

    # normalizing the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(S_dB_dataset.iloc[:, :-1], dtype=float))

    #  Step 4 Splitting the dataset into training and testing sets (20% of data for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))
    return X_train, X_test, y_train, y_test


def define_encoder_decoder_model(num_encoder_features, num_decoder_features):
    # Define the encoder
    encoder_inputs = S_dB_dataset(shape=(None, num_encoder_features))
    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Define the decoder
    decoder_inputs = S_dB_dataset(shape=(None, num_decoder_features))
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_features, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn encoder_inputs and decoder_inputs into decoder_outputs
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    return model

class FileLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        super(FileLogger, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
            f.write(f"Epoch {epoch + 1}: {logs}\n")

def train_encoder_decoder_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    # append to a file the training log for each epoch
    file_logger = FileLogger('training.log')
    model.fit([X_train, y_train], y_train,
              batch_size=128,
              epochs=100,
              validation_data=([X_test, y_test], y_test),
              callbacks=[file_logger])
    return model


# main program

# read files and process into a from the wav files into mel spectrograms
Dataset = create_raw_dataset(data_source_path, genres)
# preprocess the dataset and split into training and testing sets
X_train, X_test, y_train, y_test = preprocess_dataset(Dataset)
# define the encoder-decoder model
en_decoder_model = define_encoder_decoder_model(
                            num_encoder_features,
                            num_decoder_features)
# train the model and save the model's epoch data to a file
en_decoder_model = train_encoder_decoder_model(en_decoder_model, X_train, y_train, X_test, y_test)
# Save the encoder model for future use.
encoder_model = Model(en_decoder_model.input[0], en_decoder_model.layers[2].output)
encoder_model.save('encoder_model.h5')
# calculate accuracy using the test data
test_loss, test_acc = en_decoder_model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)

# predictions
predictions = en_decoder_model.predict(X_test)
print(np.argmax(predictions[0]))














# Loading Keras
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow import keras
from keras import layers, models

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
