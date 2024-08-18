'''
A program to utilize an encoder-decoder ML model to extract features from a dataset.
first version (0.0.1) uses wav files as input and genre as output

Step 1 read in the Kaggle GTZAND audio Data files and process the files
       into mel spectrograms in decibles.
Step 2 create a dataset of the mel spectrograms and the genre of the file
Step 3 create an encoder-decoder model to extract features from the dataset
Step 4 train the model on the dataset
'''
# List of required packages
required_packages = ['librosa', 'matplotlib', 'numpy', 'scikit-learn', 'pandas', 'tensorflow',
                      'keras', 'seaborn', 'csv']
import subprocess
import sys
# Check if each package is installed, and if not, install it
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
import librosa
from matplotlib import pyplot as plt
import numpy as np
import os
import csv
import sys
import sklearn
import pandas as pd

# note: to import keras, install only tensorflow and not keras
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import mse, mae
from tensorflow.keras.utils import to_categorical
from keras import layers, models, Model, callbacks, optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#from keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix
import seaborn as sns
import subprocess
#from keras.optimizers import Adam

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

def create_preprocessed_dataset(data_source_path, genres):
    # walk through the folders of the GTZAND audio files by genres
    # create a dataset of the mel spectrograms and the genre of the file
    # normalize the spectrogram and scale the columns
    # perform PCA on the spectrogram to reduce the dimensions to a list

    S_dB_dataset = [] # create an empty dataset
    for g in genres:
        for filename in os.listdir(f'{data_source_path}/{g}'):
            filename_path = f'{data_source_path}/{g}/{filename}'
            S_dB = compute_mel_spectrogram(filename_path) # convert the file to a mel spectrogram in db
            # normalize the spectrogram
            S_dB = sklearn.preprocessing.normalize(S_dB)
            # scale the spectrogram columns
            scaler = StandardScaler()
            S_dB = scaler.fit_transform(S_dB)
            # Perform PCA on the spectrgram to reduce the dimensions to a list
            pca = PCA(n_components=1)
            pca_result = pca.fit_transform(S_dB.T)
            pca_result = pca_result.astype(np.float64)
            pca_result_list = pca_result.flatten().astype(np.float64)
            filename = filename_path.split('/')[-1] # get the last node of the filename_path
            S_dB_dataset.append((filename, pca_result_list, g)) # add the spectrogram and genre to the dataset
        # since this may take a while, save the dataset to a file in csv format
        file = open(f'{landing_path}/generated_feature_set_spectro.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(S_dB_dataset)    

    return S_dB_dataset


# Step 3 - create an encoder-decoder model to extract features from the dataset

# define the steps as functions

def pad_vectors(dataset, target_length):
    padded_dataset = []
    for row in dataset:
        if len(row) < target_length:
            padding = [0.000001] * (target_length - len(row))
            padded_row = np.concatenate([row, padding])
        else:
            padded_row = row[:target_length]
        padded_dataset.append(padded_row)
    return np.array(padded_dataset)

def split_dataset_into_train_and_test_sets(dataset, genre_list):
    # preprocess the dataset by scaling and normalizing the data

    #number_of_features = len(dataset[0][1])
    #max_length = max(len(row[1]) for row in dataset)
    #dataset = pad_vectors(dataset, max_length)
    target_length = max(len(row[1]) for row in dataset)
    X = pad_vectors([row[1] for row in dataset], target_length)
    X = np.array(X)

    # Scale and normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # encode the genres labels to integers``
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    genres_dict = dict(zip(np.unique(y), np.unique(genre_list)))
    # split the dataset into training, testing, and validation sets
    # First, split the data into train (50%) and temp (50%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)

    # Then split the temp data into test (25% of total) and validation (25% of total)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val, genres_dict

def split_reduced_features_into_train_and_test_sets(dataset, genre_list):
    X = np.array(dataset[1])  # Features
    y = np.array(dataset[0])  # Genres

    # Encode the genres labels to integers
    # encode the genres labels to integers``
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    genres_dict = dict(zip(np.unique(y), np.unique(genre_list)))
    #genres_dict = dict(zip(np.unique(y), encoder.classes_))

    # Split the dataset into training, testing, and validation sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val, genres_dict






class FileLogger(callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename
   
    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
            metrics_to_log = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'mse', 'val_mse', 'mae', 'val_mae']
            log_string = f"Epoch {epoch+1}"
            for metric in metrics_to_log:
                if metric in logs:
                    log_string += f", {metric}: {logs[metric]:.4f}"
            f.write(log_string + "\n")
        return

def define_encoder_decoder_model(input_dim):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(100, activation='relu')(input_layer)
    encoder = layers.Dense(75, activation='relu')(encoder)
    encoder_output = layers.Dense(64, activation='relu', name='bottleneck')(encoder)
    
    # Decoder
    decoder = layers.Dense(64, activation='relu')(encoder_output)
    decoder = layers.Dense(100, activation='relu')(decoder)
    decoder_output = layers.Dense(input_dim, activation='relu')(decoder)
    
    # Model
    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    
    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse', 'mae'])
    
    return autoencoder, input_layer, encoder_output

def train_encoder_decoder_model(model, X_train, y_train, X_test, x_test):
    
    # Train the model
    file_logger = FileLogger('training.log')

    history = model[0].fit(X_train, y_train, 
              batch_size=100, 
              epochs=100, 
              verbose=2, 
              validation_data=(X_test, x_test),
              callbacks=[file_logger])

    return model

def save_encoder_model(autoencoder_tuples, filename):
    encoder_layers = autoencoder_tuples.layers[:autoencoder_tuples.layers.index(
        autoencoder_tuples.get_layer('bottleneck'))+1]
    encoder = tf.keras.Sequential(encoder_layers)

    # Compile the encoder
    encoder.compile(optimizer='adam', loss='mse')

    # Save the compiled encoder
    encoder.save('encoder_model.h5')    

def select_top_features(trained_model, X_test, n_features=64):
    # Get the encoded representation
    encoder = trained_model.get_layer('encoder')
    encoded_features = encoder.predict(X_test)
    
    # Calculate the importance of each feature
    feature_importance = np.mean(np.abs(encoded_features), axis=0)
    
    # Get the indices of the top n_features
    top_feature_indices = np.argsort(feature_importance)[-n_features:]
    
    # Create a new dataset with only the top features
    X_test_top_features = encoded_features[:, top_feature_indices]
    
    return X_test_top_features, top_feature_indices


def generate_report(trained_model, X_test, y_test, genres_dictionary, filename_prefix):
    # Calculate accuracy and loss using the test data
    test_loss = trained_model.evaluate(X_test, X_test, verbose=0)
    
    # Predictions
    predictions = trained_model.predict(X_test)

    # Save output to file
    with open(f'{filename_prefix}_report.txt', 'w') as f:
        f.write(f'genres dictionary: {genres_dictionary}\n')
        f.write(f'Test_loss: {test_loss}\n')
        f.write(f'predictions: {np.argmax(predictions[0])}\n')
    #print("generating confusion matrix plot")
    # Generate and save confusion matrix
    #y_pred_classes = np.argmax(predictions, axis=1)
    #y_true_classes = np.argmax(y_test, axis=1)
    #cm = confusion_matrix(y_true_classes, y_pred_classes)
    #plt.figure(figsize=(10, 7))
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genres, yticklabels=genres)
    #plt.xlabel('Predicted')
    #plt.ylabel('True')
    #plt.title('Confusion Matrix')
    #plt.savefig(f'{filename_prefix}_confusion_matrix.png')

    print("Ploting metrics")
    # generate plots by epoch for loss, accuracy, mse, and mae from the log file
    epoch, loss, val_loss, accuracy, val_accuracy, mse, val_mse, mae, val_mae = [], [], [], [], [], [], [], [], []
    with open('training.log', 'r') as f:
        for line in f:
            if 'Epoch' in line:
                parts = line.split(',')
                epoch.append(int(parts[0].split(' ')[-1]))
                loss.append(float(parts[1].split(':')[-1]))
                val_loss.append(float(parts[2].split(':')[-1]))
                accuracy.append(float(parts[3].split(':')[-1]))
                val_accuracy.append(float(parts[4].split(':')[-1]))
                mse.append(float(parts[5].split(':')[-1]))
                val_mse.append(float(parts[6].split(':')[-1]))
                mae.append(float(parts[7].split(':')[-1]))
                val_mae.append(float(parts[8].split(':')[-1]))

    # plot the metrics                
    plt.figure(figsize=(10, 7))
    plt.plot(epoch, loss, label='Training Loss')
    plt.plot(epoch, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{filename_prefix}_loss_plot.png')

    plt.figure(figsize=(10, 7))
    plt.plot(epoch, accuracy, label='Training Accuracy')
    plt.plot(epoch, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(f'{filename_prefix}_accuracy_plot.png')

    plt.figure(figsize=(10, 7))
    plt.plot(epoch, mse, label='Training MSE')
    plt.plot(epoch, val_mse, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.legend()
    plt.savefig(f'{filename_prefix}_mse_plot.png')

    plt.figure(figsize=(10, 7))
    plt.plot(epoch, mae, label='Training MAE')
    plt.plot(epoch, val_mae, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.savefig(f'{filename_prefix}_mae_plot.png')

    return
                

                                    

def run_encoder_decoder(data_source_path, output_file, genres):
    if len(sys.argv) >= 2:
        if len(sys.argv[3]) < 1:
            print("No command line parameter for genres, will use default genres list")
            genres = genres
        elif len(sys.argv[3]) >= 3:   
            genres = sys.argv[2].split()
            print("Building raw dataset from the command line parameter for folders of audio files")
        else:
            print("Creating the dataset from folders of GTZAND audio files")
            genres = os.listdir(data_source_path)
    S_dB_dataset = create_preprocessed_dataset(data_source_path, genres)
    if S_dB_dataset is None:
        print("No dataset created, end program.")
        exit()
    print("splitting autoencoder dataset")        
    genres_dataset = np.array([row[2] for row in S_dB_dataset])
    # Preprocess the dataset and split into training and testing sets
    X_train, X_test, X_val, y_train, y_test, y_val, genres_dict = split_dataset_into_train_and_test_sets(S_dB_dataset, genres_dataset)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    encoder_decoder_model = define_encoder_decoder_model(np.shape(X_train[1])[0])
    print("training autoencoder")
    # Train the model and save the model's epoch data to a file
    trained_model = train_encoder_decoder_model(encoder_decoder_model, X_train, X_train, X_val, X_val)
    # Save trained encode-decoder model.
    print("saving autoencoder models")
    trained_model[0].save('encoder_decoder_model.h5')
    # Save the encoder model for future use.
    save_encoder_model(trained_model[0], 'encoder_model.h5')
    #encoder_model = Model(inputs=encoder_decoder_model[1], outputs=encoder_decoder_model[2])
    #encoder_model.save('encoder_model.h5')

    # Generate report
    generate_report(trained_model[0], X_train, X_train, genres_dict, 'autoencoder')
    return

def create_reduced_feature_set(extraction_path, landing_path, genres):
    # Create a list of all the folders in the data_source_path
    genres = os.listdir(data_source_path)
    S_dB_dataset = create_preprocessed_dataset(data_source_path, genres)
    if S_dB_dataset is None:
        print("No dataset created, end program.")
        exit()
    # prepare the dataset into X and Y
    target_length = max(len(row[1]) for row in S_dB_dataset)
    X = np.array(pad_vectors([row[1] for row in S_dB_dataset], target_length))
    Y = np.array([[row[0], row[2]] for row in S_dB_dataset])

    # load the encoder model generate by the autoencoder model
    encoder_model = tf.keras.models.load_model('encoder_model.h5', custom_objects={'mse': mse, 'mae': mae})

    # generate the encoder features
    encoded_features = encoder_model.predict(X)
    
        # Find the maximum length of encoded features
    max_length = max(len(feature) for feature in encoded_features)

    # Pad the encoded features and add small value to zeros
    padded_encoded_features = []
    for feature in encoded_features:
        padded_feature = np.pad(feature, (0, max_length - len(feature)), 
                                constant_values=0.00001)
        padded_feature[padded_feature == 0] = 0.00001
        padded_encoded_features.append(padded_feature)

     # Convert the padded_encoded_features to a numpy array
    features_array = np.array(padded_encoded_features)

    # Create numpy arrays for filenames and genres
    filenames_array = np.array([row[0] for row in S_dB_dataset])
    genres_array = np.array([row[2] for row in S_dB_dataset])

    # Save the arrays separately
    np.savez('reduced_features.npz', 
             filenames=filenames_array, 
             features=features_array, 
             genres=genres_array)

    return {
        'filenames': filenames_array,
        'features': features_array,
        'genres': genres_array
    }, genres

# end of autoencoder feature extraction methods

#To implement a custom training loop using tf.GradientTape(), you can add the following code after defining your model and compiling it:

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = models.model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    gradients = tape.gradient(loss, models.model.trainable_variables)
    models.optimizer.apply_gradients(zip(gradients, models.model.trainable_variables))
    return loss

def define_classifier_model(input_shape, genres, X_train, Y_train):
    print("input_shape", input_shape)
    # Define the model
    #model = models.Sequential([
    #layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    #layers.Dense(len(genres), activation='softmax')
    #])
    #model = models.Sequential()
    model = models.Model
    #model.add(layers.Flatten(input_shape=(input_shape,)))
    model.add(layers.Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(len(genres), activation='softmax'))
    
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy','mse', 'mean_absolute_error'])

    # Custom training loop
    #for epoch in range(100):
    #    for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(32):
    #        loss = train_step(x_batch, y_batch)
    #    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
    return model

def data_generator(X, y, batch_size):
    while True:
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

def train_classifier( model, X_train, Y_train, X_test, Y_test):
    file_logger = FileLogger('classifier_training.log')
    print("X_train shape:", X_train.shape, "Y_train shape:", Y_train.shape, "X_test shape:", X_test.shape, "Y_test shape:", Y_test.shape)
    print(np.unique(Y_train))
    print(np.unique(Y_test))
    Y_train = Y_train.astype(np.int32)
    Y_test = Y_test.astype(np.int32)
    #Y_train = to_categorical(Y_train)
    #Y_test = to_categorical(Y_test)
    Y_train = np.argmax(y_train, axis=0)
    Y_test = np.argmax(y_test, axis=0)
    #X_train = to_categorical(X_train)
    #X_test = to_categorical(X_test)
    # Fit the model
    #history = model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_data=(X_test, Y_test),
    #               verbose=1, callbacks=[file_logger])
    #history = model.fit(data_generator(X_train, Y_train, 32),
    #                steps_per_epoch=len(X_train)//32,
    #                epochs=100,
    #                validation_data=data_generator(X_test, Y_test, 32),
    #                validation_steps=len(X_test)//32,
    #                callbacks=[file_logger])
    history = model[0].fit(X_train, Y_train,
                steps_per_epoch=len(X_train)//32,
                epochs=100,
                validation_data=(X_test, Y_test),
                validation_steps=len(X_test)//32,
                callbacks=[file_logger])
    return history
    
def test_classifier(model, X_test, Y_test):
    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("Loss: %.2f%%" % (scores[2]))
    print("Mean absolute error: %.2f%%" % (scores[3]))
    return

# main program

# Example usage
landing_path = 'generated_data_sets/'
data_source_path = "d:/workspace/data_sets/Kaggle GTZAND audio Data/genres_original"
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
output_file = 'autoencoder_metrics_report.txt'
print("Creating encoder model from the autoencoder model for the GTZAND dataset")
# create the trained encoder model from the autoencoder model for the GTZAND dataset
run_encoder_decoder(data_source_path, output_file, genres)
# Create the reduced feature set from the trained encoder model and save as a file "reduced_features.npy"
print("Creating reduced feature set from the trained encoder model")
#dataset = create_reduced_feature_set(data_source_path, landing_path, genres)
reduced_features, genres = create_reduced_feature_set(data_source_path, landing_path, genres)

genres_dataset = reduced_features['genres']
#reduced_features['features'] = np.array(reduced_features['features'].tolist)
#reduced_features['genres'] = np.array(reduced_features['genres'].tolist)
#dataset = [[reduced_features['features'][i], reduced_features['genres'][i]] for i in range(len(reduced_features['genres']))]
#dataset = {
#    'genres': reduced_features['genres'],
#    'features': reduced_features['features']
#    }
dataset = []
dataset.append(reduced_features['genres'])
dataset.append(reduced_features['features'])
L = len(dataset[1])
#X_train, X_test, X_val, y_train, y_test, y_val, genres_dict = split_dataset_into_train_and_test_sets(dataset)

# split the dataset into training and testing sets
X_train, X_test, X_val, y_train, y_test, y_val, genres_dict = split_reduced_features_into_train_and_test_sets(dataset, genres_dataset)
output_file = 'classifier_metrics_report.txt'
# create_ a 4 layer classifier model
classifier_model = define_classifier_model(len(reduced_features['features'][0]), genres, X_train, y_train)
print("training classifier model")
# Train the model and save the model's epoch data to a file


trained_model = train_classifier(classifier_model, X_train, y_train, X_test, y_test)
print("testing classifier model")
# test the trained classifier model
test_classifier(trained_model, X_test, y_test)
print("Generating classifier reports")
generate_report(trained_model, X_train, y_train, genres_dict, 'classifier')

# end of main program
