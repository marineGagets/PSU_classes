## Autoencoder feature extraction

 key points about the code:

- It processes audio files from the GTZAN dataset, converting them to mel spectrograms.

- The script creates an encoder-decoder model to extract features from the mel spectrograms with a sampling rate of 128 bands

- It includes functions for data preprocessing, model definition, training, and evaluation.

- The encoder part of the model is saved separately for feature extraction.

- It generates various plots and reports to visualize model performance.

- the encoder portion of the autoencoder is saved as a separate file for later use for feature extraction (currently the 64 most significant features are used)

- After feature extraction, it creates a classifier model to predict music genres. - not this does not work yet

- The code uses libraries like librosa for audio processing, TensorFlow/Keras for deep learning, and scikit-learn for data preprocessing.

- It implements custom callbacks for logging training progress.

This script provides a comprehensive pipeline for music genre classification, from data preprocessing to model training and evaluation. It's a good example of applying deep learning techniques to audio classification tasks.

# Usage

Each of the steps of the autoecoder and classifier are written as individual methods

An Example of how to use the autoencoder is encluded at the bottom of the program file

# References

- Bat Echolocation Call Analysis with Deep Learning Models (https://data.mendeley.com/datasets/9x2g6dsbtv/1)
- GTZAN Dataset (https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- An Efficient Neural Network Design Incorporating Autoencoders for the Classification of Bat Echolocation Sounds (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10451853/)
- mel-spectrogram (https://www.mathworks.com/help/audio/ref/melspectrogram.html)
- Gentle Introduction to the Adam Optimization Algorithm for Deep Learning (https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
- TensorFlow (https://www.tensorflow.org/)
- Keras (https://keras.io/)
- librosa (https://librosa.org/)
- scikit-learn (https://scikit-learn.org/)
