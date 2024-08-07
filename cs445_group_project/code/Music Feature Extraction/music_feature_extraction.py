'''
Music Feature Extraction based on the blog https://medium.com/towards-data-science/extract-features-of-music-75a3f9bc265d

'''

# load the audio file
import librosa
audio_path = 'audio-path'
x , sr = librosa.load(audio_path)
# the default sampling rate is 22Khz and can be over ridden by passing the sr argument
# librosa.load(audio_path, sr=44100)
# the sampling can be disabled by passing the sr=None
print(type(x), type(sr))

# playing the audio file
import IPython.display as ipd
ipd.Audio(audio_path)

#display waveform
%matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
# waveplot is used to display the waveform of the audio file
librosa.display.waveplot(x, sr=sr)

#display Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
#If to pring log of frequencies  
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

# Feature Extraction
# Zero Crossing Rate
x, sr = librosa.load(audio_path)
#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)

#Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()

#Computing the zero crossing rate of the signal:
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))

#spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound
import sklearn
# .spectral_centroid is used to calculate the spectral centroid for each frame.
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
# .frames_to_time converts frame to time. time[i] == frame[i].
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')

# ||Spectral Rolloff|| -- frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies.
#spectral bandwidth -- width of the band of frequencies present in the sound
spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

#MFCCs Mel-Ferequency Cepstral Coefficients
mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)
#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

#Chroma feature
#Chroma feature is a powerful feature that captures the harmonic and melodic characteristics of an audio signal.
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=512)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')







