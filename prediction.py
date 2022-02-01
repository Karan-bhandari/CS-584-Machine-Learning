import numpy as np
import librosa as lb
# import tensorflow as tf
from tensorflow.keras.models import load_model


# Path to the model
model_path = 'model.h5'

# Classes
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Path to file
file_path = 'file.wav'

# Loading the saved and trained model
cnn = load_model(model_path)

print(cnn.summary())

# Preprocessing the data before making prediction
y, sr = lb.core.load(file_path)

mel_spec = lb.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
mel_spec = lb.power_to_db(mel_spec, ref=np.max)

mel_spec.resize(128,660, refcheck=False)

print(mel_spec.shape)

mel_spec = mel_spec.flatten()

X = mel_spec.reshape(-1, 1)

# Scaling the data
X /= -80

# Reshaping 
X = X.reshape(-1, 128, 660, 1)

print(X.shape)

# Make prediction
prediction = cnn.predict(X)

# Probabilities of prediction
print(prediction)

# Get index
print('Index')
print(np.argmax(prediction))

# Print the class name
print(classes[np.argmax(prediction)])
