from django.http.response import JsonResponse
from django.shortcuts import render
import numpy as np
import librosa as lb
import tensorflow as tf
from tensorflow.keras.models import load_model

# Create your views here.
def home(request):
    context = {
        "test" : "World"
    }
    return render(request, "main/index.html",context)

def getMusic(request):
    if request.is_ajax and request.method == "POST":
        try:
            print("File: ", request.FILES["music"])
            files = request.FILES["music"]
            print("File Received", files)
            prediction = predict(files)
            print("Prediction: ", prediction, type(prediction))
            return JsonResponse({"data" : "Success", "value" : prediction})
        except:
            print("Error")
            return JsonResponse({"data" : "Failed"})

def predict(file):
    print("Predict Function")
    # Path to the model
    model_path = 'model.h5'
    print("Predict Function - File Path Loaded")
    # Classes
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    # Path to file
    file_path = '/home/karan/Documents/Masters/SEM I/Machine Learning/Project/genres/rock/rock.00000.wav'

    # Loading the saved and trained model
    cnn = load_model(model_path)
    print("Predict Function - CNN Model Loaded")
    # print(cnn.summary())

    # Preprocessing the data before making prediction
    print("Predict Function - File Path - ", file)
    y, sr = lb.core.load(file)
    print("Predict Function - File Loaded")

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
    return str(classes[np.argmax(prediction)])
