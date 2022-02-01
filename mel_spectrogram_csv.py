count = 0
genre = []
filenames = []
music_data = []
mel_dim = []
labels = []
base_directory = 'genres/'
for subdir, dirs, files in os.walk(base_directory):
    for file in files:
        # print(os.path.join(subdir, file))
        # print(subdir)

        # print(file)
        # audio_file = subdir + '/' + file
        # print(os.path.join(subdir, file))
        # print(audio_file)
        # y, sr = librosa.core.load(audio_file)
        # Loading the audio
        # Get the audio data and the sampling rate
        # Rename y to something better
        y, sr = lb.core.load(os.path.join(subdir, file))
        
        
        # Get mel spectrogram from the audio data usinf y and sr
        mel_spec = lb.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        # Change the scale to decibles
        mel_spec = lb.power_to_db(mel_spec, ref=np.max)
        
        # Reshaping to 128x660, by adding 0
        if mel_spec.shape[1] != 660:
            mel_spec.resize(128,660, refcheck=False)
            
        # print(mel_spec.shape)
        
        
        # Converting into a 1d array using flatten
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
        mel_spec = mel_spec.flatten()
        music_data.append(mel_spec)
        
        
        count = count + 1
        # Storing the corresponding filename
        filenames.append(file)
        # labels, removing base_directory name from the labels
        labels.append(subdir.replace(base_directory, ''))
        
print('Files parsed: ', count)

# print(type(music_data))
# Converting list to array
music_data = np.array(music_data)
# print(type(music_data))
# print(music_data.shape)

# Reshapig the lists
labels = np.array(labels).reshape(count,1)
filenames = np.array(filenames).reshape(count,1)

        
# subdir will be the label
# rest of it will be the same
# file will be the file name
