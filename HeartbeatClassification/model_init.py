import tensorflow as tf
from python_speech_features import mfcc as psfmfcc
import numpy as np

class Model:
    def __init__(self, filepath, save_type):
        self.fold_name = filepath
        self.model_dir = './models/' + self.fold_name
        # Loading the saved model from directory
        if (save_type == 'pb'):
            self.model = tf.saved_model.load(self.model_dir)
        elif (save_type == 'h5'):
            self.model = tf.keras.models.load_model(self.model_dir)

    # Preprocess and run audio through model, returns a map to results
    def process_audio(self, wav, rate):
        nfeat=13
        nfilt=26
        nfft=2048
        sr=16000
        step = int(sr/10)
        
        y_prob = []
        for i in range(0, wav.shape[0]-step, step):
            sample = wav[i:i+step]
            X_sample = psfmfcc(sample, rate, numcep=nfeat, nfilt=nfilt, nfft=nfft).T
            if self.fold_name== "cnn":
                X_sample = psfmfcc(sample, rate, numcep=40, nfilt=nfilt, nfft=nfft).T
                X_sample = tf.reshape(X_sample, (-1, X_sample.shape[0], X_sample.shape[1], 1))
            else:
                X_sample = X_sample.T
                X_sample = tf.reshape(X_sample, (-1, X_sample.shape[0], X_sample.shape[1]))
            X_sample = tf.cast(X_sample, dtype=tf.float32)
            y_hat = self.model(X_sample)
            y_prob.append(y_hat.numpy())
        result = np.mean(y_prob, axis=0).flatten()
        result *= 100
        
        mapped_result = map_result(result)
        max_result = max(mapped_result, key=mapped_result.get)
        # Convert result to a numpy array
        return mapped_result, max_result
    
# Converts numpy_array results into mapped data with labels
def map_result(np_arr):

    # 4 values which represent what heartbeat sounds our model classifies
    chars = ['Artifact', 'Extra Heart Sound', 'Murmur', 'Normal']

    mapping = {}

    for (key, value) in zip(chars, np_arr):
        mapping[key] = value.item()

    return mapping