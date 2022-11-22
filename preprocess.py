import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import librosa
#import IPython.display as ipd
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
#import tensorflow as tf
from mpi4py import MPI

comm = MPI.COMM_WORLD

my_rank = comm.Get_rank()
num_processes = comm.Get_size()

# Dataset location
SOURCE_PATH = 'wav_samples'

# Path to labels and processed data file, json format.
JSON_PATH = 'mfcc_genres{}.json'.format(my_rank)

# Sampling rate.
sr = 44100

# Let's make sure all files have the same amount of samples and pick a duration equal to 30 seconds.
TOTAL_SAMPLES = 30 * sr

genres = 'Ambient Classical Country Dance Electronic Experimental Folk HipHop Jazz Pop Psychedelia Punk RNB Rock'.split()

# X amount of slices => X times more training examples.
NUM_SLICES = 9
SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)

def preprocess_data(source_path, json_path, norm=False, deltas=False):

    # Let's create a dictionary of labels and processed data.
    mydict = {
        "labels": [],
        "mfcc": []
        }

    # Let's browse each file, slice it and generate the 13 band mfcc for each slice.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(source_path)):
        for file in filenames:
            # exclude a corrupted file that makes everything crash.
            if file != '.DS_Store':
                song, sr = librosa.load(os.path.join(dirpath, file), duration=30,sr=44100)
                for s in range(NUM_SLICES):
                    start_sample = SAMPLES_PER_SLICE * s
                    end_sample = start_sample + SAMPLES_PER_SLICE
                    mfcc = librosa.feature.mfcc(y=song[start_sample:end_sample], sr=sr, n_mfcc=13)
                    mfcc = mfcc.T
                    if norm:
                        mfcc = normalize(mfcc)
                    if deltas:
                        delta_mfcc = librosa.feature.delta(mfcc)
                        delta2_mfcc = librosa.feature.delta(mfcc,order=2)
                        if norm:
                            delta_mfcc = normalize(delta_mfcc)
                            delta2_mfcc = normalize(delta2_mfcc)
                        mfcc = np.concatenate((mfcc, delta_mfcc, delta2_mfcc))
                    label = genres.index(os.path.join(dirpath, file).split('/')[1])
                    mydict["labels"].append(label)
                    mydict["mfcc"].append(mfcc.tolist())
                    print('Process {}: Just added {} to dictionary'.format(my_rank, file))
                    #ipd.display(ipd.Audio(y=song[start_sample:end_sample], rate=sr))
                    print(os.path.join(dirpath, file), label, mfcc.shape)
            else:
                pass
    # Let's write the dictionary in a json file.    
    with open(json_path, 'w') as f:
        json.dump(mydict, f)
    f.close()
    
if my_rank == 0:
	norm = False
	deltas = False

if my_rank == 1:
	norm = False
	deltas = True
	
if my_rank == 2:
	norm = True
	deltas = False
	
if my_rank == 3:
	norm = True
	deltas = True

preprocess_data(SOURCE_PATH, JSON_PATH, norm=norm, deltas=deltas)

MPI.Finalize
