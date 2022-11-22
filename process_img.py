import numpy as np
import librosa
import os
import pandas as pd
from tqdm import tqdm
import skimage.io
import sys
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

#librosa gives us an annoying warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
print('Done importing')

header = 'Filename Ambient Classical Country Dance Electronic Experimental Folk HipHop Jazz Pop Psychedelia Punk RNB Rock'.split()


add_noise = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.7),
])
pitch_shift = Compose([
    PitchShift(min_semitones=-4, max_semitones=12, p=0.5),
])

SOURCE_PATH = 'multiclass_flac_samples'
IMG_PATH = 'multiclass_img_samples'

SAMPLE_RATE = 44100
info = []

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled
    
def save_spec(audio, out='test.png', sr=44100, n_fft=1024, hop_length=512, n_mels=128, deltas=False, mfcc=False):
	if mfcc == False:
		S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
		S = np.log(S + 1e-9) # add small number to avoid log(0)

		# min-max scale to fit inside 8-bit range
		S = scale_minmax(S, 0, 255).astype(np.uint8)
		
		if deltas:
			delta_S = librosa.feature.delta(S)
			delta_S = scale_minmax(delta_S, 0, 255).astype(np.uint8) #min-max scale to fit inside 8-bit range
			
			delta2_S = librosa.feature.delta(S, order=2)
			delta2_S = scale_minmax(delta2_S, 0, 255).astype(np.uint8) #min-max scale to fit inside 8-bit range
			
			S = np.dstack((S, delta_S, delta2_S))
	else:
		S = librosa.feature.mfcc(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
		S = scale_minmax(S, 0, 255).astype(np.uint8)
		
		if deltas:
			delta_S = librosa.feature.delta(S)
			delta_S = scale_minmax(delta_S, 0, 255).astype(np.uint8) #min-max scale to fit inside 8-bit range
			
			delta2_S = librosa.feature.delta(S, order=2)
			delta2_S = scale_minmax(delta2_S, 0, 255).astype(np.uint8) #min-max scale to fit inside 8-bit range
			
			S = np.dstack((S, delta_S, delta2_S))
		
		
		
	S = np.flip(S, axis=0) # put low frequencies at the bottom in image
	S = 255-S # invert. make black==more energy

	# save as PNG
	skimage.io.imsave(out, S)
	#print(f'Saved {out}. Size of spec {S.shape}')
	
    


from sklearn.preprocessing import MultiLabelBinarizer
genres = 'Ambient Classical Country Dance Electronic Experimental Folk HipHop Jazz Pop Psychedelia Punk RNB Rock'.split()
mlb = MultiLabelBinarizer(classes=genres)
	


def preprocess_data(source_path, img_path, aug = False, deltas=False, mfcc = False, slices=1):
	
	SAMPLES_PER_SLICE = int((30*SAMPLE_RATE)/slices)
	img_path = f'{img_path}_{int(aug)}_{int(deltas)}_{int(mfcc)}_{slices}'

	for label in tqdm(os.listdir(source_path)):
		folder_path = os.path.join(source_path, label)
		hot_label = mlb.fit_transform([label.split()])[0]
		
		for idx, file in enumerate(os.listdir(folder_path)):
			filepath = os.path.join(folder_path, file)
			
			signal, sr = librosa.load(filepath, sr=SAMPLE_RATE)
			#now we get a 30 second segment from it from the quarter, halfway, and three-quarters mark
			num_samples = len(signal)
			Q1 = int(num_samples*.25)
			Q2 = int(num_samples*.50)
			Q3 = int(num_samples*.75)
			
			window = 5*sr #five seconds
			sample = signal[Q1 - window : Q1 + window]
			sample = np.concatenate((sample, signal[Q2 - window : Q2 + window]), axis=None)
			sample = np.concatenate((sample, signal[Q3 - window : Q3 + window]), axis=None)
			
			for s in range(slices):
				start_sample = SAMPLES_PER_SLICE*s
				end_sample = start_sample + SAMPLES_PER_SLICE
				segment = sample[start_sample:end_sample]
				
				
				lab = label.replace(' ', '_')
				filepath = f'{img_path}/{lab}_{idx}_noaug_{s}.png'
				#save the melspec of the audio
				save_spec(audio=segment, out=filepath, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128, deltas=deltas, mfcc=mfcc)
				row = [0]*15
				row[0] = filepath
				row[1:] = hot_label
				info.append(row)
				
				if aug:
					noisy_audio = add_noise(segment, 44100)
					pitch_audio = pitch_shift(segment, 44100)
					
					filepath = f'{img_path}/{lab}_{idx}_noise_{s}.png'
					save_spec(audio=noisy_audio, out=filepath, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128, deltas=deltas, mfcc=mfcc)
					row = [0]*15
					row[0] = filepath
					row[1:] = hot_label
					info.append(row)
					
					filepath = f'{img_path}/{lab}_{idx}_pitch_{s}.png'
					save_spec(audio=pitch_audio, out=filepath, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128, deltas=deltas, mfcc=mfcc)
					row = [0]*15
					row[0] = filepath
					row[1:] = hot_label
					info.append(row)
					
					
					
					
					



 
# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]

aug = bool(int(argumentList[0]))
deltas = bool(int(argumentList[1]))
mfcc = bool(int(argumentList[2]))
slices = int(argumentList[3])
print(aug, deltas, mfcc, slices)
os.system(f'mkdir multiclass_img_samples_{int(argumentList[0])}_{int(argumentList[1])}_{int(argumentList[2])}_{int(argumentList[3])}')


preprocess_data(SOURCE_PATH, IMG_PATH, aug=aug, deltas=deltas, mfcc=mfcc, slices=slices)
df = pd.DataFrame(info, columns=header)
df.to_csv(f'images_{int(aug)}_{int(deltas)}_{int(mfcc)}_{slices}.csv', index=False)
