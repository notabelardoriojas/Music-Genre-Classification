import librosa 
import tensorflow as tf
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import soundfile as sf
import skimage.io

#load our mashup
signal, sr = librosa.load('mashup_10.flac', sr=44100)
#get the segments
segments = np.split(signal, 32)
for i, seg in enumerate(segments):
	sf.write(f'segs/{i}.flac', data=seg, samplerate=44100)
#generate melspecs for each segment	
melspecs = []

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled
    
def save_spec(audio, out='test.png', sr=44100, n_fft=1024, hop_length=512, n_mels=128, deltas=False, mfcc=False,index=0):
	if mfcc == False:
		S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
		S = np.log(S + 1e-9) # add small number to avoid log(0)

		# min-max scale to fit inside 8-bit range
		S = scale_minmax(S, 0, 255).astype(np.uint8)
		np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
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
	melspecs.append(S)
	skimage.io.imsave(f'segs/{index}.png', S)
	
index=0
deltas=False
mfcc = True
for seg in segments:
	save_spec(seg, deltas=deltas, mfcc=mfcc,index=index)
	index+=1
	

melspecs = np.stack(melspecs,axis=0)
shape = melspecs.shape
if deltas==False:
	melspecs = melspecs.reshape(shape[0], shape[1], shape[2], 1)
print(melspecs.shape)
melspecs = np.split(melspecs, 2)
	
model = tf.keras.models.load_model(f'saved_model_multi_0_0_1_3/my_model')

# Check its architecture
model.summary()
genres = 'Ambient Classical Country Dance Electronic Experimental Folk HipHop Jazz Pop Psychedelia Punk RNB Rock'.split()
for specs in melspecs:
	preds = model.predict(specs)
	for pred in preds:
		ind = np.argwhere(pred>.5)
		for idx in ind:
			idx = idx[0]
			print(genres[idx])
		print(pred)
		print('------')


	

