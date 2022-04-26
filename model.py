import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import skimage.io
import sys

argumentList = sys.argv[1:]
aug = bool(int(argumentList[0]))
deltas = bool(int(argumentList[1]))
mfcc = bool(int(argumentList[2]))
slices = int(argumentList[3])
print(aug, deltas, mfcc, slices)
test_img = f'multiclass_img_samples_{int(aug)}_{int(deltas)}_{int(mfcc)}_{slices}/Ambient_0_noaug_0.png'

img = skimage.io.imread(test_img)
shape=img.shape


if deltas==False:
    shape = (shape[0], shape[1], 1)

print(shape)

df = pd.read_csv(f'images_{int(aug)}_{int(deltas)}_{int(mfcc)}_{slices}.csv')
df = df.sample(frac=1)
columns = 'Ambient Classical Country Dance Electronic Experimental Folk HipHop Jazz Pop Psychedelia Punk RNB Rock'.split()

datagen=ImageDataGenerator(rescale=1/255.)
test_datagen=ImageDataGenerator(rescale=1/255.)

n_rows = len(df)
cm = "grayscale"
if deltas:
    cm = "rgb"

train_generator=datagen.flow_from_dataframe(
dataframe=df[:int(n_rows*.8)],
x_col="Filename",
y_col=columns,
batch_size=16,
color_mode = cm,
seed=42,
shuffle=True,
class_mode="raw",
class_mode="raw",
target_size=(shape[0], shape[1]))

val_generator=test_datagen.flow_from_dataframe(
dataframe=df[int(n_rows*.8):int(n_rows*.9)],
x_col="Filename",
y_col=columns,
batch_size=16,
color_mode = cm,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(shape[0], shape[1]))

test_generator=test_datagen.flow_from_dataframe(
dataframe=df[int(n_rows*.9):],
x_col="Filename",
batch_size=1,
seed=42,
shuffle=False,
color_mode = cm,
class_mode=None,
target_size=(shape[0], shape[1]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, BatchNormalization, Flatten, MaxPooling2D

def macro_f1(y, y_hat, thresh=0.5):
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def macro_soft_f1(y, y_hat):
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels

    return macro_cost


model = Sequential()
if mfcc:
    model.add(Conv2D(8, kernel_size=(3,3), strides=(1,1), input_shape = shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2)))

    model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2)))

    model.add(Conv2D(64, kernel_size=(1,3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2)))

    model.add(Conv2D(128, kernel_size=(1,3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=.3))

    model.add(Dense(14, activation = 'sigmoid'))
else:
    model.add(Conv2D(8, kernel_size=(3,3), strides=(1,1), input_shape = shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=.3))

    model.add(Dense(14, activation = 'sigmoid'))



model.compile(optimizer='adam', loss=macro_soft_f1, metrics=[tf.keras.metrics.BinaryAccuracy(), macro_f1])
model.summary()

history = model.fit(x = train_generator, epochs = 50, validation_data=val_generator, verbose=1)


maxi = max(history.history['val_binary_accuracy'])
# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.ylim([0, 1])
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'accuracy_{int(aug)}_{int(deltas)}_{int(mfcc)}_{slices}_{maxi}.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['macro_f1'])
plt.plot(history.history['val_macro_f1'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'loss_{int(aug)}_{int(deltas)}_{int(mfcc)}_{slices}_{maxi}.png')

import os
#save model
os.system(f'mkdir -p saved_model_multi_{int(aug)}_{int(deltas)}_{int(mfcc)}_{slices}')
model.save(f'saved_model_multi_{int(aug)}_{int(deltas)}_{int(mfcc)}_{slices}/my_model')

#confirm it loads
new_model = tf.keras.models.load_model(f'saved_model_multi_{int(aug)}_{int(deltas)}_{int(mfcc)}_{slices}/my_model', custom_objects={'macro_soft_f1': macro_soft_f1, 'macro_f1':macro_f1})

# Check its architecture
new_model.summary()