"""
union-net test fashion_mnist, best to 0.9755.
"""
import os
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import keras

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.utils import plot_model

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

from keras import optimizers

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras import layers
from keras.callbacks import ReduceLROnPlateau
from keras.activations import elu
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import backend as K
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2

# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 100
INIT_LR = 1e-2
BS = 128

# grab the Fashion MNIST dataset (if this is your first time running
# this the dataset will be automatically downloaded)
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# if we are using "channels first" ordering, then reshape the design
# matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
	trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
	testX = testX.reshape((testX.shape[0], 1, 28, 28))
 
# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
 
# scale data to the range of [0, 1]
trainX = trainX.astype("float64") / 255.0
testX = testX.astype("float64") / 255.0

# one-hot encode the training and testing labels
trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

# initialize the label names
labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)


from keras.activations import elu

chanDim=-1
		      
my_input = Input(shape=(28,28,1),name='my_input')

d1 = layers.Conv2D(32, 3, activation="elu", padding="same")(my_input)
d1 = layers.BatchNormalization(axis=chanDim)(d1)
d1 = layers.Conv2D(32, 3, activation="elu", padding="same")(d1)
d1 = layers.BatchNormalization(axis=chanDim)(d1)
d1 = layers.Conv2D(32, 3, activation="elu", padding="same")(d1)
d1 = layers.BatchNormalization(axis=chanDim)(d1)
d1 = layers.Conv2D(32, 3, activation="elu", padding="same")(d1)
d1 = layers.BatchNormalization(axis=chanDim)(d1)
d1 = layers.MaxPooling2D(2)(d1)


a1 = layers.Conv2D(32, 3, activation="elu", padding="same")(my_input)
a1 = layers.BatchNormalization(axis=chanDim)(a1)
a1 = layers.Conv2D(32, 3, activation="elu", padding="same")(a1)
a1 = layers.BatchNormalization(axis=chanDim)(a1)
a1 = layers.Conv2D(32, 3, activation="elu", padding="same")(a1)
a1 = layers.BatchNormalization(axis=chanDim)(a1)
a1 = layers.MaxPooling2D(2)(a1)
b1 = layers.Conv2D(32, 3, activation="elu", padding="same")(my_input)
b1 = layers.BatchNormalization(axis=chanDim)(b1)
b1 = layers.Conv2D(32, 3, activation="elu", padding="same")(b1)
b1 = layers.BatchNormalization(axis=chanDim)(b1)
b1 = layers.MaxPooling2D(2)(b1)
c1 = layers.Conv2D(32, 3, activation="elu", padding="same")(my_input)
c1 = layers.BatchNormalization(axis=chanDim)(c1)
c1= layers.MaxPooling2D(2)(c1)
m1=layers.add([a1, b1, c1, d1])

#m1 = layers.Dropout(0.5)(m1)
m11= Activation('relu')(m1)

d2 = layers.Conv2D(64, 3, activation="elu", padding="same")(m11)
d2 = layers.BatchNormalization(axis=chanDim)(d2)
d2 = layers.Conv2D(64, 3, activation="elu", padding="same")(d2)
d2 = layers.BatchNormalization(axis=chanDim)(d2)
d2 = layers.Conv2D(64, 3, activation="elu", padding="same")(d2)
d2 = layers.BatchNormalization(axis=chanDim)(d2)
d2 = layers.Conv2D(64, 3, activation="elu", padding="same")(d2)
d2 = layers.BatchNormalization(axis=chanDim)(d2)

c2 = layers.Conv2D(64, 3, activation="elu", padding="same")(m11)
c2 = layers.BatchNormalization(axis=chanDim)(c2)
c2 = layers.Conv2D(64, 3, activation="elu", padding="same")(c2)
c2 = layers.BatchNormalization(axis=chanDim)(c2)
c2 = layers.Conv2D(64, 3, activation="elu", padding="same")(c2)
c2 = layers.BatchNormalization(axis=chanDim)(c2)

b2 = layers.Conv2D(64, 3, activation="elu", padding="same")(m11)
b2 = layers.BatchNormalization(axis=chanDim)(b2)
b2 = layers.Conv2D(64, 3, activation="elu", padding="same")(b2)
b2 = layers.BatchNormalization(axis=chanDim)(b2)

a2 = layers.Conv2D(64, 3, activation="elu", padding="same")(m11)
a2 = layers.BatchNormalization(axis=chanDim)(a2)

m2=layers.add([a2, b2, c2, d2])
m22 = Activation('elu')(m2)

d3 = layers.Conv2D(128, 3, activation="elu", padding="same")(m22)
d3 = layers.BatchNormalization(axis=chanDim)(d3)
d3 = layers.Conv2D(128, 3, activation="elu", padding="same")(d3)
d3 = layers.BatchNormalization(axis=chanDim)(d3)
d3 = layers.Conv2D(128, 3, activation="elu", padding="same")(d3)
d3 = layers.BatchNormalization(axis=chanDim)(d3)
d3 = layers.Conv2D(128, 3, activation="elu", padding="same")(d3)
d3 = layers.BatchNormalization(axis=chanDim)(d3)


a3 = layers.Conv2D(128, 3, activation="elu", padding="same")(m22)
a3 = layers.BatchNormalization(axis=chanDim)(a3)
a3 = layers.Conv2D(128, 3, activation="elu", padding="same")(a3)
a3 = layers.BatchNormalization(axis=chanDim)(a3)
a3 = layers.Conv2D(128, 3, activation="elu", padding="same")(a3)
a3 = layers.BatchNormalization(axis=chanDim)(a3)

b3 = layers.Conv2D(128, 3, activation="elu", padding="same")(m22)
b3 = layers.BatchNormalization(axis=chanDim)(b3)
b3 = layers.Conv2D(128, 3, activation="elu", padding="same")(b3)
b3 = layers.BatchNormalization(axis=chanDim)(b3)

c3 = layers.Conv2D(128, 3, activation="elu", padding="same")(m22)
c3 = layers.BatchNormalization(axis=chanDim)(c3)

m3=layers.add([a3, b3, c3, d3])
n1=layers.Conv2D(128, 1, padding='same')(m1)
n2=layers.Conv2D(128, 1, padding='same')(m2)
n3=layers.Conv2D(128, 1, padding='same')(m3)
x=layers.add([n1, n2, n3])
x = Activation('elu')(x)

x = layers.Conv2D(256, 3, activation="elu", padding="same")(x)
x = layers.BatchNormalization(axis=chanDim)(x)
#x = layers.Conv2D(512, 3, activation="relu", padding="same")(x)
#x = layers.BatchNormalization(axis=chanDim)(x)


x = layers.MaxPooling2D(2)(x)
x = layers.MaxPooling2D(2)(x)
#x = layers.MaxPooling2D(2)(x)
#x = layers.Dropout(0.5)(x)
x = layers.GlobalAveragePooling2D()(x) 
#x = layers.Flatten()(x)
#x = layers.Dense(128,activation='relu')(x)
#x = layers.BatchNormalization(axis=chanDim)(x)
#x = layers.Dropout(0.5)(x)
#x = layers.BatchNormalization(axis=chanDim)(x)


		
		# softmax classifier


		
output = layers.Dense(10,activation='softmax')(x)




model = Model(inputs=my_input,outputs=output)


model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training model...")
H = model.fit(trainX, trainY,
	validation_data=(testX, testY),
	batch_size=BS, epochs=NUM_EPOCHS)

# make predictions on the test set
preds = model.predict(testX)

# show a nicely formatted classification report
print("[INFO] evaluating network...")
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
	target_names=labelNames))

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

# initialize our list of output images
images = []

# randomly select a few testing fashion items
for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
	# classify the clothing
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = labelNames[prediction[0]]
 
	# extract the image from the testData if using "channels_first"
	# ordering
	if K.image_data_format() == "channels_first":
		image = (testX[i][0] * 255).astype("uint8")
 
	# otherwise we are using "channels_last" ordering
	else:
		image = (testX[i] * 255).astype("uint8")

	# initialize the text label color as green (correct)
	color = (0, 255, 0)

	# otherwise, the class label prediction is incorrect
	if prediction[0] != np.argmax(testY[i]):
		color = (0, 0, 255)
 
	# merge the channels into one image and resize the image from
	# 28x28 to 96x96 so we can better see it and then draw the
	# predicted label on the image
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
		color, 2)

	# add the image to our list of output images
	images.append(image)

# construct the montage for the images
montage = build_montages(images, (96, 96), (4, 4))[0]

# show the output montage
cv2.imshow("Fashion MNIST", montage)
cv2.waitKey(0)
