# USAGE
# python train_model.py

# set the matplotlib backend so figures can be saved in the background
#import matplotlib
#matplotlib.use("Agg")
import os
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import keras
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.utils import plot_model
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from pyimagesearch.cancernet import CancerNet
from pyconfig import configt
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

#from ranger import Ranger
#import torch as t
#from torch import optim
# CONV => RELU => POOL
chanDim=-1
		      
my_input = Input(shape=(48,48,3),name='my_input')

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

x = layers.Conv2D(128, 3, activation="elu", padding="same")(x)
x = layers.BatchNormalization(axis=chanDim)(x)
#x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
#x = layers.BatchNormalization(axis=chanDim)(x)
#x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
#x = layers.BatchNormalization(axis=chanDim)(x)
#x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
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


		
output = layers.Dense(17,activation='softmax')(x)
#model = keras.Model(inputs, outputs, name="toy_resnet")



model = Model(inputs=my_input,outputs=output)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = 37
INIT_LR = 1e-2
BS = 32

# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(configt.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(configt.VAL_PATH)))
totalTest = len(list(paths.list_images(configt.TEST_PATH)))

# account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.2,
	height_shift_range=0.3,
	shear_range=0.1,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	configt.TRAIN_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	configt.VAL_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS )

# initialize the testing generator
testGen = valAug.flow_from_directory(
	configt.TEST_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize our CancerNet model and compile it

#opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
#opt= Ranger(params=CancerNet.parameters(), lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0)
#opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=0.5, decay=INIT_LR / NUM_EPOCHS)
#opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
#opt = optimizers.Adadelta(lr=0.01, rho=0.95, epsilon=0.05, decay=1e-2)
#opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=0.5, decay=INIT_LR/NUM_EPOCHS)
#opt = optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt = optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)


reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=3, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0)
# 绘制训练过程中的损失曲线和精度曲线


model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])



H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain*16 // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	class_weight=classWeight,
        callbacks=[reduce_lr], 
	epochs=NUM_EPOCHS)

# reset the testing generator and then use our trained model to
# make predictions on the data
model.summary()
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
model.save("model17flower9696.h5")
import seaborn as sns
predIdxs = np.argmax(predIdxs, axis=1)
print(confusion_matrix(testGen.classes, predIdxs))
# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys(), digits=4))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity


import matplotlib.pyplot as plt

acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
loss = H.history['loss']
val_loss = H.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b:', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.plot(epochs, loss, 'b--', label='Training loss')
plt.plot(epochs, val_loss, 'r-.', label='Validation loss')
plt.title('Training Loss and Accuracy on Dataset')
plt.legend()

# === 混淆矩阵：真实值与预测值的对比 ===
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
con_mat = confusion_matrix(testGen.classes, predIdxs)

con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化

con_mat_norm = np.around(con_mat_norm, decimals=4)

# === plot ===
plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')

plt.ylim(0, 8)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


