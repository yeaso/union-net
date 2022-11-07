"""
union-net test cifar100, best to 0.7769.
"""
from __future__ import print_function
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
from keras.callbacks import CSVLogger, EarlyStopping
import os
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import keras
from keras.regularizers import l2
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

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
from sklearn.metrics import roc_curve, auc
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

from keras import optimizers
from keras.activations import elu
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation


from keras import backend as K
from keras import layers
from keras.callbacks import ReduceLROnPlateau



lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=3, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0)

opt = optimizers.Nadam(lr=0.001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
reg=0.0001



from keras.activations import elu



batch_size = 128
nb_classes = 100
nb_epoch = 100
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

chanDim=-1
		      
my_input = Input(shape=(32,32,3),name='my_input')

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
m11= Activation('elu')(m1)

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
x = layers.Conv2D(512, 3, activation="elu", padding="same")(x)
x = layers.BatchNormalization(axis=chanDim)(x)
#x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
#x = layers.BatchNormalization(axis=chanDim)(x)
#x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
#x = layers.BatchNormalization(axis=chanDim)(x)
#x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
#x = layers.BatchNormalization(axis=chanDim)(x)



#x = layers.MaxPooling2D(2)(x)
#x = layers.Dropout(0.5)(x)
x = layers.GlobalAveragePooling2D()(x) 
		
		# softmax classifier
		
output = layers.Dense(100,activation='softmax')(x)
#model = keras.Model(inputs, outputs, name="toy_resnet")

model = Model(inputs=my_input,outputs=output)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
H =model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=nb_epoch, verbose=1, max_q_size=100,
                        callbacks=[lr_reducer])



model.save("modelc100.h5")



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


plt.show()
