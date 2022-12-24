# 输入必需的模块。
from keras import layers
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras import backend as K


class Union_Net:
	@staticmethod
	def build(width, height, depth, classes, kernel_reg, kernel_init):
#width ：图像宽度（以像素为单位）、height ：图像高度、 depth ：图像的通道数、classes ：分类数、kernel_reg : 正则化方法(L1或L2)、kernel_init ：内核初始化程序。
		inputShape = (height, width, depth)


		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)


		
		      
		my_input = Input(shape=inputShape,name='my_input')

		d1 = layers.Conv2D(32, 3, activation="relu", padding="same")(my_input)
		d1 = layers.BatchNormalization(axis=-1)(d1)
		d1 = layers.Conv2D(32, 3, activation="relu", padding="same")(d1)
		d1 = layers.BatchNormalization(axis=-1)(d1)
		d1 = layers.Conv2D(32, 3, activation="relu", padding="same")(d1)
		d1 = layers.BatchNormalization(axis=-1)(d1)
		d1 = layers.Conv2D(32, 3, activation="relu", padding="same")(d1)
		d1 = layers.BatchNormalization(axis=-1)(d1)
		d1 = layers.MaxPooling2D(2)(d1)


		c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(my_input)
		c1 = layers.BatchNormalization(axis=-1)(c1)
		c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
		c1 = layers.BatchNormalization(axis=-1)(c1)
		c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
		c1 = layers.BatchNormalization(axis=-1)(c1)
		c1 = layers.MaxPooling2D(2)(c1)
		b1 = layers.Conv2D(32, 3, activation="relu", padding="same")(my_input)
		b1 = layers.BatchNormalization(axis=-1)(b1)
		b1 = layers.Conv2D(32, 3, activation="relu", padding="same")(b1)
		b1 = layers.BatchNormalization(axis=-1)(b1)
		b1 = layers.MaxPooling2D(2)(b1)
		a1 = layers.Conv2D(32, 3, activation="relu", padding="same")(my_input)
		a1 = layers.BatchNormalization(axis=-1)(a1)
		a1= layers.MaxPooling2D(2)(a1)
		m1=layers.add([a1, b1, c1, d1])

		#m1 = layers.Dropout(0.5)(m1)
		m11= Activation('relu')(m1)

		d2 = layers.Conv2D(64, 3, activation="relu", padding="same")(m11)
		d2 = layers.BatchNormalization(axis=-1)(d2)
		d2 = layers.Conv2D(64, 3, activation="relu", padding="same")(d2)
		d2 = layers.BatchNormalization(axis=-1)(d2)
		d2 = layers.Conv2D(64, 3, activation="relu", padding="same")(d2)
		d2 = layers.BatchNormalization(axis=-1)(d2)
		d2 = layers.Conv2D(64, 3, activation="relu", padding="same")(d2)
		d2 = layers.BatchNormalization(axis=-1)(d2)

		c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(m11)
		c2 = layers.BatchNormalization(axis=-1)(c2)
		c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
		c2 = layers.BatchNormalization(axis=-1)(c2)
		c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
		c2 = layers.BatchNormalization(axis=-1)(c2)

		b2 = layers.Conv2D(64, 3, activation="relu", padding="same")(m11)
		b2 = layers.BatchNormalization(axis=-1)(b2)
		b2 = layers.Conv2D(64, 3, activation="relu", padding="same")(b2)
		b2 = layers.BatchNormalization(axis=-1)(b2)

		a2 = layers.Conv2D(64, 3, activation="relu", padding="same")(m11)
		a2 = layers.BatchNormalization(axis=-1)(a2)

		m2=layers.add([a2, b2, c2, d2])
		m22 = Activation('relu')(m2)

		d3 = layers.Conv2D(128, 3, activation="relu", padding="same")(m22)
		d3 = layers.BatchNormalization(axis=-1)(d3)
		d3 = layers.Conv2D(128, 3, activation="relu", padding="same")(d3)
		d3 = layers.BatchNormalization(axis=-1)(d3)
		d3 = layers.Conv2D(128, 3, activation="relu", padding="same")(d3)
		d3 = layers.BatchNormalization(axis=-1)(d3)
		d3 = layers.Conv2D(128, 3, activation="relu", padding="same")(d3)
		d3 = layers.BatchNormalization(axis=-1)(d3)


		c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(m22)
		c3 = layers.BatchNormalization(axis=-1)(c3)
		c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
		c3 = layers.BatchNormalization(axis=-1)(c3)
		c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
		c3 = layers.BatchNormalization(axis=-1)(c3)

		b3 = layers.Conv2D(128, 3, activation="relu", padding="same")(m22)
		b3 = layers.BatchNormalization(axis=-1)(b3)
		b3 = layers.Conv2D(128, 3, activation="relu", padding="same")(b3)
		b3 = layers.BatchNormalization(axis=-1)(b3)

		a3 = layers.Conv2D(128, 3, activation="relu", padding="same")(m22)
		a3 = layers.BatchNormalization(axis=-1)(a3)

		m3=layers.add([a3, b3, c3, d3])
		n1=layers.Conv2D(128, 1, padding='same')(m1)
		n2=layers.Conv2D(128, 1, padding='same')(m2)
		n3=layers.Conv2D(128, 1, padding='same')(m3)
		x=layers.add([n1, n2, n3])
		x = Activation('relu')(x)

		x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
		x = layers.BatchNormalization(axis=-1)(x)

		x = layers.MaxPooling2D(2)(x)

		x = layers.GlobalAveragePooling2D()(x) 

		
		# softmax 分类器


		
		output = layers.Dense(classes,activation='softmax')(x)

		model = Model(inputs=my_input,outputs=output)

		return model
