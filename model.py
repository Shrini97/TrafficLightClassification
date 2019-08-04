from __future__ import print_function

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Flatten,Conv2D
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input

import keras

def Classifier(inputs,n_classes):
	base_model = ResNet50(input_tensor=inputs,weights='imagenet',include_top =False)
	layer1 = base_model.output
	layer3 = GlobalAveragePooling2D()(layer1)
	layer4 = Dense(n_classes,activation = 'sigmoid')(layer3)
	# layer4 = Dense(n_classes,activation = 'softmax')(layer3)

	model = Model(inputs =base_model.input,outputs=layer4)
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model
