import tensorflow as tf
import re
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam,SGD
import os
import numpy as np
from PIL import Image ,ImageOps 
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation,Bidirectional, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from math import floor,log10
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint
from keras.regularizers import l2
from scipy import misc


inception_v2 = InceptionResNetV2(include_top=False,pooling = 'avg')
print (inception_v2.summary())

for layer in inception_v2.layers[:-4]:
  layer.trainable = False

inp = Input(shape=(299,299,3))
x = inception_v2(inp)
#x = Flatten()(x)
x = Dense(8, activation='softmax', name='predictions')(x)
model = Model(inputs=inp, outputs=x)



model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

path =  'Dynamic_Image'
X = []
y = []

X_train = []
X_test = []
X_val = []
y_train = []
y_test = []
y_val = []

prev = 0
for f in sorted(os.listdir(path)):
	img_path = path+'/'+f
	if(prev==0):
		prev = int(f[3])
		X.append(misc.imread(img_path))
		y.append(int(f[3])-1)
	elif(prev==int(f[3])):
		X.append(misc.imread(img_path))
		y.append(int(f[3])-1)
	else:
		X = np.array(X)
		tot_size = len(y)
		y= to_categorical(y,8)
 		train_size = int(0.7*tot_size)
 		print (train_size)
 		X_train.extend(X[:train_size,:,:,:])
 		y_train.extend(y[:train_size,:])
 		test_size = (2*(tot_size-train_size))//3
 		X_test.extend(X[train_size:train_size+test_size,:,:,:])
 		y_test.extend(y[train_size:train_size+test_size,:])
 		X_val.extend(X[train_size+test_size:,:,:,:])
 		y_val.extend(y[train_size+test_size:,:])

 		prev = int(f[3])
 		X = []
 		y = []
 		X.append(misc.imread(img_path))
		y.append(int(f[3])-1)


tot_size = len(y)
X = np.array(X)
y= to_categorical(y,8)
train_size = int(0.7*tot_size)
X_train.extend(X[:train_size,:,:,:])
y_train.extend(y[:train_size,:])
test_size = (2*(tot_size-train_size))//3
X_test.extend(X[train_size:train_size+test_size,:,:,:])
y_test.extend(y[train_size:train_size+test_size,:])
X_val.extend(X[train_size+test_size:,:,:,:])
y_val.extend(y[train_size+test_size:,:])

X_train = np.array(X_train)/255
X_test = np.array(X_test)/255
X_val = np.array(X_val)/255
encoded_y_train = np.array(y_train)
encoded_y_test = np.array(y_test)
encoded_y_val = np.array(y_val)	

print (X_train.shape)
print (X_test.shape)
print (X_val.shape)
print (encoded_y_train.shape)


history =model.fit(X_train, encoded_y_train, validation_data = (X_val,encoded_y_val), epochs = 250, batch_size = 32)
score = model.evaluate(X_test,encoded_y_test,batch_size= 32)
print (score)
"""
np.save('split/X_test.npy', np.array(X_test))
np.save('split/Y_test.npy',Y_test)
np.save('split/X_train.npy', np.array(X_train))
np.save('split/Y_train.npy',Y_train)
"""
#graph of loss and acc

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
