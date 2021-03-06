# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 15:35:25 2018
@author: Nishit Mehta, Kshitij Shah, Samraddhi Gupta
"""
#%%
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
from pylab import imshow
import numpy as np
import os
from os import path
import pickle
from keras.preprocessing import image
from helper import plot_images, get_class_names, predict_classes, plot_model
#%%
IMAGE_SIZE = 100
LR = 1e-1
#%%
# Renaming
path =  os.getcwd()
path1 = os.path.join(path, 'test_set')
path = os.path.join(path, 'training_set')
filenames = os.listdir(path)
filenames1 = os.listdir(path1)
i = 0
# FOr training Set
for filename in filenames:
    temp = os.path.join(path, filename)
    sub_filenames = os.listdir(os.path.join(path, filename))
    i = 0
    for sub_filename in sub_filenames:
        temp1 = os.path.join(temp, sub_filename)
        fn, extension = os.path.splitext(temp1)
        temp1 = fn + extension
        newfilename = os.path.join(temp, filename + str(i) + ".jpg")
        if not os.path.exists(newfilename):
            os.rename(temp1, newfilename)
        i+=1
        #os.rename(os.path.join(temp, os.path.join(fn, extension)),os.path.join(temp, fn))

i = 0
# For test Set
for filename in filenames1:
    temp = os.path.join(path1, filename)
    sub_filenames = os.listdir(os.path.join(path1, filename))
    i = 0
    for sub_filename in sub_filenames:
        temp1 = os.path.join(temp, sub_filename)
        fn, extension = os.path.splitext(temp1)
        temp1 = fn + extension
        newfilename = os.path.join(temp, filename + str(i) + ".jpg")
        if not os.path.exists(newfilename):
            os.rename(temp1, newfilename)
        i+=1
        
#%%

classifier = Sequential()

# Convolutional + MaxPooling -> 1
classifier.add(Conv2D(32, (3,3), input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.2))

# Convolutional + MaxPooling -> 2
classifier.add(Conv2D(64, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.2))

# Convolutional + MaxPooling -> 3
classifier.add(Conv2D(64, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

# Convolutional + MaxPooling -> 4
classifier.add(Conv2D(32, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 128, activation = 'relu'))

#OUTPUT Layer
classifier.add(Dense(units = 5, activation = 'softmax'))

#Compile
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(classifier.summary())
#%%
# preprocessing

train_data = ImageDataGenerator(rescale = 1./255, zca_whitening=False, rotation_range=20, shear_range = 0.2, zoom_range = 0.2, horizontal_flip=True,
    vertical_flip=True)
test_data = ImageDataGenerator(rescale = 1./255)
training_set = train_data.flow_from_directory('training_set',
target_size = (IMAGE_SIZE, IMAGE_SIZE),
batch_size = 64,
class_mode = 'categorical')
test_set = test_data.flow_from_directory('test_set',
target_size = (IMAGE_SIZE, IMAGE_SIZE),
batch_size = 64,
class_mode = 'categorical')

#%%
# Checkpoints
checkpoint = ModelCheckpoint('checkpoints/best_model_improved_cp.h5',  # model filename
                             monitor='val_acc', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto')

#%% Training
model_hist = classifier.fit_generator(training_set,
steps_per_epoch = 100,
epochs = 100,
validation_data = test_set,
validation_steps = 100,
callbacks=[checkpoint],
workers = 16)
#%% Saving Model and History
classifier.save('trained_model')
pickle.dump(model_hist.history, open("trainHistoryDict/save.p", "wb"))
#%% Loading Model and History
classifier = load_model('trained_model')
model_hist = pickle.load(open("trainHistoryDict/save.p", "rb"))
#%%
test_image1 = image.load_img('keyboard_test.jpg', target_size = (IMAGE_SIZE, IMAGE_SIZE))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis = 0)
result = classifier.predict(test_image1, batch_size = 1, verbose = 0)
print(result)
result = classifier.predict_classes(test_image1, batch_size = 1, verbose = 0)
print(result)
#abc = training_set.class_indices
#print(abc)
if result == 0:
    prediction = 'Keyboard'
elif result == 1:
    prediction = 'Laptop'
elif result == 2:
    prediction = 'Mobile Phone'
elif result == 3:
    prediction = 'Mouse'
elif result == 4:
    prediction = 'Watch'
print(prediction)

#%%
# Graphs Plotting
#plt.plot(model.history['acc'])
#plt.plot(model.history['val_acc'])
#print(model_hist.history.keys())
plt.figure(figsize=[8,6])
plt.plot(model_hist['acc'],'r',linewidth=3.0)
plt.plot(model_hist['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)