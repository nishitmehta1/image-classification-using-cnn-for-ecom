"""
Created on Mon Mar 12 10:51:41 2018

@author: Nishit Mehta
"""
#%%

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
from pylab import imshow
import itertools
import numpy as np
from keras.preprocessing import image
from helper import plot_images, get_class_names, predict_classes, plot_model

#%%
IMAGE_SIZE = 50
LR = 1e-1
#%%

classifier = Sequential()

# Convolutional + MaxPooling -> 1
classifier.add(Conv2D(32, (3,3), input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Convolutional + MaxPooling -> 2
classifier.add(Conv2D(64, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Convolutional + MaxPooling -> 3
classifier.add(Conv2D(32, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))

#OUTPUT Layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compile
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(classifier.summary())
#%%
# preprocessing

train_data = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data = ImageDataGenerator(rescale = 1./255)
training_set = train_data.flow_from_directory('training_set',
target_size = (IMAGE_SIZE, IMAGE_SIZE),
batch_size = 32,
class_mode = 'binary')
test_set = test_data.flow_from_directory('test_set',
target_size = (IMAGE_SIZE, IMAGE_SIZE),
batch_size = 32,
class_mode = 'binary')

#%%
# Checkpoints
checkpoint = ModelCheckpoint('best_model_improved_cp.h5',  # model filename
                             monitor='acc', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto')

#%%
model = classifier.fit_generator(training_set,
steps_per_epoch = 120,
epochs = 50,
validation_data = test_set,
validation_steps = 120,
callbacks=[checkpoint],
workers = 16)

#print(model.history.keys())
#plt.plot(model.history['acc'])
#%%
# SAVING AND LOADING MODEL
classifier.save('initial_train_50_epochs')
#%%
model = load_model('initial_train_1_epochs')
'''
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
'''
#%%
test_image1 = image.load_img('mouse_test.jpg', target_size = (IMAGE_SIZE, IMAGE_SIZE))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis = 0)
result = classifier.predict(test_image1, batch_size = 1, verbose = 0)
print(result)
abc = training_set.class_indices
print(abc)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'   
print(prediction)

#%%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%
test_image = image.load_img('c.jpg', target_size = (IMAGE_SIZE, IMAGE_SIZE))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
predict1 = classifier.predict(test_image, batch_size = 1, verbose = 0)
print(predict1)

round_predict = classifier.predict_classes(test_image1, batch_size = 2, verbose=0)

for i in round_predict:
    print(i)
round_predict = np.append(round_predict, '0')
print(round_predict)
    
test_labels = ['dog','cat']
cm_plot_labels = ['cat','dog']

cm = confusion_matrix(test_labels, round_predict)

plot_confusion_matrix(cm, cm_plot_labels, title = "")