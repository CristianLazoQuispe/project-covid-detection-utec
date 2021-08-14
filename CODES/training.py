# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../RESOURCES')

from loss_functions import * 
from metrics import * 

import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks import  EarlyStopping,ModelCheckpoint


from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as k
import tensorflow as tf

import numpy as np
from numpy import expand_dims

import matplotlib.pyplot as plt



WIDTH = 512
HEIGHT = 512


PATH_DATASET = '../DATASET/NEW_DATASET/MODELAMIENTO/PA'
EPOCHS = 1000
model_name = "../RESULTS/model_"+str(EPOCHS)+".h5"
batch_size = 16






from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.xception import Xception
from keras.layers.merge import concatenate

#base_model = VGG16(weights='imagenet', include_top=False,input_shape=(HEIGHT, WIDTH,3))
#base_model = DenseNet121(weights='imagenet', include_top=False,input_shape=(HEIGHT, WIDTH,3))
#base_model = ResNet50V2(weights='imagenet', include_top=False,input_shape=(HEIGHT, WIDTH,3))
#base_model = MobileNetV2(weights='imagenet', include_top=False,input_shape=(HEIGHT, WIDTH,3))
base_model = DenseNet121(weights='imagenet', include_top=False,input_shape=(HEIGHT, WIDTH,3))
#base_model = DenseNet121(weights='imagenet', include_top=False,input_shape=(HEIGHT, WIDTH,3))

model = tf.keras.models.Sequential()

model.add(base_model)

model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.layers[0].trainable = False

model.compile(
    #loss= 'binary_crossentropy',#
    loss =['categorical_crossentropy'],#categorical_focal_loss
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #metrics=[tf.keras.metrics.AUC()]
    metrics=[tf.keras.metrics.AUC(),'accuracy',custom_f1]

)

model.summary()





# load the image
img = load_img(os.path.join(PATH_DATASET,'TEST/COVID/COVID-CXNet_281.jpg'))
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
    rotation_range=10,
    shear_range=0.1,
    brightness_range=[0.4,1.25],
    horizontal_flip=True,
    #vertical_flip=True,
    #preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)# prepare iterator
it = train_datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0]#.astype('uint8')
    # plot raw pixel data
    plt.imshow(image)
# show the figure
plt.show()




train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
    rotation_range=10,
    shear_range=0.1,
    brightness_range=[0.4,1.25],
    horizontal_flip=True,
    #vertical_flip=True,
    #preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=10,
    shear_range=0.1,
    brightness_range=[0.4,1.25],
    horizontal_flip=True,
    #preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)


train_generator = train_datagen.flow_from_directory(
        os.path.join(PATH_DATASET,'TRAIN'),  # this is the target directory
        target_size=(HEIGHT, WIDTH),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        os.path.join(PATH_DATASET,'VALIDATION'),
        target_size=(HEIGHT, WIDTH),
        batch_size=batch_size,
        class_mode='categorical')
        

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss',mode='min')


#'''
history = model.fit_generator(
        train_generator,
        steps_per_epoch= len(train_generator),
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps= len(validation_generator),
    callbacks=[earlyStopping, mcp_save])


model.save(model_name)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

score_f1= history.history['custom_f1']
val_score_f1= history.history['val_custom_f1']

score_auc= history.history['auc']
val_score_auc= history.history['val_auc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(history.epoch) + 1)


fig_model_performance = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(epochs_range, acc, label='Training')
plt.plot(epochs_range, val_acc, label='Validation')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('Model accuracy')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.tight_layout()
#plt.show()

#plt.figure(figsize=(10,5))
plt.subplot(2,2,2)

plt.plot(epochs_range, loss, label='Training')
plt.plot(epochs_range, val_loss, label='Validation')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.tight_layout()

plt.subplot(2,2,3)
plt.plot(epochs_range, score_f1, label='Training')
plt.plot(epochs_range, val_score_f1, label='Validation')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('score_f1')
plt.title('Model score_f1')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.tight_layout()
#plt.show()

plt.subplot(2,2,4)
plt.plot(epochs_range, score_auc, label='Training')
plt.plot(epochs_range, val_score_auc, label='Validation')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('score_auc')
plt.title('Model score_auc')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.tight_layout()
#plt.show()
plt.show()

fig_model_performance.savefig("../RESULTS/metricas_modelo_"+str(EPOCHS)+".png", dpi=fig_model_performance.dpi)
#'''