from __future__ import print_function

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
# from dense_tensor import DenseTensor, tensor_factorization_low_rank

# from dense_tensor.utils import l1l2
from keras.regularizers import l2
import keras
# from dense_tensor.example_utils import experiment
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Conv2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Layer
import numpy as np
import cPickle
import time
from keras.callbacks import Callback, ModelCheckpoint
import os
import tensorflow as tf
from utils import *
import shutil

def train_vgg_tt(training_generator,testing_generator,steps,path,tt_parameters,nb_classes=10,batch_size=128):
    if not os.path.isdir(path):
        os.mkdir(path)
    infoCallback = EnriquesCallback(testing_generator,steps[1],path,0.01,evaluate=True)
    save_best_callback = ModelCheckpoint(path+'best_model.h5',monitor='val_acc',verbose=1,mode='max',save_best_only=True)
    model = get_vgg_style_net_tt(tt_parameters,input_shape=(32,32,3),blocks=[3,3,3],outNeurons=512,init_filters=64,dropout=True,nb_classes=nb_classes,weightDecay=10.e-4)
    sgd = SGD(lr=0.01, momentum=0.9)
    if os.path.isfile(path +'model.h5'):
        model.load_weights(path+'model.h5', by_name=True)
    #datagen.flow(testing_data[0],testing_data[1],batch_size=128)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    model.summary()
    tmp = model.fit_generator(training_generator,steps_per_epoch=steps[0],epochs=350,callbacks=[save_best_callback,infoCallback],validation_data=testing_generator,validation_steps=steps[1],verbose=1,initial_epoch=len(infoCallback.history['time']))
    
    history = tmp.history
    # history['accuracyTraining'] = tmp.history['acc']
    # history['lossTraining'] = tmp.history['loss']

    history.update(infoCallback.history)
    save_full_model(model,history=history,path=path)
    plot_history(history,path)
    return model

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.as_default()

stringOfHistory = './Run1/'
batch_size = 128
nb_classes = 10
dataset = cifar10
(X_train, y_train), (X_test, y_test) = dataset.load_data() #GET DATA

Y_train = np_utils.to_categorical(y_train, nb_classes).astype(K.floatx())
Y_test = np_utils.to_categorical(y_test, nb_classes).astype(K.floatx())

X_train /= 255
X_test /= 255

if not os.path.isdir(stringOfHistory):
    os.mkdir(stringOfHistory)
if not os.path.isfile(stringOfHistory+'run_file.py'):
    shutil.copyfile('./train_tensor_net.py',stringOfHistory+'run_file.py')

steps_training = np.ceil(1.*len(X_train)/batch_size).astype(int)
steps_testing = np.ceil(1.*len(X_test)/batch_size).astype(int)
X_train = X_train.astype(K.floatx())
X_test = X_test.astype(K.floatx())

datagen = ImageDataGenerator(featurewise_center=True,  #MEAN 0
                             featurewise_std_normalization=True,  #STD 1
                             zca_whitening=True,  #WHITENING
                             rotation_range=False,  # randomly rotate images in the range (degrees, 0 to 180)
                             width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                             horizontal_flip=True,  # randomly flip images
                             vertical_flip=False)

datagen.fit(X_train)

training_generator = datagen.flow(X_train,Y_train,batch_size=batch_size)
testing_generator = datagen.flow(X_test,Y_test,batch_size=batch_size)

#PRECISION AND PIXEL VALUES IN THE RANGE FROM 0 T#O 1
tt_input_shape = [4,4,4,4,4,4]
# tt_output_shape = [2,4,2,4,2,4]
tt_output_shape = [5,5,5,5,5,5]
# tt_ranks = [8,8,8,8,8,8,8]
tt_ranks = [8,8,8,8,8,8,8]
print(stringOfHistory)
print(str([tt_input_shape,tt_output_shape,tt_ranks]))
train_vgg_tt(training_generator,testing_generator,steps=(steps_training,steps_testing),path=stringOfHistory,tt_parameters=[tt_input_shape,tt_output_shape,tt_ranks],nb_classes=nb_classes,batch_size=batch_size)
pass

