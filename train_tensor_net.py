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
from keras.layers.normalization import BatchNormalization
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

def get_vgg_style_net_tt(tt_parameters,input_shape=(32,32,3),blocks=[3,3],outNeurons=512,init_filters=16,dropout=False,nb_classes=10,weightDecay=10.e-3):
  inputs = Input(shape=input_shape)
  x = Conv2D(init_filters,3,kernel_regularizer=l2(weightDecay),padding='same',activation='relu')(inputs)
  blocks[0] -= 1
  for current_block in blocks:
    for current_layer_in_block in range(current_block):
      x = Conv2D(init_filters,3,kernel_regularizer=l2(weightDecay),padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    init_filters *= 2
    if dropout:
      x = Dropout(0.5)(x)
  x = Flatten()(x)
  x = out_block_low_rank_representation(x,tt_parameters=tt_parameters,weightDecay=weightDecay,outNeurons=outNeurons,nb_classes=nb_classes)

  return Model(inputs=inputs,outputs=x)

def out_block_low_rank_representation(node,tt_parameters,nb_classes=10,weightDecay=10.e-4,outNeurons=512):
    x = TensorTrainDense2(tt_parameters[0], tt_parameters[1], tt_parameters[2])(node)
    # x = Flatten()(x)
    # for i in range(len(tt_parameters[1])):
    # x = BatchNormalization(axis=1)(x)
    # x = Flatten()(x)
    x = Activation('relu')(x)
    x = Dense(outNeurons,kernel_regularizer=l2(weightDecay),activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(outNeurons/2,kernel_regularizer=l2(weightDecay),activation='relu')(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weightDecay),activation='softmax')(x)
    return x


def _generate_orthogonal_tt_cores(input_shape, output_shape, ranks):
    # Generate random orthogonalized tt-tensor.
        input_shape = np.array(input_shape)
        output_shape = np.array(output_shape)
        ranks = np.array(ranks)
        cores_arr_len = np.sum(input_shape * output_shape * ranks[1:] * ranks[:-1])
        cores_arr = np.zeros(cores_arr_len).astype(K.floatx())
        cores_arr_idx = 0
        core_list = []
        middle_cores_list = []
        glarot_style = (np.prod(input_shape) * np.prod(ranks))**(1.0 / input_shape.shape[0])
        # glarot_style = 0.1
        rv = 1
        for k in range(input_shape.shape[0]):
            shape = [ranks[k], input_shape[k], output_shape[k], ranks[k+1]]
            curr_shape = (input_shape[k] * ranks[k + 1], ranks[k] * output_shape[k])
            tall_shape = (np.prod(shape[:3]), shape[3])
            curr_core = np.dot(rv, np.random.normal(0, 1, size=(shape[0], np.prod(shape[1:]))))
            curr_core = curr_core.reshape(tall_shape)
            if k < input_shape.shape[0]-1:
                curr_core, rv = np.linalg.qr(curr_core)
            # if np.mod(k,2) == 1:
            if k == 0 or k == input_shape.shape[0]-1:
                core_list.append((0.1 / glarot_style) * curr_core.astype(K.floatx()).reshape(curr_shape))
            else:
                middle_cores_list.append((0.1 / glarot_style) * curr_core.astype(K.floatx()).reshape(curr_shape))
            # cores_arr[cores_arr_idx:cores_arr_idx+curr_core.size] = curr_core.flatten()
            # cores_arr_idx += curr_core.size
        # TODO: use something reasonable instead of this dirty hack.
        #NORMALIZE CORES
        # for i in range(len(core_list)):
        #     mean = np.mean(core_list[i].flatten())
        #     std = np.std(core_list[i].flatten())
        #     core_list[i] -= mean
        #     core_list[i] /= std
        # for i in range(len(middle_cores_list)):
        #     mean = np.mean(middle_cores_list[i].flatten())
        #     std = np.std(middle_cores_list[i].flatten())
        #     middle_cores_list[i] -= mean
        #     middle_cores_list[i] /= std
        core_list.append(np.asarray(middle_cores_list).astype(K.floatx()))
        # flattened_cores = np.concatenate([core_list[0].flatten(),core_list[1].flatten(),core_list[2].flatten()]).flatten()
        # mean = np.mean(flattened_cores)
        # std = np.std(flattened_cores)

        # for i in range(len(core_list)):
        #     # mean = np.mean
        #     core_list[i] -= mean
        #     core_list[i] /= std

        return core_list
class TensorTrainDense2(Layer):

    def __init__(self, tt_input_shape, tt_output_shape, tt_ranks,bias=True,**kwargs):
        self.output_dim = np.prod(tt_output_shape)
        super(TensorTrainDense2, self).__init__(**kwargs)
        num_inputs = int(np.prod(tt_input_shape))
        tt_input_shape = np.asarray(tt_input_shape)
        tt_output_shape = np.asarray(tt_output_shape)
        tt_ranks = np.asarray(tt_ranks)
        # if np.prod(tt_input_shape) != num_inputs:
        #     raise ValueError("The size of the input tensor (i.e. product "
        #                      "of the elements in tt_input_shape) should "
        #                      "equal to the number of input neurons %d." %
        #                      (num_inputs))
        if tt_input_shape.shape[0] != tt_output_shape.shape[0]:
            raise ValueError("The number of input and output dimensions "
                             "should be the same.")
        if tt_ranks.shape[0] != tt_output_shape.shape[0] + 1:
            raise ValueError("The number of the TT-ranks should be "
                             "1 + the number of the dimensions.")
        self.tt_input_shape = tt_input_shape
        self.tt_output_shape = tt_output_shape.tolist()
        self.tt_ranks = tt_ranks
        self.num_dim = tt_input_shape.shape[0]
        self.input_shape_x = None

        cores = _generate_orthogonal_tt_cores(tt_input_shape,
                                                       tt_output_shape,
                                                       tt_ranks)
        self.first_core = cores[0]
        self.last_core = cores[1]
        self.middle_cores = cores[2]
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.middle_kernel = self.add_weight(name='middle_cores',shape=self.middle_cores.shape,trainable=True,initializer='uniform')
        self.first_kernel = self.add_weight(name='first_core',shape=self.first_core.shape,trainable=True,initializer='uniform')
        self.last_kernel = self.add_weight(name='last_core',shape=self.last_core.shape,trainable=True,initializer='uniform')
        self.set_weights([self.middle_cores,self.first_core,self.last_core])
        self.input_shape_x = input_shape
        self.num_units = np.prod(self.tt_output_shape)
        self.bias = self.add_weight(shape=(self.num_units,),
                                    initializer=keras.initializers.Constant(0),
                                    name='bias_cores_arr',
                                    regularizer=None,
                                    trainable=True)
        super(TensorTrainDense2, self).build(input_shape)

    def call(self, x):
        res = x
        res = K.reshape(res,(K.shape(x)[0],-1,self.last_kernel.shape[0].value))
        res = K.dot(res,self.last_kernel)
        res = K.reshape(res,(K.shape(x)[0],self.tt_output_shape[-1], -1))
        res = K.reshape(res,(K.shape(x)[0],-1,self.middle_kernel[-1].shape[0].value))
        res = K.dot(res,self.middle_kernel[-1])
        res = K.reshape(res,(K.shape(x)[0], self.tt_output_shape[-2],-1))
        res = K.reshape(res,(K.shape(x)[0],-1,self.middle_kernel[-2].shape[0].value))
        res = K.dot(res,self.middle_kernel[-2])
        res = K.reshape(res,(K.shape(x)[0], self.tt_output_shape[-3],-1))
        res = K.reshape(res,(K.shape(x)[0],-1,self.middle_kernel[-3].shape[0].value))
        res = K.dot(res,self.middle_kernel[-3])
        res = K.reshape(res,(K.shape(x)[0], self.tt_output_shape[-4],-1))
        res = K.reshape(res,(K.shape(x)[0],-1,self.middle_kernel[-4].shape[0].value))
        res = K.dot(res,self.middle_kernel[-4])
        res = K.reshape(res,(K.shape(x)[0], self.tt_output_shape[-5],-1))
        res = K.reshape(res,(K.shape(x)[0],-1,self.middle_kernel[-5].shape[0].value))
        res = K.dot(res,self.middle_kernel[-5])
        res = K.reshape(res,(K.shape(x)[0], self.tt_output_shape[-6],-1))
        res = K.reshape(res,(K.shape(x)[0],-1,self.first_kernel.shape[0].value))
        res = K.dot(res,self.first_kernel)
        res = K.reshape(res,(K.shape(x)[0], self.tt_output_shape[-7],-1))

        res = K.batch_flatten(res)

        if self.bias is not None:
            res = K.bias_add(res, self.bias)

        # res = K.reshape(res,(K.shape(x)[0],5,5,5,5,5,5,5))
        return res

    def get_config(self):
        base_config = super(TensorTrainDense2, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        # return [input_shape[0]] + self.tt_output_shape
        return (input_shape[0],self.num_units)

def train_vgg_tt(training_generator,testing_generator,steps,path,tt_parameters,nb_classes=10,batch_size=128):
    if not os.path.isdir(path):
        os.mkdir(path)
    infoCallback = EnriquesCallback(testing_generator,steps[1],path,0.01,evaluate=True)
    save_best_callback = ModelCheckpoint(path+'best_model.h5',monitor='val_acc',verbose=1,mode='max',save_best_only=True)
    model = get_vgg_style_net_tt(tt_parameters,input_shape=(32,32,3),blocks=[3,3],outNeurons=512,init_filters=128,dropout=False,nb_classes=nb_classes,weightDecay=0)
    # x = model.predict(training_generator.next()[0])
    sgd = SGD(lr=0.01, momentum=0.9)
    if os.path.isfile(path +'model.h5'):
        model.load_weights(path+'model.h5', by_name=True)
    #datagen.flow(testing_data[0],testing_data[1],batch_size=128)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
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
                             zca_whitening=False,  #WHITENING
                             rotation_range=False,  # randomly rotate images in the range (degrees, 0 to 180)
                             width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                             horizontal_flip=False,  # randomly flip images
                             vertical_flip=False)

datagen.fit(X_train)

training_generator = datagen.flow(X_train,Y_train,batch_size=batch_size)
testing_generator = datagen.flow(X_test,Y_test,batch_size=batch_size)

#PRECISION AND PIXEL VALUES IN THE RANGE FROM 0 T#O 1
tt_input_shape = [4,4,4,4,4,4,4]
# tt_output_shape = [2,4,2,4,2,4]
tt_output_shape = [5,5,5,5,5,5,5]
# tt_ranks = [8,8,8,8,8,8,8]
tt_ranks = [1,8,8,8,8,8,8,1]
print(stringOfHistory)
print(str([tt_input_shape,tt_output_shape,tt_ranks]))
train_vgg_tt(training_generator,testing_generator,steps=(steps_training,steps_testing),path=stringOfHistory,tt_parameters=[tt_input_shape,tt_output_shape,tt_ranks],nb_classes=nb_classes,batch_size=batch_size)
pass

