
from keras.layers import Input
from keras.models import Model
from keras.regularizers import l2
import keras
# from dense_tensor.example_utils import experiment
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class EnriquesCallback(Callback):

    def __init__(self,testing_generator,test_num_batches,stringOfHistory,lr,evaluate=True):
        self.testing_generator = testing_generator
        self.lr = lr
        self.test_num_batches = test_num_batches
        self.evaluate = evaluate
        self.stringOfHistory = stringOfHistory  
        if os.path.isfile(self.stringOfHistory + 'runObj.txt'):
            runInfo = cPickle.load(open(self.stringOfHistory+'runObj.txt','r'))
            self.history = runInfo['history']
            self.learningChanges = runInfo['learningChanges']
            self.counter = runInfo['counter']
        else:
            self.history = dict()
            self.history['lossTest'] = list()
            self.history['accuracyTest'] = list() 
            self.history['time'] = list()
            self.counter = 0    
            self.learningChanges = 0
         
    def on_train_begin(self,logs={}):
        self.threshold = 70
        self.t0 = time.time() #START TIMER ON EPOCH
    def on_epoch_end(self,epoch,logs={}):
        self.history['time'].append((time.time() - self.t0)/60.)
        if epoch == 1:
            K.set_value(self.model.optimizer.lr, self.lr*0.1)
        if self.evaluate:
            tmp = self.model.evaluate_generator(self.testing_generator,self.test_num_batches)
            # print('TEST LOSS : %.2f TEST ACCURACY : %.2f' % (tmp[0], tmp[1]))
            self.history['lossTest'].append(tmp[0])
            self.history['accuracyTest'].append(tmp[1])
        self.counter += 1 
        if self.counter > self.threshold:
            # print('TEST LOSS : %.2f TEST ACCURACY : %.2f' % (tmp[0], tmp[1]))
            # print('t loss :' + str(tmp[0]) + '  t acc :' + str(tmp[1]) + ' count :' + str(self.counter) + ' diff:' + str(np.mean(self.history['accuracyTest'][-10::]) - np.mean(self.history['accuracyTest'][-20:-10])))
            if(np.mean(self.history['accuracyTest'][-10::]) < np.mean(self.history['accuracyTest'][-20:-10])):
                self.threshold = 35
                print('\nLEARNING CHANGE')
                self.learningChanges += 1
                K.set_value(self.model.optimizer.lr, self.lr*(0.1**self.learningChanges))
                self.counter = 0
            if(self.learningChanges > 4):
                print('BREAK')
                self.model.stop_training = True
        if logs['acc'] >= 0.99:
            print('\nTRAINING STOPPED TO AVOID OVERFITTING')
            self.model.stop_training = True
        runObj = dict()
        runObj['history'] = self.history
        runObj['learningChanges'] = self.learningChanges
        runObj['counter'] = self.counter

        save_full_model(self.model,path=self.stringOfHistory)
        with open(self.stringOfHistory+'runObj.txt','w') as fp:
            cPickle.dump(runObj,fp)

def plot_history(history,savingFolder):
    if 'time' in history.keys():
        del history['time']
    for current_data_string in history.keys():
        plt.figure()
        plt.plot(history[current_data_string],lw=2,alpha=3)
        plt.xlabel('EPOCHS')
        if current_data_string[0].lower() == 'a':
            plt.ylabel('ACCURACY')
        else:
            plt.ylabel('LOSS')
        plt.savefig(savingFolder + current_data_string+ '.jpg',bbox_inches='tight',format='jpg',dmi=1000)

def save_full_model(model,history=None,path='./'):
  if history != None:
    with open(path + 'history.txt','w') as fp:
      cPickle.dump(history,fp)
  model.save(path+'model.h5')

def _generate_orthogonal_tt_cores(input_shape, output_shape, ranks):
    # Generate random orthogonalized tt-tensor.
        input_shape = np.array(input_shape)
        output_shape = np.array(output_shape)
        ranks = np.array(ranks)
        cores_arr_len = np.sum(input_shape * output_shape *
                               ranks[1:] * ranks[:-1])
        cores_arr = np.zeros(cores_arr_len).astype(K.floatx())
        cores_arr_idx = 0
        core_list = []
        rv = 1
        for k in range(input_shape.shape[0]):
            shape = [ranks[k], input_shape[k], output_shape[k], ranks[k+1]]
            tall_shape = (np.prod(shape[:3]), shape[3])
            curr_core = np.dot(rv, np.random.normal(0, 1, size=(shape[0], np.prod(shape[1:]))))
            curr_core = curr_core.reshape(tall_shape)
            if k < input_shape.shape[0]-1:
                curr_core, rv = np.linalg.qr(curr_core)
            cores_arr[cores_arr_idx:cores_arr_idx+curr_core.size] = curr_core.flatten()
            cores_arr_idx += curr_core.size
        # TODO: use something reasonable instead of this dirty hack.
        glarot_style = (np.prod(input_shape) * np.prod(ranks))**(1.0 / input_shape.shape[0])
        return (0.1 / glarot_style) * cores_arr.astype(K.floatx())

class TensorTrainDense(Layer):

    def __init__(self, tt_input_shape, tt_output_shape, tt_ranks,bias=True,**kwargs):
        self.output_dim = np.prod(tt_output_shape)
        super(TensorTrainDense, self).__init__(**kwargs)
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
        self.tt_output_shape = tt_output_shape
        self.tt_ranks = tt_ranks
        self.num_dim = tt_input_shape.shape[0]
        self.input_shape_x = None

        self.local_cores_arr = _generate_orthogonal_tt_cores(tt_input_shape,
                                                       tt_output_shape,
                                                       tt_ranks)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='cores_arr',shape=self.local_cores_arr.shape,trainable=True,initializer='uniform')
        self.set_weights([self.local_cores_arr])
        self.input_shape_x = input_shape
        num_units = np.prod(self.tt_output_shape)
        self.bias = self.add_weight(shape=(num_units,),
                                    initializer=keras.initializers.Constant(0),
                                    name='bias_cores_arr',
                                    regularizer=None)
        super(TensorTrainDense, self).build(input_shape)

    def call(self, x):
        res = x
        # TODO: it maybe faster to precompute the indices in advance.
        core_arr_idx = 0
        for k in range(self.num_dim - 1, -1, -1):
            curr_shape = [self.tt_input_shape[k] * self.tt_ranks[k + 1], self.tt_ranks[k] * self.tt_output_shape[k]]
            curr_core = K.reshape(self.kernel[core_arr_idx:core_arr_idx+np.prod(curr_shape).astype(int)],curr_shape)
            res = K.dot(K.reshape(res,(-1, curr_shape[0])), curr_core)
            res = K.transpose(K.reshape(res,(-1, self.tt_output_shape[k])))
            core_arr_idx += np.prod(curr_shape)
        res = K.transpose(K.reshape(res,(-1, self.input_shape_x[0])))
        if self.bias is not None:
            res = res + K.transpose(self.bias)
        return res

    def get_config(self):
        base_config = super(TensorTrainDense, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def get_vgg_style_net(tt_parameters,input_shape=(32,32,3),blocks=[3,3],outNeurons=512,init_filters=16,dropout=False,nb_classes=10,weightDecay=10.e-3):
  inputs = Input(batch_shape=input_shape)
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

def out_block_low_rank_representation(node,tt_parameters,nb_classes=10,weightDecay=10.e-3,outNeurons=512):
    x = TensorTrainDense(tt_parameters[0], tt_parameters[1], tt_parameters[2])(node)
    # x = Activation('relu')(x)
    x = Dense(outNeurons,kernel_regularizer=l2(weightDecay),activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(outNeurons/2,kernel_regularizer=l2(weightDecay),activation='relu')(x)
    out = Dense(nb_classes,kernel_regularizer=l2(weightDecay),activation='softmax')(x)
    return out
